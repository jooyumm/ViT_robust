"""동일 500장·P16 한 패치 위협에서 위치 ablation과 hard-routing 적응형 공격을 비교한다.

Random은 detector 발동 여부를 유지하고, 발동 시 4개 사분면의 uniform 기대값을
정확히 계산한다. Adaptive는 기존 vanilla 공격에서 시작한 추가 최적화다.
경로는 매번 실제 detector로 재계산하고 선택된 경로를 미분한다. BPDA/EOT는 아니다.
진행 로그 대신 최종/진행 JSON 하나와 분리된 PNG 그림을 갱신하고 공격 이미지는 data/에 보관한다.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

BASE=Path(__file__).resolve().parent.parent
ROOT=BASE.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'02_detection_defense')]
from src.models import load_vit_model
from src.data.paths import PATCHFOOL
from src.attacks import patch_fool_attack
from src.provenance import model_hash,sha256
from src.patches import coverage_from_images,patch_to_region
from src.data.protocol import validate_attack_manifest,vanilla_p16_config
from region_switch.core.inference import predict_defended,forward_defended,verify_transform
from region_switch.core.attention import PatchAttentionDetector
from region_switch.core.switch import converted_patch_kernel,build_region_swap_batch,forward_with_replacement_mass


def atomic_json(path,record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.pending')
    tmp.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');tmp.replace(path)


class BestCandidate:
    """실제 방어 오분류를 우선하고, 동률이면 표본별 CE가 큰 후보를 보존한다."""
    def __init__(self,images,labels):
        self.images=images.detach().clone();self.labels=labels
        self.success=torch.zeros(len(images),device=images.device,dtype=torch.bool)
        self.loss=torch.full((len(images),),-float('inf'),device=images.device)

    @torch.no_grad()
    def __call__(self,images,logits):
        loss=F.cross_entropy(logits,self.labels,reduction='none')
        if not bool(torch.isfinite(loss).all()):raise ValueError('Non-finite adaptive candidate')
        success=logits.argmax(-1)!=self.labels
        improve=(success & ~self.success)|((success==self.success)&(loss>self.loss))
        self.images[improve]=images[improve];self.loss[improve]=loss[improve];self.success[improve]=success[improve]


@torch.inference_mode()
def score_inputs(model,kernel,detector,images,device,batch_size,*,random_regions):
    """이미지 전용 정책 평가. GT·label을 받지 않는다."""
    predictions=[];defended=[];random=[];decisions={k:[] for k in ['flag','region','score','patch']}
    for start in range(0,len(images),batch_size):
        x=torch.from_numpy(images[start:start+batch_size]).to(device)
        base,output,decision=predict_defended(model,kernel,x,detector)
        pred=base.argmax(-1).cpu().numpy();predictions.append(pred)
        defended.append(output.argmax(-1).cpu().numpy())
        for key in decisions:decisions[key].append(decision[key])
        if random_regions:
            choices=np.repeat(pred[:,None],4,axis=1)
            active=np.flatnonzero(decision['flag'])
            if len(active):
                for region in range(4):
                    sequence=build_region_swap_batch(model,kernel,x[active],torch.full((len(active),),region,device=device))
                    choices[active,region]=forward_with_replacement_mass(model,sequence).argmax(-1).cpu().numpy()
            random.append(choices)
    return dict(plain=np.concatenate(predictions),detected=np.concatenate(defended),
                random=np.concatenate(random) if random_regions else None,
                decision={k:np.concatenate(v) for k,v in decisions.items()})


def make_row(key,title,labels,clean,attacked,*,policy,budget,gt):
    clean_ok=clean['plain']==labels
    def correct(result):
        pred=result['plain'] if policy=='none' else result[policy]
        return (pred==labels[:,None]).mean(1) if pred.ndim==2 else (pred==labels).astype(float)
    clean_policy,attack_policy=correct(clean),correct(attacked)
    initial_wrong=clean_ok & (attacked['plain']!=labels)
    flags=attacked['decision']['flag'].astype(bool)
    hit=(attacked['decision']['region']==patch_to_region(gt))
    return dict(key=key,title=title,n=len(labels),attack_budget=budget,
                clean_accuracy=float(clean_ok.mean()),
                attack_accuracy=float((attacked['plain']==labels).mean()),
                defended_accuracy=float(attack_policy.mean()),
                clean_policy_accuracy=float(clean_policy.mean()),
                asr_after_defense=float((1-attack_policy[clean_ok]).mean()) if clean_ok.any() else None,
                recovery_rate=float(attack_policy[initial_wrong].mean()) if initial_wrong.any() else None,
                tpr=float(flags.mean()) if policy!='none' else None,
                fpr=float(clean['decision']['flag'].mean()) if policy!='none' else None,
                flagged_region_accuracy=(.25 if policy=='random' else float(hit[flags].mean())) if policy!='none' and flags.any() else None)


def render_figs(report,path):
    from region_switch.visualize import render_summary
    baseline_path=BASE/'results/comparison.json'
    baseline=json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    render_summary(report,BASE/'results',baseline)


def read_source(run,manifest,limit):
    keys=['sample_index','label','clean__images','patchfool__images','patchfool__gt_coverage','clean__prediction','patchfool__prediction']
    data={k:[] for k in keys}
    for chunk in manifest['chunks']:
        path=(run/chunk['path']).resolve()
        if run.resolve() not in path.parents or sha256(path)!=chunk['sha256']:raise ValueError('Source chunk hash/path mismatch')
        with np.load(path,allow_pickle=False) as z:
            for key in keys:data[key].append(z[key])
    data={k:np.concatenate(v)[:limit] for k,v in data.items()}
    np.testing.assert_array_equal(data['sample_index'],manifest['sample_indices'][:limit])
    return data


def main(args,report):
    manifest=json.loads((args.run/'manifest.json').read_text());validate_attack_manifest(manifest)
    if not 1<=args.limit<=len(manifest['sample_indices']) or args.steps<1 or args.batch_size<1:raise ValueError('Invalid sample/step/batch limit')
    torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    model=load_vit_model(16,args.device,strict=True).requires_grad_(False)
    original_hash=model_hash(model)
    if original_hash!=manifest['model_sha256']:raise ValueError('Model hash mismatch')
    kernel=converted_patch_kernel(model);detector=PatchAttentionDetector()
    data=read_source(args.run,manifest,args.limit);n=len(data['label']);labels=data['label'];gt=data['patchfool__gt_coverage'].argmax(1)
    verify_transform(model,kernel,torch.from_numpy(data['clean__images'][:1]).to(args.device))
    report.update(state='RUNNING',stage='nonadaptive',n=n,rows=[],job_id=os.environ.get('SLURM_JOB_ID'),attack_archive=str(args.run.resolve()),
                  source_manifest_sha256=sha256(args.run/'manifest.json'),model_sha256=original_hash,
                  adaptive_scope='fixed original P16 patch; exact hard detector routing each iteration; gradients through selected branch only; no BPDA/EOT',
                  random_policy='same detector flag; exact expectation over 4 uniformly selected quadrants',
                  clean_definition='undefended original P16; clean_policy_accuracy is separately named',
                  steps=args.steps,seed=args.seed,batch_size=args.batch_size)
    atomic_json(args.output.with_suffix('.json'),report)
    clean=score_inputs(model,kernel,detector,data['clean__images'],args.device,args.batch_size,random_regions=True)
    attacked=score_inputs(model,kernel,detector,data['patchfool__images'],args.device,args.batch_size,random_regions=True)
    np.testing.assert_array_equal(clean['plain'],data['clean__prediction'])
    np.testing.assert_array_equal(attacked['plain'],data['patchfool__prediction'])
    for key,title,policy in [('none','No defense','none'),('detected','Detected region','detected'),('random','Random region (same trigger)','random')]:
        report['rows'].append(make_row(key,title,labels,clean,attacked,policy=policy,budget='vanilla 250',gt=gt))
    if not args.adaptive:
        if model_hash(model)!=original_hash:raise ValueError('Model weights changed')
        report.update(state='COMPLETED',stage='published',unchanged_pretrained_weights=True)
        atomic_json(args.output.with_suffix('.json'),report);render_figs(report,args.output)
        return
    report.update(stage='adaptive',adaptive_completed=0)
    atomic_json(args.output.with_suffix('.json'),report);render_figs(report,args.output)
    config=vanilla_p16_config();config['train_attack_iters']=args.steps
    files=[Path(__file__),BASE/'core/inference.py',BASE/'core/switch.py',BASE/'core/attention.py',
           ROOT/'src/attacks/patch_fool.py',ROOT/'src/attention.py',ROOT/'src/attack_utils.py',
           ROOT/'src/models.py',ROOT/'src/patches.py',ROOT/'src/data/protocol.py']
    sources={str(p.relative_to(ROOT)):sha256(p) for p in files}
    contract=dict(schema_version=1,attack_target='region_switch_v1_hard_routing',defense_aware=True,
                  model_sha256=original_hash,source_manifest_sha256=report['source_manifest_sha256'],source_sha256=sources,
                  sample_indices=data['sample_index'].tolist(),patchfool_protocol=config,
                  seed=args.seed,batch_size=args.batch_size,warm_start='vanilla P16 archive; same selected patch',
                  fixed_patch_source='warm-start pixel support, known to attacker; never passed to defense',
                  candidate_selection='best actual-defense misclassification, then highest per-image CE; includes warm start',
                  additional_steps=args.steps,total_gradient_steps=250+args.steps,
                  limitation=report['adaptive_scope'])
    args.adaptive_run.mkdir(parents=True,exist_ok=True);am=args.adaptive_run/'manifest.json'
    if am.exists():
        adaptive_manifest=json.loads(am.read_text())
        if any(adaptive_manifest.get(k)!=v for k,v in contract.items()):raise ValueError('Adaptive archive protocol changed; use another archive path')
    else:
        adaptive_manifest=dict(contract,complete=False,chunks=[]);atomic_json(am,adaptive_manifest)
    adaptive_images=np.empty_like(data['patchfool__images']);offset=0
    for chunk in adaptive_manifest['chunks']:
        path=args.adaptive_run/chunk['path']
        if sha256(path)!=chunk['sha256']:raise ValueError('Adaptive chunk hash mismatch')
        with np.load(path,allow_pickle=False) as z:
            count=len(z['images']);np.testing.assert_array_equal(z['sample_index'],data['sample_index'][offset:offset+count])
            adaptive_images[offset:offset+count]=z['images'];offset+=count
    started=time.monotonic();initial_offset=offset
    for start in range(offset,n,args.batch_size):
        stop=min(n,start+args.batch_size)
        x=torch.from_numpy(data['clean__images'][start:stop]).to(args.device)
        warm=torch.from_numpy(data['patchfool__images'][start:stop]).to(args.device)
        y=torch.from_numpy(labels[start:stop]).to(args.device)
        torch.manual_seed(args.seed+1009*start)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed+1009*start)
        best=BestCandidate(warm,y)
        def objective(images):return forward_defended(model,kernel,images,detector)[1]
        with torch.no_grad():best(warm,objective(warm))
        patch_fool_attack(model,x,y,args.device,**config,classification_forward=objective,
                          initial_delta=warm-x,candidate_callback=best,
                          selected_patches=torch.as_tensor(gt[start:stop,None],device=args.device,dtype=torch.long))
        candidate=best.images.detach()
        with torch.no_grad():
            final=objective(candidate)
            torch.testing.assert_close(final.argmax(-1)!=y,best.success)
        cov=coverage_from_images(x,candidate,16)
        if not bool(((candidate>=-1)&(candidate<=1)).all()):raise ValueError('Adaptive pixels out of range')
        if not np.all((cov>0).sum(1)==1):raise ValueError('Adaptive attack must affect exactly one P16 patch')
        np.testing.assert_array_equal(cov.argmax(1),gt[start:stop])
        if any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()):raise ValueError('Leaked attack/defense hooks')
        images=candidate.cpu().numpy();adaptive_images[start:stop]=images
        path=args.adaptive_run/f'chunk_{start:06d}.npz';tmp=path.with_suffix('.pending')
        with tmp.open('wb') as stream:np.savez_compressed(stream,sample_index=data['sample_index'][start:stop],images=images,coverage=cov)
        tmp.replace(path)
        adaptive_manifest['chunks'].append(dict(path=path.name,n=stop-start,sha256=sha256(path)))
        atomic_json(am,adaptive_manifest)
        elapsed=time.monotonic()-started
        report.update(adaptive_completed=stop,adaptive_elapsed_seconds=elapsed,
                      adaptive_eta_seconds=elapsed/(stop-initial_offset)*(n-stop),adaptive_archive=str(args.adaptive_run.resolve()))
        atomic_json(args.output.with_suffix('.json'),report)
    adaptive_manifest['complete']=True;atomic_json(am,adaptive_manifest)
    report.update(stage='adaptive_evaluation');atomic_json(args.output.with_suffix('.json'),report)
    adaptive=score_inputs(model,kernel,detector,adaptive_images,args.device,args.batch_size,random_regions=False)
    row=make_row('adaptive','Adaptive → detected region',labels,clean,adaptive,policy='detected',
                 budget=f'vanilla 250 + adaptive {args.steps} extra; best candidate',gt=gt)
    if row['defended_accuracy']>report['rows'][1]['defended_accuracy']+1e-12:raise ValueError('Best adaptive candidate lost a warm-start attack success')
    report['rows'].append(row)
    if model_hash(model)!=original_hash:raise ValueError('Model weights changed')
    report.update(state='COMPLETED',stage='published',unchanged_pretrained_weights=True,adaptive_archive=str(args.adaptive_run.resolve()))
    atomic_json(args.output.with_suffix('.json'),report);render_figs(report,args.output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=PATCHFOOL)
    parser.add_argument('--adaptive',action='store_true',help='추가 적응형 공격을 명시적으로 실행')
    parser.add_argument('--adaptive-run',type=Path,default=None,help='--adaptive 실행 시 새 공격 데이터셋 저장 경로')
    parser.add_argument('--output',type=Path,default=BASE/'results/defense_comparison.json')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--steps',type=int,default=250)
    parser.add_argument('--seed',type=int,default=21)
    parser.add_argument('--limit',type=int,default=500,help='smoke test only; main comparison uses all 500')
    args=parser.parse_args()
    if args.adaptive and args.adaptive_run is None:parser.error('--adaptive에는 --adaptive-run 경로가 필요합니다.')
    report={}
    try:main(args,report)
    except BaseException as error:
        report.update(state='FAILED',error=repr(error))
        atomic_json(args.output.with_suffix('.json'),report)
        raise
