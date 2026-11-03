"""02와 동일한 저장 공격본으로 01 전체 결과 갱신. 새 공격 생성 없이 중간 결과를 검증 후 게시한다."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from src.data.paths import PATCHFOOL,PAIRED_ATTENTION
from src.provenance import sha256
from src.cohort import validate_default_manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=parser.parse_args()
    stage=ROOT/'data/attention/.official_refresh';stage.mkdir(parents=True,exist_ok=True)
    status=stage/'status.json';state=dict(state='RUNNING',job_id=os.environ.get('SLURM_JOB_ID'),stage='starting',started=time.time(),attack_source=str(PATCHFOOL))
    def save():
        tmp=status.with_suffix('.pending');tmp.write_text(json.dumps(state,indent=2));tmp.replace(status)
    def step(name,entry,*argv):
        state.update(stage=name);save()
        completed=subprocess.run([sys.executable,str(ROOT/entry),*map(str,argv)],cwd=ROOT,stderr=subprocess.PIPE,text=True)
        if completed.returncode:raise RuntimeError(name+': '+completed.stderr[-5000:])
    try:
        save()
        if PAIRED_ATTENTION.exists():
            meta=json.loads((PAIRED_ATTENTION/'manifest.json').read_text())
            if not meta.get('complete'):
                shutil.rmtree(PAIRED_ATTENTION)
        if not PAIRED_ATTENTION.exists():
            step('extract_attention','src/data/extract_attention.py','--device',args.device,'--output',PAIRED_ATTENTION)
        validate_default_manifest(json.loads((PAIRED_ATTENTION/'manifest.json').read_text()))
        state.update(stage='verify_identical_pixels');save()
        from src.data.paired import load_images,load_paired_dataset
        import numpy as np
        original,source=load_images(PATCHFOOL);meta,cache=load_paired_dataset(PAIRED_ATTENTION)
        for key in ['sample_index','label','clean__images','patchfool__images','patchfool__gt_coverage']:
            np.testing.assert_array_equal(source[key],cache[key])
        # Prediction identity is also required; different numeric paths may change raw logits slightly.
        offset=0
        for chunk in original['chunks']:
            with np.load(PATCHFOOL/chunk['path']) as z:
                n=chunk['n']
                for key in ['clean__prediction','patchfool__prediction']:
                    np.testing.assert_array_equal(z[key],cache[key][offset:offset+n])
                offset+=n
        state.update(verified_identical_pixels=True,source_manifest_sha256=sha256(PATCHFOOL/'manifest.json'));save()
        del source,cache
        results=stage/'results'
        if results.exists():shutil.rmtree(results)
        step('01_attention_maps','01_attention_characterization/experiments/01_attention_maps/compare_attention.py','--output',results/'01_attention_maps')
        step('01_success_split','01_attention_characterization/experiments/01_attention_maps/patchfool_success_split.py','--output',results/'01_attention_maps/patchfool')
        step('02_layer_profiles','01_attention_characterization/experiments/02_layer_profiles/final_peak.py','--output',results/'02_layer_profiles')
        flow=stage/'region_flow'
        if flow.exists():shutil.rmtree(flow)
        step('03_extract_flow','01_attention_characterization/experiments/03_region_attention/extract_region_flow.py','--device',args.device,'--output-root',flow)
        step('03_region_attention','01_attention_characterization/experiments/03_region_attention/analyze_regions.py','--data',flow/'extracted/flow.npz','--output',results/'03_region_attention')
        step('03_outcomes','01_attention_characterization/experiments/03_region_attention/outcome_breakdown.py','--output',results/'03_region_attention/patchfool/00_outcome_breakdown.png')
        # Publish only a complete set; remove the superseded projected results rather than mixing them.
        previous=stage/'previous_results';target=HERE/'results'
        if previous.exists():shutil.rmtree(previous)
        if target.exists():target.rename(previous)
        try:results.rename(target)
        except BaseException:
            if previous.exists():previous.rename(target)
            raise
        if previous.exists():shutil.rmtree(previous)
        target_flow=HERE/'experiments/03_region_attention/extracted'
        if target_flow.exists():shutil.rmtree(target_flow)
        (flow/'extracted').rename(target_flow)
        (target/'README.md').write_text('# Shared official PatchFool dataset\n\nAll 01 outputs use exactly the same stored 500-image attack as 02.\nSource: '+str(PATCHFOOL)+'\nSource manifest SHA256: '+sha256(PATCHFOOL/'manifest.json')+'\n')
        state.update(state='COMPLETED',stage='published',finished=time.time());save()
    except BaseException as error:
        state.update(state='FAILED',error=repr(error),finished=time.time());save();raise


if __name__=='__main__':main()
