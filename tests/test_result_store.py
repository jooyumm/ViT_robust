"""JSON 결과 교체, 실패 시 이전 결과 유지, seed·tag 분리 검사."""
from pathlib import Path
import json
import sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_patch_size_tradeoff'))
from result_store import read_results,save_result


class ResultStore(unittest.TestCase):
    def record(self, seed=42, tag='', ca=80.):
        return dict(attack='patch_fool',patch_size=8,seed=seed,tag=tag,metrics=dict(CA=ca,RA=30.,ASR=62.5))

    def test_replace_same_condition_preserves_other_seeds_and_tags(self):
        with tempfile.TemporaryDirectory() as t:
            save_result(t,self.record());save_result(t,self.record(seed=123));save_result(t,self.record(tag='area'))
            path=save_result(t,self.record(ca=90.))
            self.assertEqual(len(list(Path(t).iterdir())),3)
            self.assertEqual(json.loads(path.read_text())['metrics']['CA'],90.)
            self.assertEqual(len(read_results(t)),3)
            self.assertEqual(len(read_results(t,{''})),2)

    def test_invalid_new_result_does_not_replace_last_good_result(self):
        with tempfile.TemporaryDirectory() as t:
            path=save_result(t,self.record());old=path.read_bytes()
            with self.assertRaises(ValueError):save_result(t,self.record(ca=float('nan')))
            self.assertEqual(path.read_bytes(),old)
            self.assertFalse(list(Path(t).glob('*.pending')))

    def test_tag_cannot_escape_output_directory(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):save_result(t,self.record(tag='../escape'))
            self.assertEqual(list(Path(t).iterdir()),[])
