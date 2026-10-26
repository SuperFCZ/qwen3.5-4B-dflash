"""CPU tests only: never assert hardware support."""
import sys
import unittest
import tempfile
from pathlib import Path
import json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stage2a import dot,convert,encode_doc_candidate,make_cases,sdk_index

class TestStage2A(unittest.TestCase):
    def test_raw_dot(self):
        a=np.full((16,128),-128,dtype=np.int8)
        b=np.full((128,64),-128,dtype=np.int8)
        z=dot(a,b)
        self.assertTrue(np.all(z==2097152))
        with self.assertRaises(ValueError): dot(a.astype(np.float16),b)

    def test_codec_candidate_only(self):
        s=np.ones(64,dtype='<f4');s[1]=0.5;s[2]=1.0009;s[3]=0
        words,eff=encode_doc_candidate(s)
        self.assertEqual(int(words[0]),(1<<46)|0x3f800000)
        self.assertEqual(int(words[1]),(1<<46)|0x3f000000)
        self.assertEqual(float(eff[2]),1.)
        self.assertEqual(float(eff[3]),0.)
        with self.assertRaises(ValueError):encode_doc_candidate(np.full(64,-1,dtype='<f4'))

    def test_fixture_outputs(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'data';m=make_cases(p)
            self.assertEqual(len(m['cases']),5)
            d=p/'per_channel_powers'
            a=np.fromfile(d/'a_i8.bin',dtype='i1').reshape(16,128)
            b=np.fromfile(d/'b_i8.bin',dtype='i1').reshape(128,64)
            z=np.fromfile(d/'z_i32.bin',dtype='<i4').reshape(16,64)
            self.assertTrue(np.array_equal(z,dot(a,b)))
            s=np.fromfile(d/'scale_f32.bin',dtype='<f4')
            ref=np.fromfile(d/'ideal_fp16.bin',dtype='<f2').reshape(16,64)
            self.assertTrue(np.array_equal(ref.view('u2'),convert(z.astype(np.int64),s).view('u2')))
            self.assertEqual(json.loads((p/'manifest.json').read_text())['status'],'CPU_MODEL_ONLY_NOT_NPU_VERIFIED')
            with self.assertRaises(FileExistsError):make_cases(p)

    def test_sdk_symbols_do_not_prove_support(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'cann';asc=root/'aarch64-linux'/'asc'
            p=asc/'impl/adv_api/detail/matmul/stage/copy_cube_out/quant/quant_processor_datacopy.h'
            p.parent.mkdir(parents=True);p.write_text('private line\nSetQuantVector\n')
            (root/'version.info').write_text('9.0.0')
            res=sdk_index(root)
            hits=[x for x in res['files'] if x['state']=='INDEXED_NOT_REVIEWED']
            self.assertEqual(hits[0]['hits']['SetQuantVector'],[2])
            self.assertTrue(all(x['status']=='UNRESOLVED' for x in res['claims']))
            self.assertNotIn('private line',json.dumps(res))
            with self.assertRaises(ValueError):sdk_index(root,('../outside',))

if __name__=='__main__':unittest.main()
