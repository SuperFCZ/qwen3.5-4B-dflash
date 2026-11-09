"""CPU contracts only. No test equates a model or compilation with hardware behavior."""
import struct,sys,subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import vdeq_common as c
import tiling_decode as td
import vdeq_sdk as sdk
import schema_tool as st
import run_vdeq_review as run
from vdeq_analysis import analyze


def test_mcb_sign_separation_and_no_mutation():
 word=0x3f000000|(1<<46);f=c.models.word_fields(word)
 assert f['mcb']==0 and f['sign_bit46']==1 and f['n_raw']==0 and f['multiplier']==.5
 f=c.models.word_fields(word|(1<<36)|(3<<32))
 assert f['mcb']==1 and f['sign_bit46']==1 and f['n_documented_value']==4
 assert word==(0x3f000000|(1<<46))


@pytest.mark.parametrize('dot',[2049,-2049,2051,-2051,0,2,-2,-75057,-83821,-67885,-89917,-70957,-73525])
def test_midpoint_sign_parity_models(dot):
 scale=np.array([2**-11],'<f4').view('<u4')[0];prediction=c.models.predict(dot,int(scale))
 assert prediction['A_original']==int((np.float32(dot)*np.float32(2**-11)).astype('<f2').view('<u2'))
 assert c.models.midpoint_info(dot,int(scale))['int32_exact_in_fp32']
 if abs(dot) in (2049,2051):assert c.models.midpoint_info(dot,int(scale))['exact_midpoint']
 assert c.models.DEFINITIONS['B_even_toward_zero'].endswith('hypothesis')


def test_frozen_shape_analysis_all_elements():
 dots=(np.arange(4096,dtype=np.int64)-2048).reshape(16,256)
 m=np.array([2.**(-12+i%6) for i in range(256)],'<f4').view('<u4')
 words=m.astype('<u8')|np.uint64(1<<46);saved=words.tobytes()
 expected=(dots.astype('<f4')*m.view('<f4')).astype('<f2').view('<u2');actual=expected.copy();actual[0,84]+=1
 summary,rows,bad=analyze(dots,words,actual,expected)
 assert len(rows)==4096 and len(bad)==1 and len(summary['key_positions'])==6
 assert rows[84]['ulp_vs_original_cpu']==1 and rows[84]['absolute_error_vs_original_cpu']>0
 assert summary['new_vdeq16_execution']=='NOT_RUN' and summary['cause']=='UNRESOLVED'
 assert words.tobytes()==saved


def test_illegal_frozen_words_not_silently_admitted():
 dots=np.zeros((16,256),np.int64);words=np.full(256,0x3f800000|(1<<46),'<u8');out=np.zeros((16,256),'<u2')
 words[0]|=np.uint64(1<<36)
 with pytest.raises(ValueError,match='not frozen'):analyze(dots,words,out,out)


def test_all_serialized_fields_and_formal_coverage():
 names=['M','N','transLength','shareUbSize']+['other'+str(i) for i in range(46)]
 schema=dict(bytes=200,fields=[dict(name=n,offset=i*4) for i,n in enumerate(names)])
 fp=td.decode(struct.pack('<50i',*range(50)),schema);raw=fp.copy();raw['transLength']=174080;raw['other45']=99
 report=td.compare_fields(fp,raw,dict(M=0,N=1),dict(M=0,N=1))
 assert report['field_count']==50 and len(report['formal_fields_missing'])==48
 rows={r['field']:r for r in report['fields']}
 assert rows['transLength']['classification']=='CTYPE_WORKSPACE_RELATED_EXACT_DELTA_UNRESOLVED'
 assert rows['other45']['classification']=='UNEXPLAINED_DIFFERENCE'
 with pytest.raises(ValueError):td.compare_fields(fp,raw,dict(M=77),dict(M=0))


@pytest.mark.parametrize('error',['size','overlap','unaligned','padding'])
def test_bad_serialization_rejected(error):
 schema=dict(bytes=12,fields=[dict(name='a',offset=0),dict(name='b',offset=4)]);data=bytes(12)
 if error=='size':data=bytes(8)
 if error=='overlap':schema['fields'][1]['offset']=0
 if error=='unaligned':schema['fields'][1]['offset']=3
 if error=='padding':data=bytes(8)+b'bad!'
 with pytest.raises(ValueError):td.decode(data,schema)


def test_schema_identity_scope(tmp_path):
 header=tmp_path/'schema.h';header.write_text('BEGIN_TILING_DATA_DEF(TCubeTiling)\n'+''.join(f'TILING_DATA_FIELD_DEF(int32_t, x{i});\n' for i in range(50))+'END_TILING_DATA_DEF;')
 binding=dict(files=[dict(id='schema',file=c.identity(header))]);names=sdk.schema_names(binding)
 assert len(names)==50
 generated=st.source(names)
 assert 'set_x49(0x13579bdf)' in generated and 'aclInit' not in generated
 header.write_text(header.read_text().replace('int32_t','uint32_t'))
 with pytest.raises(ValueError):sdk.schema_names(binding)


@pytest.mark.parametrize('phase',['host','build','bounded'])
def test_execution_gates_stay_closed(tmp_path,phase):
 a=SimpleNamespace(stage=phase,output_dir=tmp_path/phase,cann_root=tmp_path/'sdk',frozen_root=None,raw_root=None)
 assert run.main(a)==1
 r=c.read(a.output_dir/'report.json')
 assert r['status']=='BLOCKED' and r['opp']=='NOT_CREATED' and r['npu']=='NOT_RUN'
 assert 'commands' not in r


def test_no_executable_kernel_or_intrinsic_bypass():
 text=(HERE/'local_check/interfaces.cpp').read_text()
 assert '__global__' not in text and 'copy_matrix_' not in text
 assert 'DataCopy(l0c,ubInt,p,plain)' in text and 'DeqScale::VDEQ16' in text
 assert not (HERE/'op_kernel').exists() and not (HERE/'runner').exists()
 assert c.admission()['l0c_injection']=='UNRESOLVED_NOT_PROVEN_UNSUPPORTED'


def test_cli_import_and_missing_reports():
 subprocess.run([sys.executable,'-B',str(HERE/'run_vdeq_review.py'),'--help'],check=True,capture_output=True)
 with pytest.raises(SystemExit):run.parse(['tiling','--output-dir','/tmp/out'])
 with pytest.raises(SystemExit):run.parse(['analyze','--output-dir','/tmp/out'])


def test_key_dots_exact_binary32():
 assert len(c.KEY_DOTS)==6
 for value in c.KEY_DOTS.values():assert value<0 and value%2 and int(np.float32(value))==value


def test_no_false_field_equality_from_missing_formal_records():
 a=dict(baseN=256,hidden=0);r=td.compare_fields(a,a,dict(baseN=256),dict(baseN=256))
 assert r['formal_fields_missing']==['hidden']
 assert r['formal_all_serialized_fields'].startswith('UNRESOLVED')
