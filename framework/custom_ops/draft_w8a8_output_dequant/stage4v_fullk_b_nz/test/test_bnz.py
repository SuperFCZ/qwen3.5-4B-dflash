import inspect,importlib.util,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import nz_layout as z
import nz_transform as t
import nz_common as c
import run_b_nz as run
import nz_protocol as protocol


def test_all_elements_prepack_unpack_signed_and_every_sdk_window(tmp_path):
 source=tmp_path/'b.bin'
 with source.open('wb') as f:
  for tile in range(76):f.write(((np.arange(2560*256,dtype=np.int32)*17+tile*73)%256-128).astype('i1').tobytes())
 before=c.identity(source);out=tmp_path/'b_nz.bin';r=z.prepack(source,out);z.verify_pack(source,out,r)
 assert c.identity(source)==before and r['physical_bytes']==49807360 and r['padding_bytes']==0
 assert r['retained_old_plus_new_disk_bytes']==99614720 and r['permutation_ms']>0
 old=np.memmap(source,mode='r',dtype='i1',shape=(76,2560,256));new=np.memmap(out,mode='r',dtype='i1',shape=z.SHAPE)
 for tile in range(76):
  assert np.array_equal(z.unpack_tile(new[tile]),old[tile])
  flat=new[tile].reshape(-1)
  # SDK CopyNZ2NZ: row0, col=g*128, gRow=N256; nburst4, blockLen256*32B.
  for g in range(20):
   off=(g*128)*256;segment=flat[off:off+4*256*32].reshape(4,16,16,32)
   logical=segment.transpose(0,3,1,2).reshape(128,256)
   assert np.array_equal(logical,old[tile,g*128:(g+1)*128])
  for k in (0,31,32,127,128,2559):
   for n in (0,15,16,31,32,255):assert new.reshape(-1)[z.offset(tile,k,n)]==old[tile,k,n]
 for tile in (0,37,38,75):assert r['tiles'][tile]['tile']==tile
 assert r['logical_elements']==76*2560*256
 with out.open('r+b') as f:f.seek(12345);f.write(b'!')
 with pytest.raises(ValueError):z.verify_pack(source,out,r)


def test_invalid_layout_and_amortization():
 with pytest.raises(ValueError):z.pack_tile(np.zeros((2560,256),dtype=np.uint8))
 with pytest.raises(ValueError):z.unpack_tile(np.zeros((80,16,32,16),dtype='i1'))
 with pytest.raises(ValueError):z.offset(76,0,0)
 assert z.amortization(10,2,1)['break_even_calls']==10
 assert z.amortization(10,1,1)['break_even_calls'] is None
 assert z.amortization(10,1,2)['break_even_calls'] is None


def test_exact_only_b_representation_change():
 t.verify_repository()
 for kind in t.NAMES:
  old=(t.BASE/kind/'op_kernel'/(t.OLD[kind][1]+'.cpp')).read_text()
  new=(HERE/kind/'op_kernel'/(t.NAMES[kind][1]+'.cpp')).read_text()
  normalized=new.replace(t.NAMES[kind][1],t.OLD[kind][1])
  normalized=normalized.replace('using BType=MatmulType<TPosition::GM,CubeFormat::NZ,int8_t,true>;','using BType=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;').replace('mm.SetTensorB(bg[tile*2560*256],true)','mm.SetTensorB(bg[tile*2560*256],false)')
  assert normalized==old


@pytest.fixture(scope='module')
def exes(tmp_path_factory):
 old=HERE.parent/'stage4v';root=tmp_path_factory.mktemp('bnz-kernel');paths={}
 for i,kind in enumerate(('candidate','fault')):
  text=(HERE/kind/'op_kernel'/(t.NAMES[kind][1]+'.cpp')).read_text().replace(t.NAMES[kind][1],'d_flash_w8_a8_full_k_channel')
  paths[i]=root/(kind+'.cpp');paths[i].write_text(text)
 spec=importlib.util.spec_from_file_location('_bnz_cpu',old/'test/test_stage4v.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 code=inspect.getsource(m.kernel_exes.__wrapped__).replace("@pytest.fixture(scope='module')\n",'').replace('def kernel_exes(tmp_path_factory):','def build_reused(tmp_path_factory):')
 code=code.replace("str(HERE/'op_kernel/d_flash_w8_a8_full_k_channel.cpp')",'str(paths[fault])')
 code=code.replace("'V_S,MTE2_S,S_MTE3'","'V_S,MTE2_S,S_MTE3,MTE3_MTE2,MTE3_S'")
 code=code.replace('template<TPosition P,CubeFormat F,class T>struct MatmulType{};','template<TPosition P,CubeFormat F,class T,bool TRANS=false>struct MatmulType{};')
 code=code.replace('Require(!trans);b=v;hb=true;','Require(trans);b=v;hb=true;')
 code=code.replace('b.data[k*256+n]','b.data[(k/32)*8192+n*32+k%32]')
 code=code.replace('b[(t*2560+k)*256+n]','b[t*655360+(k/32)*8192+n*32+k%32]')
 ns=dict(m.__dict__,paths=paths);exec(compile(code,'bnz_cpu_builder','exec'),ns)
 return ns['build_reused'](tmp_path_factory)


@pytest.mark.parametrize('blocks,tile',[(7,-1),(2,-1),(7,0),(7,37),(7,38),(7,75)])
def test_actual_kernel_packed_b_indices_and_numerics(exes,tmp_path,blocks,tile):
 out=tmp_path/'out';subprocess.run([str(exes[0]),str(out),str(blocks),str(tile),'0','0'],check=True,capture_output=True)
 actual=np.fromfile(out,'<f2').reshape(16,19456);factor=sum((k%5-2)*(k%3-1) for k in range(2560))
 for tt in range(76):
  if tile>=0 and tt!=tile:assert np.all(actual[:,tt*256:(tt+1)*256].view('<u2')==0x7f7f);continue
  dot=np.array([[factor*(m%3-1)*(tt%3-1)*(n%5-2) for n in range(256)] for m in range(16)],np.int64)
  scale=np.array([1/(1<<(n%4+3)) for n in range(256)],'<f4')
  assert actual[:,tt*256:(tt+1)*256].tobytes()==(dot.astype('<f4')*scale).astype('<f2').tobytes()


@pytest.mark.parametrize('mode',range(2,8))
def test_actual_fault_source_still_checks_all_sites(exes,tmp_path,mode):
 out=tmp_path/'out';cp=subprocess.run([str(exes[1]),str(out),'1','0',str(mode),'0'],check=True,capture_output=True,text=True)
 assert out.read_bytes()==protocol.fault_expected(mode,0)
 assert int(cp.stdout)==(0 if mode in (3,6) else 1)


@pytest.mark.parametrize('stage',list(run.STATES))
def test_all_cli_phase_requirements(stage,tmp_path):
 keys=['source','prepare','hostplan','build','host','tile','full','fault'];n=list(run.STATES).index(stage)
 argv=[stage,'--output-dir',str(tmp_path/'new')]
 if n:
  with pytest.raises(SystemExit):run.parse(argv)
 for k in keys[:n]:argv+=['--'+k+'-report',str(tmp_path/k)]
 assert run.parse(argv).phase==stage


def test_resources_keep_envelope_and_reject_overflow():
 import importlib.util
 spec=importlib.util.spec_from_file_location('_bnz_host_fixture',HERE.parent/'stage4v/test/test_stage4v.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 from nz_resources import validate
 platform,row=m.fixture_tiling();r=validate(platform,row)
 assert r['status']=='HOST_RESOURCE_CHECKS_PASS' and r['details']['total_ub']==220672
 assert r['details']['b_runtime_transdata'].startswith('BYPASSED')
 row['fields']['transLength']=202240
 assert validate(platform,row)['status']=='BLOCKED'


def test_no_historical_source_inventory_includes_new_experiment():
 assert not any('/stage4v_fullk_b_nz/' in f['path'] for f in c.sg.sources())


def test_storage_metadata_is_consistently_transposed():
 for kind in t.NAMES:
  host=(HERE/kind/'op_host/host_common.h').read_text()
  kernel=(HERE/kind/'op_kernel'/(t.NAMES[kind][1]+'.cpp')).read_text()
  runner=(HERE/kind/'runner/runner.cpp').read_text()
  assert 'SetBType(TPosition::GM,CubeFormat::NZ,DataType::DT_INT8,true)' in host
  assert 'MatmulType<TPosition::GM,CubeFormat::NZ,int8_t,true>' in kernel
  assert 'mm.SetTensorB(bg[tile*2560*256],true)' in kernel
  assert 'r.w.Make({76,80,16,16,32},{76,80,16,16,32}' in runner
