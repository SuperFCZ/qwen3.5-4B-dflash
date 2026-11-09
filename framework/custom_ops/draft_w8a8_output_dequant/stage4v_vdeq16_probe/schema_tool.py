"""Compile/run an offline TCubeTiling serializer self-check, never ACL/device APIs."""
import json,platform,re,shutil
from pathlib import Path
from vdeq_common import identity,verify,write,raw
from vdeq_sdk import schema_names


def source(names):
 text=r'''#include <tiling/tiling_api.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
static size_t locate(optiling::TCubeTiling& t){
 std::vector<unsigned char>b(t.GetDataSize(),0);t.SaveToBuffer(b.data(),b.size());
 size_t offset=b.size(),count=0;
 for(size_t i=0;i+4<=b.size();i+=4)if(b[i]==0xdf&&b[i+1]==0x9b&&b[i+2]==0x57&&b[i+3]==0x13){offset=i;++count;}
 if(count!=1)throw std::runtime_error("no unique serializer marker");
 for(size_t i=0;i<b.size();++i)if((i<offset||i>=offset+4)&&b[i])throw std::runtime_error("unexpected default/padding bytes");
 return offset;
}
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("output path required");
 optiling::TCubeTiling zero;std::vector<unsigned char>b(zero.GetDataSize(),0);zero.SaveToBuffer(b.data(),b.size());
 for(auto c:b)if(c)throw std::runtime_error("nonzero default");
 std::ofstream out(argv[1]);out<<"{\"status\":\"SDK_SERIALIZATION_SELF_CHECKED\",\"bytes\":"<<b.size()<<",\"fields\":[";
'''
 for i,name in enumerate(names):
  text+=' {optiling::TCubeTiling t;t.set_'+name+'(0x13579bdf);out<<"'+(',' if i else '')+'{\\"name\\":\\"'+name+'\\",\\"offset\\":"<<locate(t)<<"}";}\n'
 text+='out<<"]}\\n";if(!out)throw std::runtime_error("write");return 0;}catch(const std::exception&e){std::cerr<<e.what()<<"\\n";return 1;}}\n'
 return text


def build(root,cann,binding):
 if platform.system()!='Linux' or platform.machine() not in ('aarch64','arm64'):raise ValueError('offline serializer requires Linux ARM64 CANN libraries')
 root.mkdir();src=root/'serializer.cpp';src.write_text(source(schema_names(binding)))
 libs={}
 for name in ('tiling_api','register','c_sec','platform','ascendcl'):
  choices=[p for d in (cann/'lib64',cann/'aarch64-linux/lib64') for suffix in ('.so','.a') for p in [d/('lib'+name+suffix)] if p.is_file()]
  if not choices:raise ValueError('missing CANN library '+name)
  item=identity(choices[0])
  if cann not in Path(item['path']).parents:raise ValueError('library escaped CANN')
  libs[name]=item
 compiler=shutil.which('c++')
 if not compiler:raise ValueError('Host compiler absent')
 include=[cann/'include',cann/'include/experiment',cann/'aarch64-linux/asc/include',cann/'aarch64-linux/asc/include/utils',cann/'aarch64-linux/asc/include/adv_api']
 r=dict(source=identity(src),libraries=libs,compiler=identity(compiler),commands={},device_calls='NONE')
 env=raw.common.env(cann)
 from run_stage2b import command
 argv=[compiler,'-std=c++17','-O2','-MMD','-MF',root/'serializer.d','-MT',root/'serializer',src,'-o',root/'serializer']+['-I'+str(x) for x in include]+[verify(x) for x in libs.values()]+['-ldl']
 try:
  command(argv,root/'compile.log',env,r,'compile',300)
  from raw_host_dependency_identity import make_dependencies
  deps=make_dependencies((root/'serializer.d').read_text(),root,(root/'serializer').resolve())
  schema_file=next(x['file'] for x in binding['files'] if x['id']=='schema')
  paths=[p for p in deps if p.name=='matmul_tilingdata.h']
  if not paths or any(identity(p)!=schema_file for p in paths):raise ValueError('serializer compiled a different TCubeTiling schema')
  r['dependency_file']=identity(root/'serializer.d');r['compiled_schema_header']=schema_file
  command(['ldd',root/'serializer'],root/'ldd.log',env,r,'ldd',60)
  if 'not found' in (root/'ldd.log').read_text():raise ValueError('serializer library unresolved')
  r['loaded_libraries']=[identity(x) for x in re.findall(r'=>\s+(/\S+)\s+\(', (root/'ldd.log').read_text())]
  command([root/'serializer',root/'schema.json'],root/'run.log',env,r,'serializer',60)
  schema=json.loads((root/'schema.json').read_text())
  if [x['name'] for x in schema['fields']]!=schema_names(binding):raise ValueError('schema field order differs')
  from tiling_decode import decode
  decode(bytes(schema['bytes']),schema)
  r.update(status='OFFLINE_SERIALIZER_COMPLETED',binary=identity(root/'serializer'),schema=identity(root/'schema.json'))
  return r
 finally:write(root/'report.json',r)
