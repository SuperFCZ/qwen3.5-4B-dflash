"""Typed plans, actual post-ATC kernel evidence, isolated SDK serializer check."""
import hashlib
import mmap
from pathlib import Path
import re
import struct
from gate_common import HERE,GATE,SHAPE,OFF_PARENT,SDK_SCHEMA,SCHEMA_SHA,c,command,one,require,sha
from flow_query import assignments

CUBE=dict(baseM=16,baseN=64,baseK=256,depthA1=1,depthB1=1,stepM=1,stepN=1,stepKa=1,stepKb=1,
          dbL0A=2,dbL0B=2,dbL0C=1,transLength=34816)
TAG='GATE8_PARENT_224_PREFIX_24_CUBE_200_FIELDS_50_MATCH'


def schema():
    require(c.sha(SDK_SCHEMA)==SCHEMA_SHA,'TCube schema drift')
    fields=c.read(SDK_SCHEMA)['fields']
    require(len(fields)==50 and [f['offset'] for f in fields]==list(range(0,200,4)) and
            len({f['name'] for f in fields})==50 and all(re.fullmatch(r'[A-Za-z][A-Za-z0-9]*',f['name']) for f in fields),'invalid TCube field map')
    return fields


def sdk_layout(folder,project,cann,env,compiler):
    folder.mkdir()
    markers=folder/'cube_setters.inc'
    markers.write_text(''.join(f"cube.set_{f['name']}({0x13570000+i}u);\n" for i,f in enumerate(schema())))
    args=[compiler,'-std=c++17','-D_GLIBCXX_USE_CXX11_ABI=0','-I'+str(folder),'-I'+str(project/'op_host'),'-I'+str(cann/'include'),
          HERE/'layout_probe.cpp','-L'+str(cann/'lib64'),'-lnnopbase','-lexe_graph','-lregister','-ltiling_api','-ldl','-o',folder/'layout']
    command(args,folder/'compile.log',env)
    command([folder/'layout'],folder/'check.log',env)
    require((folder/'check.log').read_text().splitlines().count(TAG)==1,'SDK layout not proven')
    sdk_files=[c.identity(cann/'include'/rel) for rel in ('register/tilingdata_base.h','tiling/tiling_api.h')]
    require(all(cann.resolve() in Path(x['path']).parents for x in sdk_files),'serializer header escapes selected CANN')
    return dict(method=TAG,fields_schema=c.identity(SDK_SCHEMA),declaration=c.identity(project/'op_host/d_flash_group_quant_linear_tiling.h'),
                markers=c.identity(markers),binary=c.identity(folder/'layout'),witness=c.identity(folder/'check.log'),sdk_files=sdk_files,
                dependencies=c.dependencies(c.identity(folder/'layout'),folder/'layout-ldd.log',env))


def gate_plan(plan,mode,reference):
    p=plan['platform'];s=plan['schedule'];r=s['resources'];ref=reference['schedule']
    require(mode in ('off','raw') and reference['tiling_sha256']==OFF_PARENT and reference['platform']['blockDim']==8,'unbound Gate reference')
    require({k:v for k,v in p.items() if k!='host_dso'}=={k:v for k,v in reference['platform'].items() if k!='host_dso'},
            'Gate PlatformInfo/Workspace drift')
    require(s['variant']=='A' and s['route']==p['route']=='ATC' and [s[k] for k in ('m','k','n')]==SHAPE and
            s['n_tiles']==304 and s['tile_n']==64 and s['core_limit']==p['coreLimit']==0 and
            p['GetCoreNum']==p['availableCores']==s['generic']==s['baseline_block_dim']==s['selected_block_dim']==p['blockDim']==8,'not real ATC A/8-auto')
    require(s['baseline_assignments']==assignments(304,8) and all(len(x)==38 for x in s['baseline_assignments']),'missing/overlapping N tiles')
    raw=mode=='raw';scratch=185600 if raw else 202240
    desired=dict(user_ub=76544 if raw else 59904,scratch=scratch,raw_banks=2 if raw else 1,tile_k=256,codes_bytes=32768 if raw else 16384,
        scale_bytes=512 if raw else 256,weight_bytes=32768,l1_used=40960,l0a_used=16384,l0b_used=65536,l0c_used=4096,workspace=ref['resources']['workspace'])
    require(r==desired and p['workspace']==r['workspace']==2097152 and p['capacity']['UB']==r['user_ub']+r['scratch']==262144 and
            p['capacity']['L0B']==r['l0b_used'],'Gate resources/capacity mismatch')
    other=set(ref)-{'resources','tiling_hex'}
    require(set(s)==set(ref) and all(s[k]==ref[k] for k in other),'non-resource scheduler field drift')
    data=bytes.fromhex(s['tiling_hex']);old=bytes.fromhex(ref['tiling_hex'])
    require(sha(old)==OFF_PARENT and sha(data)==plan['tiling_sha256'] and len(data)==len(old)==224 and
            data[:24]==struct.pack('<IIIIQ',*SHAPE,256,scratch) and old[:24]==struct.pack('<IIIIQ',*SHAPE,256,202240),
            'parent identity/SDK-proven outer fields differ')
    require(data[24:]==old[24:],'actual nested Cube serialization differs')
    fields={f['name']:struct.unpack_from('<I',data,24+f['offset'])[0] for f in schema()}
    require(all(fields[k]==v for k,v in CUBE.items()),'base/step/depth/DB/transLength drift')
    return dict(block_dim=p['blockDim'],available=p['GetCoreNum'],assignments=s['baseline_assignments'],workspace=r['workspace'],resources=r,
                selected_pipeline='raw-prefetch-v1' if raw else 'serial-v1',selection_basis='frozen generated header, shape predicate and real callback raw_banks',
                parent_sha256=sha(data),cube_offset=24,cube_bytes=200,cube_sha256=sha(data[24:]),cube_fields=fields)


def non_gate(plan,reference):
    require(plan['platform']['blockDim']==plan['platform']['GetCoreNum']==8 and plan['schedule']['variant']=='A','full non-Gate launch drift')
    require(plan['schedule']==reference['schedule'] and plan['tiling_sha256']==reference['tiling_sha256'] and
            {k:v for k,v in plan['platform'].items() if k!='host_dso'}=={k:v for k,v in reference['platform'].items() if k!='host_dso'},'full non-target resources/Tiling changed')


def elf_sections(data):
    require(len(data)>=64 and data[:7]==b'\x7fELF\x02\x01\x01','expected little-endian ELF64')
    identity=struct.unpack_from('<16sHHIQQQIHHHHHH',data)
    off,width,count=identity[6],identity[11],identity[12]
    require(identity[3]==1 and identity[8]==width==64 and count>0 and identity[13]<count and off+width*count<=len(data),'invalid ELF section extent')
    ranges=[]
    for i in range(count):
        _,kind,flags,_,start,size,*_=struct.unpack_from('<IIQQQQIIQQ',data,off+width*i)
        require(kind==8 or start+size<=len(data),'ELF truncated section')
        if flags&4 and size:
            require(kind==1,'unexpected executable section')
            ranges.append(dict(bytes=size,sha256=sha(data[start:start+size])))
    require(bool(ranges),'empty ELF executable payload')
    return ranges


def node_binding(node):
    attrs=node['attributes']
    require(node['type']==GATE and isinstance(attrs,list),'unexpected OM dump Gate node')
    def field(key,typ):
        val=one((x['value'] for x in attrs if x.get('key')==key),'ambiguous/missing graph attr '+key)
        return val[typ]
    bid=field('_kernel_bin_id','s');symbol=field('_kernelname','s');size=int(field('_tbeKernelSize','i'))
    require(re.fullmatch(r'te_dflashgroupquantlinear_[a-f0-9]{128}',bid) and re.fullmatch(re.escape(bid[:-64])+r'__kernel\d+',symbol) and
            64<size<=16*1024*1024 and int(field('tvm_blockdim','i'))==8,'unproven specialized kernel binding or graph blockDim')
    engine=field('_ge_attr_op_kernel_lib_name','s')
    require(engine=='AIcoreEngine' and field('_op_no_tiling','b') is False,'Gate not bound to a tiled AI Core kernel')
    return dict(node=node['name'],id=bid,symbol=symbol,sha256=bid[-64:],bytes=size,block_dim=8,engine=engine)


def embedded(om,bindings):
    """Hash the full, structurally bounded ATC ELF; a kernel-name substring is not identity."""
    groups={}
    for item in bindings:
        groups.setdefault((item['bytes'],item['sha256']),[]).append(item)
    require(groups and len({b['node'] for b in bindings})==len(bindings),'empty/duplicate Gate graph nodes')
    sizes={size for size,_ in groups};found={}
    with Path(om).open('rb') as fd,mmap.mmap(fd.fileno(),0,access=mmap.ACCESS_READ) as data:
        offset=0
        while True:
            pos=data.find(b'\x7fELF\x02\x01\x01',offset)
            if pos<0:break
            offset=pos+7
            if pos+64>len(data):continue
            hdr=struct.unpack_from('<16sHHIQQQIHHHHHH',data,pos)
            size=hdr[6]+hdr[11]*hdr[12]
            if hdr[8]!=64 or hdr[11]!=64 or not hdr[12] or size not in sizes or pos+size>len(data):continue
            blob=data[pos:pos+size];key=(size,hashlib.sha256(blob).hexdigest())
            if key not in groups:continue
            require(key not in found and all(x['symbol'].encode() in blob for x in groups[key]),'ambiguous embedded ELF/symbol')
            found[key]=dict(offset=pos,executable_sections=elf_sections(blob))
    require(set(groups)==set(found),'BLOCKED_UNPROVEN_ATC_KERNEL_IDENTITY: expected complete ELF not present')
    return {x['node']:dict(x,**found[x['bytes'],x['sha256']]) for x in bindings}


def recheck_embedded(om,bindings,stored):
    require(set(stored)=={item['node'] for item in bindings} and len(stored)==len(bindings),'embedded node inventory drift')
    with Path(om).open('rb') as fp,mmap.mmap(fp.fileno(),0,access=mmap.ACCESS_READ) as data:
        for item in bindings:
            old=stored[item['node']];offset=old['offset'];size=item['bytes']
            require(all(old[k]==v for k,v in item.items()) and type(offset) is int and 0<=offset<=len(data)-size,'node/ELF reference drift')
            blob=data[offset:offset+size]
            require(sha(blob)==item['sha256'] and item['symbol'].encode() in blob and elf_sections(blob)==old['executable_sections'],'bound OM executable changed')
