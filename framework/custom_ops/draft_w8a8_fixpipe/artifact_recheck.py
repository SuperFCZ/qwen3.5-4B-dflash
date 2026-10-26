#!/usr/bin/env python3
"""Read-only existing-build recheck. No subprocess, installation or ELF loading."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
from w8_common import HERE,identity,verify,write,fresh
from w8_run import verify_tree
from compile_raw import check_artifacts,elf_info,HOST_LINK_TARGET

MAX_ELF=64*1024*1024

def sha(data):return hashlib.sha256(data).hexdigest()


def host_elf(path):
    item,header=elf_info(path,True)
    if item['bytes']>MAX_ELF:raise ValueError('Host ELF exceeds bounded inspection size')
    data=path.read_bytes();verify(item)
    fields=struct.unpack('<HHIQQQIHHHHHH',data[16:64]);phoff,phsize,phnum=fields[4],fields[8],fields[9]
    segments=[]
    for i in range(phnum):
        t,flags,offset,vaddr,paddr,filesz,memsz,align=struct.unpack_from('<IIQQQQQQ',data,phoff+i*phsize)
        if offset+filesz>len(data) or (t==1 and filesz>memsz):raise ValueError('invalid ELF segment bounds')
        segments.append(dict(index=i,type=t,flags=flags,offset=offset,vaddr=vaddr,paddr=paddr,
            filesz=filesz,memsz=memsz,align=align,sha256=sha(data[offset:offset+filesz])))
    loads=[s for s in segments if s['type']==1]
    if not loads:raise ValueError('Host library has no PT_LOAD')
    dynamic=[s for s in segments if s['type']==2]
    if len(dynamic)!=1 or dynamic[0]['filesz']%16:raise ValueError('expected one complete PT_DYNAMIC')
    d=dynamic[0];entries=[];terminated=False
    for pos in range(d['offset'],d['offset']+d['filesz'],16):
        tag,value=struct.unpack_from('<qQ',data,pos)
        if tag==0:terminated=True;break
        entries.append(dict(tag=tag,value=value,file_offset=pos))
    if not terminated:raise ValueError('unterminated PT_DYNAMIC')
    def unique(tag):
        values=[e['value'] for e in entries if e['tag']==tag]
        if len(values)!=1:raise ValueError('missing/ambiguous dynamic tag '+str(tag))
        return values[0]
    address,size=unique(5),unique(10) # DT_STRTAB / DT_STRSZ
    matches=[s for s in loads if s['vaddr']<=address and address+size<=s['vaddr']+s['filesz']]
    if len(matches)!=1:raise ValueError('dynamic strings not mapped to a unique PT_LOAD')
    strings_offset=matches[0]['offset']+address-matches[0]['vaddr'];strings=data[strings_offset:strings_offset+size]
    decoded=[]
    for entry in entries:
        if entry['tag'] not in (1,14,15,29):continue
        start=entry['value'];end=strings.find(b'\0',start)
        if start>=size or end<0:raise ValueError('invalid dynamic string')
        value=strings[start:end].decode('utf-8')
        decoded.append(dict(tag=entry['tag'],name={1:'NEEDED',14:'SONAME',15:'RPATH',29:'RUNPATH'}[entry['tag']],
                            value=value,file_offset=strings_offset+start,bytes=end-start+1))
    return dict(identity=item,elf=header,segments=segments,load_segments=loads,dynamic_entries=entries,
        dynamic_strings=decoded,dynstr=dict(offset=strings_offset,bytes=size,sha256=sha(strings))),data


def compare_host(a,b):
    left,x=host_elf(a);right,y=host_elf(b)
    ranges=[];count=0;run=None;range_count=0
    rpath=[(s['file_offset'],s['file_offset']+s['bytes']) for s in left['dynamic_strings'] if s['name'] in ('RPATH','RUNPATH')]
    rpath_only=len(x)==len(y)
    for i in range(max(len(x),len(y))):
        different=i>=len(x) or i>=len(y) or x[i]!=y[i]
        if different:
            count+=1
            if not any(lo<=i<hi for lo,hi in rpath):rpath_only=False
            if run is None:run=i
        if run is not None and (not different or i==max(len(x),len(y))-1):
            end=i if not different else i+1;range_count+=1
            if len(ranges)<128:
                ranges.append(dict(start=run,end_exclusive=end,build_hex=x[run:min(end,run+32)].hex(),
                    package_hex=y[run:min(end,run+32)].hex(),hex_truncated=end-run>32))
            run=None
    def deps(v):return [(s['tag'],s['value']) for s in v['dynamic_strings'] if s['name'] in ('NEEDED','SONAME')]
    def layouts(v):return [{k:value for k,value in s.items() if k!='sha256'} for s in v['segments']]
    return dict(build=left,package=right,identical_bytes=x==y,
        program_headers_equal=layouts(left)==layouts(right),pt_load_equal=left['load_segments']==right['load_segments'],
        needed_soname_equal=deps(left)==deps(right),
        dynamic_entries_equal=left['dynamic_entries']==right['dynamic_entries'],
        different_byte_count=count,difference_range_count=range_count,difference_ranges=ranges,
        ranges_truncated=range_count>len(ranges),
        rpath_string_only_candidate=bool(count and rpath_only),
        classification='BYTE_IDENTICAL' if x==y else 'UNRESOLVED_PACKAGING_TRANSFORM',
        acceptance='NO_RELAXATION; Build ID/NEEDED equality or RPATH-only differences do not establish provenance')


def collect_packaging(project,out):
    # Generated build tree only; never scans SDK or executes install/strip scripts.
    paths=set((project/'build_out').rglob('cmake_install.cmake'))
    paths.update((project/'build_out').rglob('CPackConfig.cmake'))
    paths.update((project/'build_out').rglob('install_manifest*.txt'))
    paths.update((project/'build_out/op_host').glob('CMakeFiles/*/link.txt'))
    paths.update(p for p in (project/'CMakeLists.txt',project/'op_host/CMakeLists.txt',project/'build_out/CMakeCache.txt') if p.is_file())
    paths.update((project/'cmake').glob('*.cmake'))
    if len(paths)>128:raise ValueError('too many packaging files; bounded inspection incomplete')
    out.mkdir();rows=[];total=0
    pattern=re.compile(r'RPATH|RUNPATH|STRIP|strip|optiling|opmaster|file\s*\(\s*INSTALL|execute_process|CPACK_INSTALL|CPACK_.*SCRIPT',re.I)
    for index,path in enumerate(sorted(paths)):
        if project.resolve() not in path.resolve(strict=True).parents:raise ValueError('packaging evidence escapes source project')
        item=identity(path);total+=item['bytes']
        if item['bytes']>2*1024*1024 or total>16*1024*1024:raise ValueError('packaging text budget exceeded')
        raw=path.read_bytes();verify(item)
        text=raw.decode('utf-8');lines=text.splitlines();hits=[i for i,line in enumerate(lines) if pattern.search(line)]
        selected=sorted({j for i in hits for j in range(max(0,i-4),min(len(lines),i+9))})
        snapshot=out/f'{index}-{path.name}';snapshot.write_bytes(raw)
        rows.append(dict(source=item,snapshot=identity(snapshot),hit_lines=[i+1 for i in hits],
            excerpts=[dict(line=i+1,text=lines[i]) for i in selected[:200]],excerpts_truncated=len(selected)>200))
    return dict(status='COLLECTED_FOR_REVIEW',files=rows,total_bytes=total,
        execution_provenance='UNRESOLVED: generated script presence alone does not prove a transformation executed')


def inspect_original(path,expected):
    item=identity(path)
    if item['sha256']!=expected:raise ValueError('original report SHA256 mismatch')
    r=json.loads(path.read_text())
    if r.get('scope')!='RAW_ND_K128_COMPILE_ONLY' or r.get('shape')!=[16,128,64] or r.get('input_count')!=2:
        raise ValueError('not the bounded raw compile experiment')
    for stage in ('msopgen','build'):
        cmd=r.get('commands',{}).get(stage,{})
        if cmd.get('status')!='COMPLETED' or type(cmd.get('returncode')) is not int or cmd['returncode']!=0:
            raise ValueError('original command not completed successfully: '+stage)
    if r.get('installation')!='NOT_RUN' or r.get('npu')!='NOT_RUN':raise ValueError('requires uninstalled, unexecuted compile evidence')
    if r.get('artifact_checks',{}).get('kernel',{}).get('status')!='VERIFIED' or not r.get('kernel_objects'):
        raise ValueError('original verified Kernel evidence missing')
    verify_tree(r)
    return r,item


def main(args):
    project=args.report.parent/'CustomOp'
    root=fresh(args.output_dir,[args.report.parent])
    result=dict(status='ARTIFACT_RECHECK_UNRESOLVED',source_review='UNRESOLVED',
        recompiled=False,installation='NOT_RUN',npu='NOT_RUN',performance='NOT_RUN',
        analysis_sources=[identity(HERE/n) for n in ('artifact_recheck.py','compile_raw.py','w8_run.py','w8_common.py')])
    try:
        old,bound=inspect_original(args.report,args.report_sha256);result['original_report']=bound
        result['original_kernel_objects']=old['kernel_objects']
        result['original_commands']=old['commands']
        checked={}
        try:check_artifacts(project,checked)
        except ValueError as e:result['artifact_error']=str(e)
        result['rechecked_artifacts']=checked
        if checked.get('artifact_checks',{}).get('kernel',{}).get('status')!='VERIFIED' or checked.get('kernel_objects')!=old['kernel_objects']:
            raise ValueError('Kernel evidence differs; cannot preserve VERIFIED')
        result['kernel_status']='VERIFIED_UNCHANGED'
        canonical=project/'build_out/op_host/libcust_opmaster_rt2.0.so'
        if identity(canonical)['sha256']!=args.build_host_sha256:raise ValueError('canonical Host differs from supplied evidence')
        aliases=sorted((project/'build_out/_CPack_Packages/Linux/External').glob(
            'custom_opp_*.run/packages/vendors/customize/op_impl/ai_core/tbe/op_tiling/liboptiling.so'))
        if len(aliases)!=1:raise ValueError('missing/ambiguous package runtime alias')
        alias=aliases[0]
        if not alias.is_symlink() or os.readlink(alias)!=HOST_LINK_TARGET:raise ValueError('unexpected package runtime alias')
        target=alias.resolve(strict=True)
        if project.resolve() not in target.parents or target!=alias.parent.resolve()/HOST_LINK_TARGET:raise ValueError('package target escapes expected layout')
        if identity(target)['sha256']!=args.package_host_sha256:raise ValueError('package Host differs from supplied evidence')
        result['package_alias']=dict(path=str(alias),target=os.readlink(alias),resolved=str(target))
        result['host_comparison']=compare_host(canonical,target)
        result['packaging_evidence']=collect_packaging(project,root/'packaging-snapshots')
        write(root/'packaging-evidence.json',result['packaging_evidence'])
        if 'artifact_error' not in result:
            result['status']='ARTIFACT_RECHECK_VERIFIED_REVIEW_UNRESOLVED'
        # A byte mismatch always remains unresolved; no rule infers strip/RPATH legitimacy.
        inspect_original(args.report,args.report_sha256)
        for side in ('build','package'):verify(result['host_comparison'][side]['identity'])
        if os.readlink(alias)!=result['package_alias']['target'] or str(alias.resolve(strict=True))!=result['package_alias']['resolved']:
            raise ValueError('runtime alias changed during inspection')
    except Exception as e:result.update(status='BLOCKED_ARTIFACT_RECHECK',error=f'{type(e).__name__}: {e}')
    finally:
        try:
            for item in result['analysis_sources']:verify(item)
            for row in result.get('packaging_evidence',{}).get('files',[]):verify(row['source']);verify(row['snapshot'])
        except Exception as e:result.update(status='BLOCKED_ARTIFACT_RECHECK',integrity_error=str(e))
        write(root/'artifact-recheck.json',result)
        c=result.get('host_comparison',{})
        (root/'artifact-recheck.md').write_text(
            '# Raw compile artifact recheck\n\n'+result['status']+'\n\n'+
            'No rebuild, install, Runner or NPU execution. Original report remains unchanged.\n\n'+
            f"Kernel: {result.get('kernel_status','UNRESOLVED')}. Host byte differences: {c.get('different_byte_count','UNRESOLVED')}. " +
            f"PT_LOAD equal: {c.get('pt_load_equal','UNRESOLVED')}. RPATH-string-only candidate: {c.get('rpath_string_only_candidate','UNRESOLVED')}.\n\n"+
            'Different load bytes are not accepted solely on Build ID, dynamic dependency names, or suspected strip/RPATH edits. '+
            'Review packaging-evidence.json and original build logs before defining any narrow transformation rule. '+
            'Generated script presence is not proof of execution; source review remains UNRESOLVED.\n')
    print(result['status']+': '+str(root/'artifact-recheck.json'))
    return 0 if result['status']=='ARTIFACT_RECHECK_VERIFIED_REVIEW_UNRESOLVED' else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--report-sha256',required=True)
    p.add_argument('--build-host-sha256',required=True);p.add_argument('--package-host-sha256',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    raise SystemExit(main(p.parse_args()))
