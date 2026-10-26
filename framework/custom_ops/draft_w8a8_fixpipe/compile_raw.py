#!/usr/bin/env python3
"""Compile-only ND INT8/INT32 diagnostic; never installs or runs an operator."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import struct
import subprocess
from w8_common import HERE,identity,verify,write,fresh
from w8_run import build_inputs,env

# Only the three exact include/module files identified during the source review.
SUPPLEMENT=(
 'include/adv_api/matmul/matmul_tiling.h',
 'include/adv_api/matmul/bmm_tiling.h',
 'impl/adv_api/detail/matmul/policy/matmul_private_modules.h',
)


def validate_preflight(path,expected_sha,cann):
    if identity(path)['sha256']!=expected_sha:raise ValueError('preflight SHA256 mismatch')
    p=json.loads(path.read_text())
    if p.get('raw_only') is not True or p.get('status')!='UNRESOLVED' or p.get('collection')!='FIXED_FILES_COLLECTED':
        raise ValueError('requires collected, unresolved raw-only preflight')
    if Path(p['cann_root']).resolve()!=cann:raise ValueError('SDK root differs')
    for item in p['sources']:verify(item)
    for f in p['files']:
        verify(f['source']);verify(f['snapshot'])
        if f['source']['sha256']!=f['snapshot']['sha256'] or f['source']['bytes']!=f['snapshot']['bytes']:
            raise ValueError('SDK source/snapshot identity differs')
    for name,tool in p['tools'].items():
        verify(tool['file'])
        actual=shutil.which(name)
        if not actual or identity(actual)!=tool['file']:raise ValueError('tool selection differs: '+name)
    if 'msopgen' not in p['tools']:raise ValueError('msopgen unavailable')
    # --version returncode is not a generation result.
    _,files,_,_=build_inputs(True,'nd',{})
    bound={f['path']:f for f in p['sources']}
    for file in files.values():
        item=identity(file)
        if bound.get(item['path'])!=item:raise ValueError('raw candidate is not bound to this preflight')
    return p


def run_build_command(argv,log,environment,report,key,timeout=1800):
    argv=list(map(str,argv))
    write(str(log)+'.command.json',dict(argv=argv,environment={k:environment.get(k) for k in
        ('ASCEND_HOME_PATH','ASCEND_OPP_PATH','ASCEND_CUSTOM_OPP_PATH','LD_LIBRARY_PATH')}))
    result=dict(argv=argv,returncode=None,status='RUNNING',log=str(log))
    report.setdefault('commands',{})[key]=result
    try:
        with log.open('w') as f:
            process=subprocess.run(argv,cwd=log.parent,env=environment,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
        result.update(returncode=process.returncode,status='COMPLETED' if process.returncode==0 else 'FAILED')
        if process.returncode:raise RuntimeError(f'{key} exited {process.returncode}: {log}')
    except subprocess.TimeoutExpired:
        result['status']='TIMED_OUT';raise
    finally:
        if log.is_file():result['log_identity']=identity(log)


def elf_info(path,host):
    before=identity(path)
    with path.open('rb') as f:header=f.read(64)
    if len(header)!=64 or header[:4]!=b'\x7fELF':raise ValueError('not a complete ELF header')
    if header[4:7]!=bytes([2,1,1]):raise ValueError('expected ELF64 little-endian version 1')
    kind,machine,version,entry,phoff,shoff,flags,ehsize,phsize,phnum,shsize,shnum,shstr=struct.unpack('<HHIQQQIHHHHHH',header[16:])
    if version!=1 or ehsize!=64:raise ValueError('invalid ELF header version/size')
    if not shnum or shsize!=64 or shoff<64 or shoff+shsize*shnum>before['bytes']:
        raise ValueError('missing/truncated ELF section table')
    if phnum and (phsize!=56 or phoff<64 or phoff+phsize*phnum>before['bytes']):raise ValueError('truncated ELF program table')
    if host and (kind!=3 or machine!=183):raise ValueError('Host library must be AArch64 ET_DYN')
    if not host and kind not in (1,2,3):raise ValueError('unsupported kernel ELF object type')
    # Device machine number is recorded, never guessed from a host architecture.
    verify(before)
    return before,dict(format='ELF64_LE',type=kind,machine=machine,flags=flags,
                       section_count=shnum,program_count=phnum)


HOST_LINK_TARGET='lib/linux/aarch64/libcust_opmaster_rt2.0.so'


def host_needed(path):
    """Read ELF DT_NEEDED without loading/executing a Host library."""
    data=path.read_bytes()
    fields=struct.unpack('<HHIQQQIHHHHHH',data[16:64]);shoff,shsize,shnum=fields[5],fields[10],fields[11]
    sections=[struct.unpack_from('<IIQQQQIIQQ',data,shoff+i*shsize) for i in range(shnum)]
    names=[]
    for sh in sections:
        if sh[1]!=6:continue # SHT_DYNAMIC
        offset,size,link,entry=sh[4],sh[5],sh[6],sh[9]
        if entry!=16 or size%16 or offset+size>len(data) or link>=len(sections):raise ValueError('invalid ELF dynamic table')
        strings=sections[link]
        if strings[1]!=3 or strings[4]+strings[5]>len(data):raise ValueError('invalid ELF dynamic string table')
        table=data[strings[4]:strings[4]+strings[5]]
        for i in range(offset,offset+size,16):
            tag,value=struct.unpack_from('<qQ',data,i)
            if tag==0:break
            if tag!=1:continue # DT_NEEDED
            end=table.find(b'\0',value)
            if value>=len(table) or end<0:raise ValueError('invalid DT_NEEDED offset')
            names.append(table[value:end].decode('ascii'))
    return names


def check_packaged_host(project,valid):
    roots=sorted((project/'build_out/_CPack_Packages/Linux/External').glob(
        'custom_opp_*.run/packages/vendors/customize/op_impl/ai_core/tbe/op_tiling'))
    rows=[]
    for root in roots:
        alias=root/'liboptiling.so';row=dict(path=str(alias),role='REQUIRED_PACKAGE_RUNTIME_ALIAS');rows.append(row)
        try:
            if not alias.is_symlink():raise ValueError('required packaged liboptiling.so symlink missing')
            row['link_target']=os.readlink(alias)
            if row['link_target']!=HOST_LINK_TARGET:raise ValueError('unexpected packaged alias target')
            resolved=alias.resolve(strict=True)
            if project.resolve() not in resolved.parents or resolved!=root.resolve()/HOST_LINK_TARGET:
                raise ValueError('packaged runtime library escapes expected location')
            item,info=elf_info(alias,True)
            matches=[f for f in valid if Path(f['path']).name==resolved.name and f['sha256']==item['sha256'] and f['bytes']==item['bytes']]
            if not matches:raise ValueError('packaged library does not match verified canonical Host library')
            row.update(status='VERIFIED',identity=item,elf=info,canonical=matches[0])
        except (OSError,ValueError,RuntimeError) as e:row.update(status='INVALID',reason=f'{type(e).__name__}: {e}')
    return dict(status=('VERIFIED' if all(r['status']=='VERIFIED' for r in rows) else 'FAILED') if rows else 'NOT_PRESENT',
                discovered_count=len(rows),candidates=rows)


def check_artifacts(project,report):
    kernel_dir=project/'build_out/op_kernel/binary/ascend310p/d_flash_w8_a8_accumulator'
    host_dir=project/'build_out/op_host'
    names={'liboptiling.so','libcust_opmaster_rt2.0.so','libcust_optiling.so'}
    groups={
      'kernel':(sorted(project.rglob('DFlashW8A8Accumulator_*.o')),kernel_dir,False),
      'host':(sorted(p for p in project.rglob('*.so') if p.name in names),host_dir,True),
    }
    checks={};report['artifact_checks']=checks
    for group,(paths,directory,host) in groups.items():
        rows=[];valid=[];bad=False;pending=[]
        for path in paths:
            row=dict(path=str(path));rows.append(row)
            if path.parent!=directory:
                row.update(status='EXCLUDED',reason='not the canonical build output; packaged copies do not prove an independent build')
                continue
            try:
                if path.is_symlink():
                    row['link_target']=os.readlink(path)
                    try:path.resolve(strict=True)
                    except FileNotFoundError:
                        if host and path.name=='liboptiling.so' and row['link_target']==HOST_LINK_TARGET:
                            row.update(status='PENDING',role='BUILD_LAYOUT_PACKAGE_ALIAS',dangling=True)
                            pending.append(row);continue
                        raise
                resolved=path.resolve(strict=True);row['resolved_path']=str(resolved)
                if project.resolve() not in resolved.parents or resolved.parent!=directory.resolve():raise ValueError('artifact escapes canonical build directory')
                if not path.is_file():raise ValueError('artifact is not a regular file')
                row['identity']=identity(path)
                if row['identity']['bytes']==0:raise ValueError('empty artifact')
                item,info=elf_info(path,host)
                if host:
                    row['dt_needed']=host_needed(path);verify(item)
                row.update(status='VERIFIED',identity=item,elf=info,role='CANONICAL_HOST_LIBRARY' if host else 'KERNEL_OBJECT');valid.append(item)
            except (OSError,ValueError,RuntimeError) as e:
                row.update(status='INVALID',reason=f'{type(e).__name__}: {e}');bad=True
        if host:
            runtime=check_packaged_host(project,valid);checks['host_runtime']=runtime
            if runtime['status']=='FAILED':bad=True
            for row in pending:
                matched=any(Path(f['path']).name=='libcust_opmaster_rt2.0.so' for f in valid)
                needed=any('liboptiling.so' in r.get('dt_needed',[]) for r in rows)
                if matched and not needed and runtime['status']=='VERIFIED':
                    row.update(status='WARNING',required_for_compile_validation=False,
                        reason='package-relative alias is dangling only in build layout; required package alias resolves to the SHA256-identical canonical library; no canonical Host DT_NEEDED uses this alias')
                else:
                    row.update(status='INVALID',reason='cannot establish optional build-alias role: missing runtime proof, canonical library, or DT_NEEDED requires alias');bad=True
        checks[group]=dict(discovered_count=len(paths),canonical_count=sum(p.parent==directory for p in paths),
            verified_count=len(valid),candidates=rows,status='VERIFIED' if valid and not bad else 'FAILED')
        report['host_libraries' if host else 'kernel_objects']=valid
    failures=[name for name,item in checks.items() if item['status'] not in ('VERIFIED','NOT_PRESENT')]
    if failures:raise ValueError('artifact verification failed: '+', '.join(failures)+'; see artifact_checks')


def record_failure(report,error):
    if report.get('cann_compile')=='BUILD_RETURNED_ZERO':
        report['status']='ARTIFACT_CHECK_FAILED_REVIEW_UNRESOLVED'
    elif report.get('cann_compile')=='RUNNING':
        report.update(status='COMPILE_FAILED_REVIEW_UNRESOLVED',cann_compile='FAILED_OR_INCOMPLETE')
    else:report['status']='BLOCKED_COMPILE_PROBE'
    report['error']=f'{type(error).__name__}: {error}'


def compile_candidate(root,cann,report):
    definition,files,config,ops=build_inputs(True,'nd',{})
    if config is not None or len(ops)!=1:raise ValueError('single raw-only operator required')
    project=root/'CustomOp';e=env(cann)
    write(root/'operators.json',definition)
    report['phase']='msopgen'
    run_build_command(['msopgen','gen','-i',root/'operators.json','-c','ai_core-Ascend310P3','-lan','cpp',
             '-out',project,'-op','DFlashW8A8Accumulator'],root/'msopgen.log',e,report,'msopgen')
    for target,source in files.items():shutil.copyfile(source,project/target)
    for sub in ('op_host','op_kernel'):
        actual={str(f.relative_to(project)) for f in (project/sub).rglob('*') if f.suffix in ('.h','.hpp','.cc','.cpp')}
        expected={f for f in files if f.startswith(sub+'/')}
        if actual!=expected:raise ValueError('unexpected generated source inventory: '+str(actual^expected))
    report['candidate']=[identity(project/t) for t in files]
    report['generated_build_script']=identity(project/'build.sh')
    report.update(phase='cann_compile',cann_compile='RUNNING');write(root/'compile-report.json',report)
    # Standard generated package build only; deliberately no install or runner command.
    run_build_command(['bash','-c','cd "$1" && bash build.sh','compile-only',project],root/'build.log',e,report,'build',3600)
    report.update(cann_compile='BUILD_RETURNED_ZERO',phase='artifact_validation')
    write(root/'compile-report.json',report)
    for item in report['candidate']:verify(item)
    verify(report['generated_build_script'])
    check_artifacts(project,report)
    report['build_metadata']=[identity(p) for p in project.rglob('*') if p.is_file() and
        p.name in ('CMakeCache.txt','compile_commands.json','CMakePresets.json')]
    report['packages_not_installed']=[identity(p) for p in (project/'build_out').glob('custom_opp_*.run')]
    report.update(status='COMPILE_COMPLETED_REVIEW_UNRESOLVED',phase='complete',cann_compile='COMPLETED')


def main(args):
    root=fresh(args.output_dir,[args.cann_root,args.preflight.parent])
    report=dict(status='UNRESOLVED',scope='RAW_ND_K128_COMPILE_ONLY',phase='identity_check',
        source_review='UNRESOLVED',cann_compile='NOT_RUN',host_tiling_runtime='NOT_RUN',
        installation='NOT_RUN',npu='NOT_RUN',performance='NOT_RUN',
        shape=[16,128,64],input_count=2,probe_sources=[identity(HERE/n) for n in ('compile_raw.py','compile_raw.sh')])
    try:
        p=validate_preflight(args.preflight,args.preflight_sha256,args.cann_root)
        report['preflight']=identity(args.preflight)
        if sys.platform!='linux' or platform.machine() not in ('aarch64','arm64'):raise ValueError('Linux aarch64 CANN receiver required')
        additions=root/'sdk-supplement';additions.mkdir();report['sdk_supplement']=[]
        for relative in SUPPLEMENT:
            source=args.cann_root/'aarch64-linux/asc'/relative
            if args.cann_root not in source.resolve().parents:raise ValueError('SDK include escapes selected root')
            item=identity(source)
            if item['bytes']>16*1024*1024:raise ValueError('oversized fixed SDK include')
            snapshot=additions/source.name;snapshot.write_bytes(source.read_bytes())
            snap=identity(snapshot)
            if snap['sha256']!=item['sha256']:raise ValueError('SDK include changed during snapshot')
            report['sdk_supplement'].append(dict(source=item,snapshot=snap))
        compile_candidate(root,args.cann_root,report)
        validate_preflight(args.preflight,args.preflight_sha256,args.cann_root)
    except Exception as e:
        record_failure(report,e)
    finally:
        try:
            for item in report['probe_sources']:verify(item)
            for pair in report.get('sdk_supplement',[]):verify(pair['source']);verify(pair['snapshot'])
        except Exception as e:report.update(status='BLOCKED_COMPILE_PROBE',integrity_error=str(e))
        report['logs']=[identity(p) for p in sorted(root.glob('*.log'))]
        write(root/'compile-report.json',report)
    print(report['status']+': '+str(root/'compile-report.json'))
    return int(report['status']!='COMPILE_COMPLETED_REVIEW_UNRESOLVED')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--preflight',type=Path,required=True)
    ap.add_argument('--preflight-sha256',required=True)
    ap.add_argument('--cann-root',type=Path,default=Path('/usr/local/Ascend/cann-9.0.0'))
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args();args.cann_root=args.cann_root.resolve()
    raise SystemExit(main(args))
