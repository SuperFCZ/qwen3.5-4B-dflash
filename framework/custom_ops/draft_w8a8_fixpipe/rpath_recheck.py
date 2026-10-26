#!/usr/bin/env python3
"""Finalize compile artifacts by exact CMake RPATH replay on an isolated copy only."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from w8_common import HERE,identity,verify,write,fresh
from compile_raw import check_artifacts,HOST_LINK_TARGET
from artifact_recheck import inspect_original,compare_host


def cmake_tokens(text):
    """Lex generated CMake, without evaluating variables, conditions or commands."""
    tokens=[];i=0;line=1
    while i<len(text):
        c=text[i]
        if c.isspace():line+=c=='\n';i+=1;continue
        start_line=line
        if c=='#':
            bracket=re.match(r'#\[(=*)\[',text[i:])
            if bracket:
                stop=']'+bracket[1]+']';end=text.find(stop,i+len(bracket[0]))
                if end<0:raise ValueError('unterminated CMake bracket comment')
                end+=len(stop);line+=text[i:end].count('\n');i=end
            else:
                end=text.find('\n',i);i=len(text) if end<0 else end
            continue
        if c in '()':tokens.append((c,line,'paren'));i+=1;continue
        if c=='"':
            i+=1;value=''
            while i<len(text) and text[i]!='"':
                if text[i]=='\\':
                    if i+1>=len(text) or text[i+1] not in ('\\','"',';'):raise ValueError('unsupported CMake string escape')
                    i+=1
                line+=text[i]=='\n';value+=text[i];i+=1
            if i==len(text):raise ValueError('unterminated CMake string')
            i+=1;tokens.append((value,start_line,'argument'));continue
        bracket=re.match(r'\[(=*)\[',text[i:])
        if bracket:
            stop=']'+bracket[1]+']';begin=i+len(bracket[0]);end=text.find(stop,begin)
            if end<0:raise ValueError('unterminated CMake bracket argument')
            value=text[begin:end];end+=len(stop);line+=text[i:end].count('\n');i=end
            tokens.append((value,start_line,'argument'));continue
        end=i
        while end<len(text) and not text[end].isspace() and text[end] not in '()#':end+=1
        if end==i:raise ValueError('unsupported CMake token')
        tokens.append((text[i:end],line,'word'));i=end
    return tokens


def file_commands(text):
    tokens=cmake_tokens(text);out=[];i=0
    while i<len(tokens):
        if tokens[i][2]!='word' or tokens[i][0].lower()!='file' or i+1>=len(tokens) or tokens[i+1][0]!='(' or tokens[i+1][2]!='paren':
            i+=1;continue
        line=tokens[i][1];i+=2;args=[]
        while i<len(tokens) and not (tokens[i][0]==')' and tokens[i][2]=='paren'):
            if tokens[i][0]=='(' and tokens[i][2]=='paren':raise ValueError('nested file command unsupported')
            args.append(tokens[i][0]);i+=1
        if i==len(tokens):raise ValueError('unterminated file command')
        out.append(dict(args=args,line_start=line,line_end=tokens[i][1]));i+=1
    return out


def target_path(value,project,packaged):
    """Bind only package-relative CMake prefix spelling or exact literal target."""
    staged=next((p for p in packaged.parents if p.name.startswith('custom_opp_') and p.name.endswith('.run')),None)
    if staged is None or project.resolve() not in staged.resolve().parents:raise ValueError('unknown CPack staging root')
    for prefix in ('$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}','${CMAKE_INSTALL_PREFIX}'):
        if value.startswith(prefix+'/'):
            suffix=value[len(prefix)+1:]
            if any(x in suffix for x in ('$',';','\\')) or '..' in Path(suffix).parts:raise ValueError('unsafe install suffix')
            choices=[p for p in (staged,staged/'packages') if (p/suffix).absolute()==packaged.absolute()]
            if len(choices)!=1:raise ValueError('install destination not bound to this packaged library')
            return dict(raw=value,install_prefix=str(choices[0]),destdir='',resolved=str(packaged))
    if value==str(packaged):return dict(raw=value,resolved=str(packaged))
    raise ValueError('unsupported or mismatched install destination')


def extract_rule(project,canonical,packaged,expected_script_sha,cann):
    script=project/'build_out/op_host/cmake_install.cmake'
    if script.resolve(strict=True)!=script.absolute():raise ValueError('install script must be the canonical generated regular file')
    bound=identity(script)
    if bound['sha256']!=expected_script_sha:raise ValueError('install script SHA256 mismatch')
    if bound['bytes']>2*1024*1024:raise ValueError('install script exceeds bounded size')
    text=script.read_text();verify(bound);commands=file_commands(text);name=canonical.name
    changes=[c for c in commands if c['args'] and c['args'][0].upper()=='RPATH_CHANGE' and any(name in a for a in c['args'][1:])]
    if len(changes)!=1:raise ValueError('missing or multiple/conflicting Host RPATH_CHANGE rules')
    rule=changes[0];a=rule['args']
    if len(a)!=7 or a[1::2]!=['FILE','OLD_RPATH','NEW_RPATH']:raise ValueError('unsupported RPATH_CHANGE rule')
    destination=target_path(a[2],project,packaged)
    if a[4]!=str(cann/'lib64')+':' or a[6]!='':raise ValueError('only confirmed CANN lib64 colon RPATH -> empty is accepted')
    installs=[]
    for c in commands:
        x=c['args']
        if not x or x[0].upper()!='INSTALL' or not any(name in v for v in x[1:]):continue
        if len(x)!=7 or x[1]!='DESTINATION' or x[3:6]!=['TYPE','SHARED_LIBRARY','FILES']:
            raise ValueError('ambiguous Host file(INSTALL) rule')
        if x[6]!=str(canonical):raise ValueError('install source is not the canonical Host library')
        dest=target_path(x[2]+'/'+name,project,packaged)
        if dest['resolved']!=destination['resolved']:raise ValueError('install and RPATH targets differ')
        installs.append(c)
    if len(installs)!=1:raise ValueError('unique source-to-package install rule required')
    return dict(script=bound,install_rule=installs[0],rpath_rule=rule,destination=destination,
                canonical=identity(canonical),packaged=identity(packaged),old_rpath=a[4],new_rpath=a[6])


def bracket(value):
    sep='='
    while ']'+sep+']' in value:sep+='='
    return '['+sep+'['+value+']'+sep+']'


def replay(rule,cmake,out,evidence):
    out.mkdir();source=verify(rule['canonical']);package=verify(rule['packaged']);source_script=verify(rule['script'])
    source_snapshot=out/'source-install.cmake';shutil.copyfile(source_script,source_snapshot)
    if identity(source_snapshot)['sha256']!=rule['script']['sha256']:raise ValueError('install script changed during snapshot')
    evidence['source_script_snapshot']=identity(source_snapshot)
    tool=verify(cmake);copy=out/'host-replay.so';shutil.copyfile(source,copy)
    if identity(copy)['sha256']!=rule['canonical']['sha256']:raise ValueError('replay copy identity mismatch')
    script=out/'replay.cmake'
    # Never execute the original install script; no original path is a write target.
    script.write_text('file(RPATH_CHANGE\n FILE "${REPLAY_FILE}"\n OLD_RPATH '+bracket(rule['old_rpath'])+'\n NEW_RPATH '+bracket(rule['new_rpath'])+'\n)\n')
    script_id=identity(script)
    argv=[str(tool),'-DREPLAY_FILE:FILEPATH='+str(copy),'-P',str(script)]
    evidence.update(rule=rule,cmake=cmake,replay_script=script_id,argv=argv,returncode=None,status='RUNNING')
    try:
        with (out/'cmake.log').open('w') as f:
            result=subprocess.run(argv,cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=60)
        evidence['returncode']=result.returncode
    finally:
        if (out/'cmake.log').is_file():evidence['log']=identity(out/'cmake.log')
    verify(rule['canonical']);verify(rule['packaged']);verify(rule['script']);verify(cmake);verify(script_id)
    if result.returncode:raise ValueError('isolated CMake RPATH replay failed')
    evidence['replay']=identity(copy)
    if evidence['replay']['sha256']!=rule['packaged']['sha256'] or evidence['replay']['bytes']!=rule['packaged']['bytes']:
        raise ValueError('replay whole-file SHA256/size differs from package')
    comparison=compare_host(source,package);evidence['elf_comparison']=comparison
    if not comparison['program_headers_equal'] or not comparison['needed_soname_equal']:
        raise ValueError('ELF mapping or dependency identity changed')
    left,right=comparison['build'],comparison['package']
    paths=lambda v:[s['value'] for s in v['dynamic_strings'] if s['name'] in ('RPATH','RUNPATH')]
    dyn=lambda v:[(s['tag'],s['value']) for s in v['dynamic_entries'] if s['tag'] not in (15,29)]
    if paths(left)!=[rule['old_rpath']] or paths(right) not in ([],['']) or dyn(left)!=dyn(right):
        raise ValueError('unexpected dynamic metadata transformation')
    evidence.update(status='EXACT_RPATH_REPLAY_VERIFIED',exact_sha_match=True)


def main(args):
    project=args.report.parent/'CustomOp';root=fresh(args.output_dir,[args.report.parent])
    r=dict(status='ARTIFACT_CHECK_FAILED_REVIEW_UNRESOLVED',scope='RAW_ND_K128_COMPILE_ARTIFACT_FINALIZATION',
           shape=[16,128,64],input_count=2,source_review='UNRESOLVED',
           host_tiling_runtime='NOT_RUN',npu='NOT_RUN',installation='NOT_RUN',performance='NOT_RUN',recompiled=False,
           validator_sources=[identity(HERE/n) for n in ('rpath_recheck.py','compile_raw.py','artifact_recheck.py','w8_common.py','w8_run.py')])
    try:
        old,bound=inspect_original(args.report,args.report_sha256);r['original_report']=bound;r['commands']=old['commands'];r['preflight']=old['preflight']
        pf=json.loads(verify(old['preflight']).read_text());cmake=pf['tools']['cmake']['file'];verify(cmake)
        canonical=project/'build_out/op_host/libcust_opmaster_rt2.0.so'
        aliases=sorted((project/'build_out/_CPack_Packages/Linux/External').glob('custom_opp_*.run/packages/vendors/customize/op_impl/ai_core/tbe/op_tiling/liboptiling.so'))
        if len(aliases)!=1:raise ValueError('missing/ambiguous required packaged alias')
        alias=aliases[0]
        if not alias.is_symlink() or os.readlink(alias)!=HOST_LINK_TARGET:raise ValueError('package alias target differs')
        package=alias.resolve(strict=True)
        if project.resolve() not in package.parents or package!=alias.parent.resolve()/HOST_LINK_TARGET:raise ValueError('package alias escapes required location')
        if identity(canonical)['sha256']!=args.build_host_sha256 or identity(package)['sha256']!=args.package_host_sha256:
            raise ValueError('Host identities differ from supplied evidence')
        checked={}
        try:check_artifacts(project,checked)
        except ValueError as e:r['strict_byte_check_error']=str(e)
        r['prior_rule_checks']=checked
        if checked.get('kernel_objects')!=old['kernel_objects'] or checked['artifact_checks']['kernel']['status']!='VERIFIED':
            raise ValueError('Kernel VERIFIED identity changed')
        r['kernel_objects']=checked['kernel_objects'];r['kernel_status']='VERIFIED_UNCHANGED'
        # Preserve every canonical-library check. Only the precisely known build alias may be unresolved here.
        build_alias=canonical.parent/'liboptiling.so'
        for c in checked['artifact_checks']['host']['candidates']:
            if c['status'] in ('VERIFIED','EXCLUDED'):continue
            if c['path']!=str(build_alias) or not build_alias.is_symlink() or os.readlink(build_alias)!=HOST_LINK_TARGET:
                raise ValueError('unrelated invalid canonical Host artifact')
            try:build_alias.resolve(strict=True)
            except FileNotFoundError:pass
            else:raise ValueError('unexpected invalid non-dangling build alias')
        verified=[c for c in checked['artifact_checks']['host']['candidates'] if c['status']=='VERIFIED']
        if not any(c['identity']==identity(canonical) for c in verified):raise ValueError('canonical Host was not ELF verified')
        if any('liboptiling.so' in c.get('dt_needed',[]) for c in verified):raise ValueError('canonical Host requires dangling build alias')
        rule=extract_rule(project,canonical,package,args.install_script_sha256,Path(pf['cann_root']))
        r['rpath_replay']={};replay(rule,cmake,root/'replay',r['rpath_replay'])
        inspect_original(args.report,args.report_sha256)
        if os.readlink(alias)!=HOST_LINK_TARGET or alias.resolve(strict=True)!=package:raise ValueError('package alias changed')
        for x in r['kernel_objects']:verify(x)
        r.update(status='COMPILE_COMPLETED_REVIEW_UNRESOLVED',cann_compile='COMPLETED',
                 host_runtime=dict(status='VERIFIED_BY_EXACT_RPATH_REPLAY',alias=str(alias),target=HOST_LINK_TARGET,
                                   identity=identity(package)),host_libraries=checked['host_libraries'])
    except Exception as e:
        r['error']=f'{type(e).__name__}: {e}'
        if r.get('rpath_replay',{}).get('status')=='RUNNING':r['rpath_replay']['status']='FAILED'
    finally:
        try:
            for f in r['validator_sources']:verify(f)
        except Exception as e:r.update(status='ARTIFACT_CHECK_FAILED_REVIEW_UNRESOLVED',integrity_error=str(e))
        write(root/'compile-report.json',r)
    print(r['status']+': '+str(root/'compile-report.json'))
    return int(r['status']!='COMPILE_COMPLETED_REVIEW_UNRESOLVED')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--report-sha256',required=True)
    p.add_argument('--build-host-sha256',required=True);p.add_argument('--package-host-sha256',required=True)
    p.add_argument('--install-script-sha256',required=True);p.add_argument('--output-dir',type=Path,required=True)
    raise SystemExit(main(p.parse_args()))
