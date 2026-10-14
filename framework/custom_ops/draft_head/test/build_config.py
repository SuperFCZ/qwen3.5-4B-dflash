#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from common import HERE,sha256,source_hashes,write_json

CONVERSIONS={'cast':0,'minmax':1,'compare_select':2}

def configure(header,output,cap,conversion="compare_select",diagnostics=False,top1="vector",profile="none",top1_probe=False):
    if top1 not in ("scalar","vector") or profile not in ("none","matmul","convert"): raise ValueError("invalid top1/profile")
    if profile!="none" and top1!="scalar": raise ValueError("profile requires scalar identity")
    if top1=="vector" and conversion!="compare_select": raise ValueError("Vector Top1 requires accepted conversion")
    if conversion not in CONVERSIONS: raise ValueError("invalid conversion")
    if type(cap) is not int or not 0<=cap<=64: raise ValueError('core cap must be 0..64')
    if type(diagnostics) is not bool or type(top1_probe) is not bool: raise ValueError('invalid probe flags')
    if top1_probe and top1!='vector': raise ValueError('Top1 probe requires Vector')
    operators=['d_flash_draft_lm_head_top1','d_flash_draft_lm_head_top1_audit']
    if diagnostics: operators+=['d_flash_head_probe','d_flash_cast_probe']
    if top1_probe: operators+=['d_flash_top1_probe']
    kind='release' if conversion=='compare_select' and profile=='none' and not diagnostics and not top1_probe else 'experiment'
    template=HERE/'op_host/head_config.h'
    if header.resolve()==template.resolve(): raise ValueError('configure generated header only')
    text=template.read_text(); marker='#define DFLASH_HEAD_CORE_LIMIT 0U'
    if text.count(marker)!=1: raise ValueError('invalid default config')
    header.write_text(text.replace(marker,f'#define DFLASH_HEAD_CORE_LIMIT {cap}U').replace(
        "#define DFLASH_HEAD_SATURATION 2",f"#define DFLASH_HEAD_SATURATION {CONVERSIONS[conversion]}").replace(
        "#define DFLASH_HEAD_VECTOR_TOP1 1",f"#define DFLASH_HEAD_VECTOR_TOP1 {int(top1=='vector')}").replace(
        "#define DFLASH_HEAD_PROFILE_STAGE 0",f"#define DFLASH_HEAD_PROFILE_STAGE {('none','matmul','convert').index(profile)}"))
    write_json(output,dict(abi='dflash-head-build-v1',core_limit=cap,conversion=conversion,top1=top1,profile=profile,build_kind=kind,operators=operators,top1_probe=top1_probe,diagnostics=diagnostics,header=str(header.resolve()),header_sha256=sha256(header),
                           source_sha256=source_hashes(),npu_execution='NOT_RUN',full_draft_validation='NOT_RUN',end_to_end='NOT_RUN'))

def load(path):
    value=json.loads(Path(path).read_text()); cap=value.get('core_limit')
    if value.get('abi')!='dflash-head-build-v1' or type(cap) is not int or not 0<=cap<=64:
        raise ValueError('invalid head build identity')
    conversion=value.get('conversion')
    if conversion not in CONVERSIONS or type(value.get('diagnostics')) is not bool: raise ValueError('invalid conversion identity')
    top1=value.get('top1'); profile=value.get('profile')
    if top1 not in ('scalar','vector') or profile not in ('none','matmul','convert'): raise ValueError('invalid top1/profile identity')
    probe=value.get('top1_probe')
    if type(probe) is not bool or (probe and top1!='vector'): raise ValueError('invalid Top1 probe identity')
    expected=['d_flash_draft_lm_head_top1','d_flash_draft_lm_head_top1_audit']
    if value['diagnostics']: expected+=['d_flash_head_probe','d_flash_cast_probe']
    if probe: expected+=['d_flash_top1_probe']
    kind='release' if conversion=='compare_select' and profile=='none' and not value['diagnostics'] and not probe else 'experiment'
    if value.get('build_kind')!=kind or value.get('operators')!=expected: raise ValueError('operator inventory/build kind changed')
    header=Path(value['header'])
    for marker in (f"#define DFLASH_HEAD_VECTOR_TOP1 {int(top1=='vector')}",
                   f"#define DFLASH_HEAD_PROFILE_STAGE {('none','matmul','convert').index(profile)}"):
        if marker not in header.read_text(): raise ValueError('top1/profile config changed')
    if f"#define DFLASH_HEAD_SATURATION {CONVERSIONS[conversion]}" not in header.read_text(): raise ValueError('conversion config changed')
    if sha256(header)!=value['header_sha256'] or f'#define DFLASH_HEAD_CORE_LIMIT {cap}U' not in header.read_text():
        raise ValueError('generated head config changed')
    if value.get('source_sha256')!=source_hashes(): raise ValueError('head/shared source changed after build')
    return value
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--header',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--core-limit',type=int,default=0); p.add_argument('--conversion',choices=tuple(CONVERSIONS),default='compare_select'); p.add_argument('--diagnostics',action='store_true')
    p.add_argument('--top1',choices=('scalar','vector'),default='vector'); p.add_argument('--profile',choices=('none','matmul','convert'),default='none')
    p.add_argument('--top1-probe',action='store_true')
    a=p.parse_args(); configure(a.header,a.output,a.core_limit,a.conversion,a.diagnostics,a.top1,a.profile,a.top1_probe)
