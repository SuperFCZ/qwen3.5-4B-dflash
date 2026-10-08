#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from common import HERE,sha256,source_hashes,write_json

def configure(header,output,cap):
    if type(cap) is not int or not 0<=cap<=64: raise ValueError('core cap must be 0..64')
    template=HERE/'op_host/head_config.h'
    if header.resolve()==template.resolve(): raise ValueError('configure generated header only')
    text=template.read_text(); marker='#define DFLASH_HEAD_CORE_LIMIT 0U'
    if text.count(marker)!=1: raise ValueError('invalid default config')
    header.write_text(text.replace(marker,f'#define DFLASH_HEAD_CORE_LIMIT {cap}U'))
    write_json(output,dict(abi='dflash-head-build-v1',core_limit=cap,header=str(header.resolve()),header_sha256=sha256(header),
                           source_sha256=source_hashes(),npu_execution='NOT_RUN',full_draft_validation='NOT_RUN',end_to_end='NOT_RUN'))

def load(path):
    value=json.loads(Path(path).read_text()); cap=value.get('core_limit')
    if value.get('abi')!='dflash-head-build-v1' or type(cap) is not int or not 0<=cap<=64:
        raise ValueError('invalid head build identity')
    header=Path(value['header'])
    if sha256(header)!=value['header_sha256'] or f'#define DFLASH_HEAD_CORE_LIMIT {cap}U' not in header.read_text():
        raise ValueError('generated head config changed')
    if value.get('source_sha256')!=source_hashes(): raise ValueError('head/shared source changed after build')
    return value
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--header',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--core-limit',type=int,default=0); a=p.parse_args(); configure(a.header,a.output,a.core_limit)
