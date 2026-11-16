"""A-only ATC adapter: preserve stdout/stderr separately, including failures."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'draft_a_raw_env'))
import raw_env_contract as ec
c=ec.c
import raw_resources as legacy

CUBE_FIELDS=set(('version kv_m80_mode a_source a_stage_bytes a_row_stride base_m base_n base_k '
    'depth_a1 depth_b1 step_m step_n step_ka step_kb db_l0a db_l0b db_l0c trans_length weight_dequantizations_per_n_tile').split())


def sources():
    return [c.identity(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]


def unique_object(pairs):
    out={}
    for k,v in pairs:
        if k in out:raise ValueError('duplicate Host JSON key: '+k)
        out[k]=v
    return out


def stdout_schema(path):
    records=[]
    for line in Path(path).read_bytes().decode('utf-8').splitlines():
        if 'DFLASH_' not in line:continue
        part=line[line.index('DFLASH_'):]
        tag,sep,body=part.partition(' ')
        if not sep or tag not in ['DFLASH_GROUP_QUANT_'+x for x in legacy.PREFIXES]:
            raise ValueError('unknown Host record: '+tag)
        expected='DFLASH_GROUP_QUANT_'+legacy.PREFIXES[len(records)%6]
        if tag!=expected:raise ValueError('out-of-order/interleaved Host records')
        value=json.loads(body,object_pairs_hook=unique_object)
        if not isinstance(value,dict):raise ValueError('Host record is not a JSON object')
        if tag.endswith('_CUBE'):
            if set(value)!=CUBE_FIELDS:raise ValueError('CUBE field inventory differs')
            if any(type(v) is not (str if k in ('kv_m80_mode','a_source') else int) for k,v in value.items()):
                raise ValueError('CUBE field type differs')
        records.append((tag,value))
    if len(records)!=36:raise ValueError('exactly six complete Host callback blocks / CUBE JSON records required')
    shapes=[tuple(records[i][1][k] for k in ('m','k','n')) for i in range(0,36,6)]
    if len(set(shapes))!=6:raise ValueError('duplicate Host callback shape')
    return dict(callbacks=6,cube_records=6,shapes=[list(s) for s in shapes])


def collect_stdout(log,out,gear,cfg):
    schema=stdout_schema(log)
    result=legacy.collect(log,out,gear,cfg)  # Original launch_evidence and all resource gates.
    result['stdout_schema']=schema
    return result


def verify_resources(resource,gear,cfg):
    log=c.verify(resource['source'])
    with tempfile.TemporaryDirectory(prefix='a-raw-host-verify-') as tmp:
        rebuilt=collect_stdout(log,Path(tmp)/'blocks',gear,cfg)
    if rebuilt!=resource:raise ValueError('Host resource evidence differs from raw stdout')


class SplitAtcRunner:
    def __init__(self,root,atc_identity,output_root,identity_text=None,expected_command=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=False)
        self.atc=atc_identity;self.output_root=Path(output_root).resolve();self.records=[]
        self.identity_text=identity_text;self.expected_command=expected_command
    def __call__(self,command,cwd):
        argv=list(map(str,command));cwd=Path(cwd).resolve(strict=True)
        if not argv or c.identity(argv[0])!=self.atc:raise ValueError('ATC executable identity differs')
        if self.expected_command is not None and argv!=self.expected_command:raise ValueError('ATC flags changed beyond approved model/output relocation')
        outputs=[x.split('=',1)[1] for x in argv if x.startswith('--output=')]
        if len(outputs)!=1 or self.output_root not in Path(outputs[0]).resolve().parents:raise ValueError('ATC output outside new directory')
        run=self.root/f'call-{len(self.records)}';run.mkdir()
        stdout=run/'stdout.log';stderr=run/'stderr.log'
        record=dict(argv=argv,cwd=str(cwd),atc=self.atc,returncode=None)
        self.records.append(record)
        try:
            with stdout.open('xb') as out,stderr.open('xb') as err:
                result=subprocess.run(argv,cwd=cwd,stdout=out,stderr=err,check=False)
            record['returncode']=result.returncode
        finally:
            for name,p in (('stdout',stdout),('stderr',stderr)):
                if p.is_file():record[name]=c.identity(p)
            c.write(run/'command.json',record)
        c.verify(self.atc)
        # Decode only after preserving both exact byte streams and identities.
        return subprocess.CompletedProcess(argv,result.returncode,stdout.read_bytes().decode('utf-8'),stderr.read_bytes().decode('utf-8'))


def one_capture(records,command):
    if len(records)!=1 or records[0]['argv']!=list(command) or records[0]['returncode']!=0:
        raise ValueError('one successful bound ATC invocation required')
    record=records[0];c.verify_tree(record)
    if record['stdout']['path']==record['stderr']['path']:raise ValueError('ATC streams were merged')
    return record
