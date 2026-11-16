"""Dedicated Host channel. stdout/stderr stay unfiltered and never supply Tiling."""
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from sidecar_common import c
from split_atc import SplitAtcRunner,unique_object,CUBE_FIELDS,collect_stdout as legacy_collect

ENV='DFLASH_GROUP_QUANT_EVIDENCE_PATH'
ORDER=('LAUNCH','GATE_PREFETCH','GATE_COPY','SCALE','K_TILE','CUBE')
FIELDS={
 'LAUNCH':set('version policy dequant_mode pipeline_mode selected_pipeline tile_m weight_reuse_rows raw_banks user_ub_bytes matmul_ub_bytes m k n tile_n tile_k available_cores core_limit block_dim n_tiles system_workspace_bytes'.split()),
 'GATE_PREFETCH':{'version','mode','enabled'},'GATE_COPY':{'version','mode','selected'},
 'SCALE':{'version','scale_mode','selected_scale','broadcast_bytes','sdk_reserved_bytes'},
 'K_TILE':set('version requested_k_tile selected_k_tile selection_reason l1_bytes l1_capacity l0a_bytes l0a_capacity l0b_bytes l0b_capacity l0c_bytes l0c_capacity'.split()),
 'CUBE':CUBE_FIELDS}
STRINGS=set('policy dequant_mode pipeline_mode selected_pipeline mode selected scale_mode selected_scale selection_reason kv_m80_mode a_source'.split())


def decode_sidecar(path):
    raw=Path(path).read_bytes()
    if not raw.endswith(b'\n'):raise ValueError('sidecar missing final complete line')
    lines=raw.decode('utf-8').splitlines()
    if len(lines)!=36:raise ValueError('sidecar requires 36 records / six callbacks')
    rows=[]
    for index,line in enumerate(lines):
        row=json.loads(line,object_pairs_hook=unique_object)
        if not isinstance(row,dict) or set(row)!={'record','payload'}:raise ValueError('unknown sidecar envelope')
        kind=row['record'];data=row['payload']
        if kind!=ORDER[index%6]:raise ValueError('sidecar callback order/interleaving differs')
        if not isinstance(data,dict) or set(data)!=FIELDS[kind]:raise ValueError('sidecar field inventory differs: '+str(kind))
        for key,value in data.items():
            expected=str if key in STRINGS else bool if key=='enabled' else int
            if type(value) is not expected:raise ValueError('sidecar field type differs: '+kind+'.'+key)
        rows.append(row)
    shapes=[tuple(rows[i]['payload'][k] for k in ('m','k','n')) for i in range(0,36,6)]
    if len(set(shapes))!=6:raise ValueError('duplicate sidecar shape')
    return rows


def collect_host_evidence(identity,out,gear,cfg):
    path=c.verify(identity);rows=decode_sidecar(path)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    # Lossless schema adaptation from valid JSON values, never stdout cleanup.
    view=out/'legacy-format-view.log'
    view.write_text(''.join('DFLASH_GROUP_QUANT_'+r['record']+' '+json.dumps(r['payload'],allow_nan=False)+'\n' for r in rows))
    result=legacy_collect(view,out/'blocks',gear,cfg)
    return dict(source=identity,legacy_view=c.identity(view),plans=result['plans'],
        sidecar_schema=dict(records=36,callbacks=6,cube_records=6),
        scope='Dedicated Host JSONL; original launch_evidence resource checks preserved')


def verify_host_evidence(resource,gear,cfg):
    with tempfile.TemporaryDirectory(prefix='sidecar-verify-') as tmp:
        rebuilt=collect_host_evidence(resource['source'],Path(tmp)/'records',gear,cfg)
    for key in ('source','plans','sidecar_schema','scope'):
        if resource[key]!=rebuilt[key]:raise ValueError('sidecar resource evidence changed')
    c.verify(resource['legacy_view'])
    if resource['legacy_view']['sha256']!=rebuilt['legacy_view']['sha256']:raise ValueError('sidecar schema adapter changed')


class SidecarAtcRunner(SplitAtcRunner):
    def __call__(self,command,cwd):
        argv=list(map(str,command));cwd=Path(cwd).resolve(strict=True)
        if not argv or c.identity(argv[0])!=self.atc:raise ValueError('ATC executable identity differs')
        if self.expected_command is not None and argv!=self.expected_command:raise ValueError('ATC flags changed')
        outputs=[x.split('=',1)[1] for x in argv if x.startswith('--output=')]
        if len(outputs)!=1 or self.output_root not in Path(outputs[0]).resolve().parents:raise ValueError('ATC output outside new directory')
        run=self.root/f'call-{len(self.records)}';run.mkdir()
        stdout=run/'stdout.log';stderr=run/'stderr.log';sidecar=run/'host-evidence.jsonl'
        fd=os.open(str(sidecar),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        before=os.fstat(fd);os.close(fd)
        env=dict(os.environ);env[ENV]=str(sidecar.resolve(strict=True))
        record=dict(argv=argv,cwd=str(cwd),atc=self.atc,returncode=None,host_evidence_environment={ENV:env[ENV]})
        self.records.append(record)
        try:
            with stdout.open('xb') as out,stderr.open('xb') as err:
                result=subprocess.run(argv,cwd=cwd,env=env,stdout=out,stderr=err,check=False)
            record['returncode']=result.returncode
        finally:
            for name,p in (('stdout',stdout),('stderr',stderr),('host_evidence',sidecar)):
                if p.is_file():record[name]=c.identity(p)
            c.write(run/'command.json',record)
        after=sidecar.lstat()
        if not stat.S_ISREG(after.st_mode) or (after.st_dev,after.st_ino)!=(before.st_dev,before.st_ino):raise ValueError('sidecar was replaced')
        c.verify(self.atc)
        for raw in (stdout.read_bytes(),stderr.read_bytes()):
            if b'DFLASH_GROUP_QUANT_EVIDENCE_ERROR' in raw:raise ValueError('Host evidence write failed')
        if any(('DFLASH_GROUP_QUANT_'+name+' ').encode() in stdout.read_bytes() for name in ORDER):
            raise ValueError('selected Host did not isolate machine evidence from stdout')
        if result.returncode==0:decode_sidecar(sidecar)
        return subprocess.CompletedProcess(argv,result.returncode,stdout.read_bytes().decode('utf-8'),stderr.read_bytes().decode('utf-8'))
