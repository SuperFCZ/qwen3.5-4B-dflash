from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'draft_a_raw_atc'))
from split_atc import c,ec
__all__=['HERE','c','ec','sources','transformed_host','audit_host']


def sources():
    return [c.identity(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]


def transformed_host(text):
    from worker_derivation import replaced
    text=replaced(text,'#include <cstdio>','#include <cstdio>\n#include "host_evidence.h"')
    first='    std::fprintf(stdout, "DFLASH_GROUP_QUANT_LAUNCH '
    text=replaced(text,first,'    dflash_host_evidence::Batch evidence;\n'+first)
    for kind in ('LAUNCH','GATE_PREFETCH','GATE_COPY','SCALE','K_TILE','CUBE'):
        text=replaced(text,'std::fprintf(stdout, "DFLASH_GROUP_QUANT_'+kind+' ',
            'evidence.Record("'+kind+'", "DFLASH_GROUP_QUANT_'+kind+' ')
    marker='    return ge::GRAPH_SUCCESS;\n}\n}  // namespace optiling'
    text=replaced(text,marker,'    if (!evidence.Commit()) return ge::GRAPH_FAILED;\n'+marker)
    return text


def audit_host():
    original=c.QUANT/'op_host/d_flash_group_quant_linear.cpp';candidate=HERE/'op_host/d_flash_group_quant_linear.cpp'
    if candidate.read_text()!=transformed_host(original.read_text()):raise ValueError('Host change exceeds logging-only transform')
    return dict(original=c.identity(original),candidate=c.identity(candidate),writer=c.identity(HERE/'op_host/host_evidence.h'),
        scope='Only six evidence outputs and failure return; math, Tiling, ABI, Workspace unchanged')
