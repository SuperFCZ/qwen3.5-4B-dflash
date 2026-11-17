"""Exact transforms: change only independent Host scheduling, plus evidence capture."""
from scheduler_common import c,HERE,PD
from worker_derivation import replaced


def host_text():
    s=(c.QUANT/'op_host/d_flash_group_quant_linear.cpp').read_text()
    s=replaced(s,'#include "tiling/tiling_api.h"','#include "tiling/tiling_api.h"\n#include "scheduler_variant.h"\n#include "scheduler_evidence.h"')
    old='    const uint32_t blockDim = draft_quant_contract::LaunchBlockCount(n, availableCores, coreLimit);'
    new='''    if (m!=16 || k!=12800 || n!=2560) return ge::GRAPH_FAILED;
    const uint32_t baselineBlockDim = draft_quant_contract::LaunchBlockCount(n, availableCores, coreLimit);
    dflash_scheduler::Plan scheduling;
    if (!dflash_scheduler::Make(platform,availableCores,baselineBlockDim,static_cast<uint32_t>(n/kTileN),coreLimit,scheduling)) {
        std::fprintf(stderr,"DFLASH_SCHEDULER_CONTRACT_BLOCKED generic=%u aic=%u aiv=%u slices=%u calc=%u soc=%u arch=%u\\n",availableCores,scheduling.aic,scheduling.aiv,scheduling.slices,scheduling.calculated,static_cast<uint32_t>(platform.GetSocVersion()),static_cast<uint32_t>(platform.GetCurNpuArch()));return ge::GRAPH_FAILED;
    }
    const uint32_t blockDim = scheduling.selected;'''
    s=replaced(s,old,new)
    marker='    return ge::GRAPH_SUCCESS;\n}\n}  // namespace optiling'
    evidence='''    if (!dflash_platform_diag::Write(context->GetPlatformInfo(),platform,m,k,n,availableCores,blockDim,coreLimit,context->GetWorkspaceSizes(1)[0]) ||
        !dflash_scheduler::Write(context,scheduling,availableCores,coreLimit,userUbBytes,matmulUbBytes,l1Used,l0aUsed,l0bUsed,l0cUsed,
                                codesBytes,scaleBytes,weightBytes,rawBanks,static_cast<uint32_t>(tileK),tiling.GetDataSize())) return ge::GRAPH_FAILED;
'''
    return replaced(s,marker,evidence+marker)


def builder_text():
    s=(PD/'build_host.py').read_text()
    s=replaced(s,'from platform_common import c\nfrom derive import audit_host','from scheduler_common import c\nfrom scheduler_derive import audit_host')
    s=replaced(s,"base=prior['reports']['build']['variants'][variant]","base=prior['reports']['build']['variants']['B']")
    s=replaced(s,"(project/'op_host/platform_diag.h').write_bytes(c.verify(audit['writer']).read_bytes())", "(project/'op_host/platform_diag.h').write_bytes(c.verify(audit['platform_writer']).read_bytes())\n    (project/'op_host/scheduler_evidence.h').write_bytes(c.verify(audit['writer']).read_bytes())")
    s=replaced(s,'    after=tree(project)', "    if variant not in ('A','B'):raise ValueError('unknown scheduler variant')\n    (project/'op_host/scheduler_variant.h').write_text('#pragma once\\n#define DFLASH_SCHEDULER_POLICY '+str(int(variant=='B'))+'\\n')\n    after=tree(project)")
    s=replaced(s,"set(after)-set(before)!={'op_host/platform_diag.h'}", "set(after)-set(before)!={'op_host/platform_diag.h','op_host/scheduler_evidence.h','op_host/scheduler_variant.h'}")
    s=s.replace('HOST_EVIDENCE_ONLY_DERIVED_OPP','HOST_SCHEDULER_EXPERIMENT_NOT_DEVICE_ADMITTED')
    s=s.replace('PLATFORM_DIAG_HOST_DSOS_BUILT_NPU_NOT_RUN','SCHEDULER_HOST_DSOS_BUILT_NPU_NOT_RUN')
    return s


def audit_host():
    p=HERE/'op_host/d_flash_group_quant_linear.cpp'
    if p.read_text()!=host_text():raise ValueError('Host differs beyond scheduling/evidence transform')
    if (HERE/'runner/main.cpp').read_bytes()!=(PD/'runner/main.cpp').read_bytes():raise ValueError('Host-only Runner changed')
    return dict(original=c.identity(c.QUANT/'op_host/d_flash_group_quant_linear.cpp'),candidate=c.identity(p),
                writer=c.identity(HERE/'op_host/scheduler_evidence.h'),platform_writer=c.identity(PD/'op_host/platform_diag.h'))


def verify():
    audit_host()
    if (HERE/'scheduler_build.py').read_text()!=builder_text():raise ValueError('builder transform changed')
