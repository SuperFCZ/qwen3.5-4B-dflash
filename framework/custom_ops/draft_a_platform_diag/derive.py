"""Bounded diagnostic transforms; original scheduling and device code unchanged."""
from platform_common import c,HERE,SIDECAR
from worker_derivation import replaced


def host_text():
    text=(c.QUANT/'op_host/d_flash_group_quant_linear.cpp').read_text()
    text=replaced(text,'#include "tiling/tiling_api.h"','#include "tiling/tiling_api.h"\n#include "platform_diag.h"')
    marker='    return ge::GRAPH_SUCCESS;\n}\n}  // namespace optiling'
    inject='''    if (!dflash_platform_diag::Write(context->GetPlatformInfo(),platform,m,k,n,availableCores,blockDim,coreLimit,context->GetWorkspaceSizes(1)[0])) {
        std::fprintf(stderr,"DFLASH_PLATFORM_DIAG_WRITE_FAILED\\n");
        return ge::GRAPH_FAILED;
    }
'''
    return replaced(text,marker,inject+marker)


def builder_text():
    text=(SIDECAR/'host_build.py').read_text()
    text=replaced(text,'from sidecar_common import c,audit_host','from platform_common import c\nfrom derive import audit_host')
    text=text.replace('host_evidence.h','platform_diag.h')
    needle="    after=tree(project)"
    text=replaced(text,needle,"    host_cmake=project/'op_host/CMakeLists.txt'\n    with host_cmake.open('a') as f:f.write('\\ntarget_link_libraries(cust_optiling PRIVATE ${CMAKE_DL_LIBS})\\ntarget_link_libraries(cust_op_proto PRIVATE ${CMAKE_DL_LIBS})\\n')\n"+needle)
    text=replaced(text,"changed!={'CMakeLists.txt','op_host/d_flash_group_quant_linear.cpp'}","changed!={'CMakeLists.txt','op_host/CMakeLists.txt','op_host/d_flash_group_quant_linear.cpp'}")
    text=text.replace("A_RAW_SIDECAR_HOST_OPP_BUILT_NPU_NOT_RUN","PLATFORM_DIAG_HOST_DSOS_BUILT_NPU_NOT_RUN")
    text=text.replace("Re-run 52 projections for each new B/R package; full Draft numerical/safety gates remain required","This package is Host-diagnostic-only. No new numerical acceptance is granted; no 104-probe execution is scheduled.")
    return text


def audit_host():
    p=HERE/'op_host/d_flash_group_quant_linear.cpp'
    if p.read_text()!=host_text():raise ValueError('diagnostic changed Host beyond read-only logging')
    return dict(original=c.identity(c.QUANT/'op_host/d_flash_group_quant_linear.cpp'),candidate=c.identity(p),writer=c.identity(HERE/'op_host/platform_diag.h'))


def verify():
    audit_host()
    if (HERE/'build_host.py').read_text()!=builder_text():raise ValueError('Host builder derivation differs')
