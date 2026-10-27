#pragma once
#include <cstdint>
#include <cstdio>

// Optional identifiers are diagnostic only; their numeric values are not admission rules.
template<class P> auto S4AProbeSoc(P& p,int) -> decltype(static_cast<long long>(p.GetSocVersion()),void()) {
    std::fprintf(stderr,"STAGE4A_SOC_PROBE soc=%lld\n",static_cast<long long>(p.GetSocVersion()));
}
template<class P> void S4AProbeSoc(P&,long) { std::fprintf(stderr,"STAGE4A_SOC_PROBE soc=API_UNAVAILABLE\n"); }
template<class P> auto S4AProbeArch(P& p,int) -> decltype(static_cast<long long>(p.GetCurNpuArch()),void()) {
    std::fprintf(stderr,"STAGE4A_ARCH_PROBE arch=%lld\n",static_cast<long long>(p.GetCurNpuArch()));
}
template<class P> void S4AProbeArch(P&,long) { std::fprintf(stderr,"STAGE4A_ARCH_PROBE arch=API_UNAVAILABLE\n"); }

struct S4ACoreSnapshot { uint32_t aic,aiv,all; };
template<class P> S4ACoreSnapshot S4AProbeCores(P& platform) {
    const S4ACoreSnapshot result{platform.GetCoreNumAic(),platform.GetCoreNumAiv(),platform.GetCoreNum()};
    std::fprintf(stderr,"STAGE4A_CORE_PROBE aic=%u aiv=%u all=%u\n",result.aic,result.aiv,result.all);
    S4AProbeSoc(platform,0);S4AProbeArch(platform,0);std::fflush(stderr);
    return result;
}
