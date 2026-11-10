#pragma once
#include <cstdint>
#include <cstdio>

// SoC is diagnostic. Architecture must be available and equal dav_2002 for admission.
template<class P> auto S4VProbeSoc(P& p,int) -> decltype(static_cast<long long>(p.GetSocVersion()),void()) {
    std::fprintf(stderr,"STAGE4V_SOC_PROBE soc=%lld\n",static_cast<long long>(p.GetSocVersion()));
}
template<class P> void S4VProbeSoc(P&,long) { std::fprintf(stderr,"STAGE4V_SOC_PROBE soc=API_UNAVAILABLE\n"); }
template<class P> auto S4VProbeArch(P& p,int) -> decltype(static_cast<long long>(p.GetCurNpuArch())) {
    const auto arch=static_cast<long long>(p.GetCurNpuArch());
    std::fprintf(stderr,"STAGE4V_ARCH_PROBE arch=%lld\n",arch);return arch;
}
template<class P> long long S4VProbeArch(P&,long) { std::fprintf(stderr,"STAGE4V_ARCH_PROBE arch=API_UNAVAILABLE\n");return -1; }

struct S4VCoreSnapshot { uint32_t aic,aiv,all; long long arch; };
template<class P> S4VCoreSnapshot S4VProbeCores(P& platform) {
    S4VCoreSnapshot result{platform.GetCoreNumAic(),platform.GetCoreNumAiv(),platform.GetCoreNum(),-1};
    std::fprintf(stderr,"STAGE4V_CORE_PROBE aic=%u aiv=%u all=%u\n",result.aic,result.aiv,result.all);
    S4VProbeSoc(platform,0);result.arch=S4VProbeArch(platform,0);std::fflush(stderr);
    return result;
}
inline uint32_t S4VSelectSchedulingLimit(const S4VCoreSnapshot& p) {
    const bool valid=p.arch==2002&&p.aic&&p.aiv&&p.all&&p.aic==p.aiv&&p.all<=p.aic;
    const uint32_t selected=valid?p.all:0;
    std::fprintf(stderr,"STAGE4V_CORE_POLICY aic=%u aiv=%u all=%u selected=%u\n",p.aic,p.aiv,p.all,selected);
    return selected; // no max(), hard-coded core count, or alternative-source fallback
}
