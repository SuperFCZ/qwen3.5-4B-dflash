#ifndef DFLASH_SCHEDULER_EVIDENCE_H
#define DFLASH_SCHEDULER_EVIDENCE_H
#include <algorithm>
#include <cstdint>
#include <limits>
#include <sstream>
#include "platform_diag.h"
#ifndef DFLASH_SCHEDULER_POLICY
#error DFLASH_SCHEDULER_POLICY must be explicitly configured as 0 (A) or 1 (B)
#endif
static_assert(DFLASH_SCHEDULER_POLICY==0 || DFLASH_SCHEDULER_POLICY==1,"invalid policy");
namespace dflash_scheduler {
struct Plan {uint32_t baseline=0,aic=0,aiv=0,independentVector=0,slices=0,calculated=0,selected=0,tiles=0;};
inline bool Make(platform_ascendc::PlatformAscendC &p,uint32_t generic,uint32_t baseline,
                 uint32_t tiles,uint32_t limit,Plan &r) {
    r.baseline=baseline;r.tiles=tiles;r.aic=p.GetCoreNumAic();r.aiv=p.GetCoreNumAiv();r.independentVector=p.GetCoreNumVector();
    // This experiment is ONLY the coupled dav_2002 family, not a decoupled 1:2 launch.
    if (p.GetSocVersion()!=platform_ascendc::SocVersion::ASCEND310P ||
        static_cast<uint32_t>(p.GetCurNpuArch())!=2002 || !tiles || limit || !generic || !baseline ||
        !r.aic || r.aic!=r.aiv) return false;
    // Worker slices, not all N tiles: existing cyclic loop assigns the remainder.
    r.slices=std::min(tiles,r.aiv);
    r.calculated=p.CalcTschBlockDim(r.slices,r.aic,r.aiv);
    if (!r.calculated || r.calculated!=r.slices || r.calculated>r.aic || r.calculated>r.aiv || r.calculated>tiles ||
        p.GetCoreNum()!=generic) return false;
    r.selected=DFLASH_SCHEDULER_POLICY ? r.calculated : r.baseline;
    return r.selected>0 && r.selected<=tiles;
}
inline std::string Assignments(uint32_t tiles,uint32_t blocks) {
    std::ostringstream out;out<<'[';
    for(uint32_t b=0;b<blocks;++b){if(b)out<<',';out<<'[';bool first=true;
        for(uint32_t t=b;t<tiles;t+=blocks){if(!first)out<<',';first=false;out<<t;}out<<']';}
    out<<']';return out.str();
}
inline bool Append(const std::string &bytes) {
    const char *path=std::getenv("DFLASH_SCHEDULER_EVIDENCE_PATH");
    if(!path||path[0]!='/')return false;
    int fd=::open(path,O_WRONLY|O_APPEND|O_CLOEXEC|O_NOFOLLOW);if(fd<0)return false;
    struct stat st{};bool good=::fstat(fd,&st)==0&&S_ISREG(st.st_mode);int rc=-1;
    if(good){do{rc=::flock(fd,LOCK_EX);}while(rc<0&&errno==EINTR);good=rc==0;}
    size_t off=0;while(good&&off<bytes.size()){ssize_t n=::write(fd,bytes.data()+off,bytes.size()-off);
        if(n<0&&errno==EINTR)continue;if(n<=0){good=false;break;}off+=static_cast<size_t>(n);}
    if(good&&::fsync(fd)!=0)good=false;if(rc==0&&::flock(fd,LOCK_UN)!=0)good=false;
    if(::close(fd)!=0)good=false;return good;
}
template<class Context> inline bool Write(Context *ctx,int64_t m,int64_t k,int64_t n,const Plan&r,uint32_t generic,uint32_t limit,
    uint64_t userUb,uint64_t scratch,uint64_t l1,uint64_t l0a,uint64_t l0b,uint64_t l0c,
    uint64_t codes,uint64_t scales,uint64_t weights,uint32_t rawBanks,uint32_t tileK,size_t tilingBytes) {
    try {
        std::ostringstream out;const char *route=std::getenv("DFLASH_PLATFORM_DIAG_ROUTE");
        if(!route)return false;
        out<<"{\"schema\":1,\"variant\":\""<<(DFLASH_SCHEDULER_POLICY?"B":"A")<<"\",\"route\":"<<dflash_platform_diag::Quote(route)
           <<" ,\"m\":"<<m<<",\"k\":"<<k<<",\"n\":"<<n<<",\"tile_n\":64,\"n_tiles\":"<<r.tiles<<",\"core_limit\":"<<limit
           <<",\"generic\":"<<generic<<",\"aic\":"<<r.aic<<",\"aiv\":"<<r.aiv<<",\"independent_vector\":"<<r.independentVector
           <<",\"slice_num\":"<<r.slices<<",\"calc_tsch_block_dim\":"<<r.calculated<<",\"baseline_block_dim\":"<<r.baseline
           <<",\"selected_block_dim\":"<<ctx->GetBlockDim()<<",\"baseline_assignments\":"<<Assignments(r.tiles,r.baseline)
           <<",\"candidate_assignments\":"<<Assignments(r.tiles,r.calculated)
           <<",\"resources\":{\"user_ub\":"<<userUb<<",\"scratch\":"<<scratch<<",\"l1_used\":"<<l1<<",\"l0a_used\":"<<l0a
           <<",\"l0b_used\":"<<l0b<<",\"l0c_used\":"<<l0c<<",\"codes_bytes\":"<<codes<<",\"scale_bytes\":"<<scales
           <<",\"weight_bytes\":"<<weights<<",\"raw_banks\":"<<rawBanks<<",\"tile_k\":"<<tileK
           <<",\"workspace\":"<<ctx->GetWorkspaceSizes(1)[0]<<"},\"tiling_bytes\":"<<tilingBytes<<",\"tiling_hex\":\"";
        const auto *data=reinterpret_cast<const unsigned char*>(ctx->GetRawTilingData()->GetData());
        const char digits[]="0123456789abcdef";for(size_t i=0;i<tilingBytes;++i)out<<digits[data[i]>>4]<<digits[data[i]&15];
        out<<"\",\"approval\":\"HOST_ONLY_NOT_DEVICE_ADMITTED\"}\n";
        return ctx->GetBlockDim()==r.selected&&Append(out.str());
    }catch(...){return false;}
}
}
#endif
