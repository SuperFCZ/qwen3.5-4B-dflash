#ifndef DFLASH_PLATFORM_DIAG_H
#define DFLASH_PLATFORM_DIAG_H
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <sstream>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dlfcn.h>
#include "platform/platform_infos_def.h"

namespace dflash_platform_diag {
inline std::string Quote(const std::string &s) {
    std::ostringstream out;out<<'"';
    for(unsigned char c:s) {
        if(c=='"'||c=='\\')out<<'\\'<<char(c);
        else if(c<32){char b[7];std::snprintf(b,sizeof(b),"\\u%04x",unsigned(c));out<<b;}
        else out<<char(c);
    }
    out<<'"';return out.str();
}
inline bool Write(fe::PlatFormInfos *info,platform_ascendc::PlatformAscendC &p,
                  int64_t m,int64_t k,int64_t n,uint32_t available,uint32_t blocks,uint32_t limit,size_t workspace) noexcept {
    const char *path=std::getenv("DFLASH_PLATFORM_DIAG_PATH");
    if(!path)return true;
    try {
        const char *route=std::getenv("DFLASH_PLATFORM_DIAG_ROUTE");
        if(!info||path[0]!='/'||!route||(std::string(route)!="ATC"&&std::string(route)!="ACLNN")||m!=16||k!=12800||n!=2560)return false;
        uint64_t ub=0,l1=0,a=0,b=0,cc=0;
        using platform_ascendc::CoreMemType;
        p.GetCoreMemSize(CoreMemType::UB,ub);p.GetCoreMemSize(CoreMemType::L1,l1);
        p.GetCoreMemSize(CoreMemType::L0_A,a);p.GetCoreMemSize(CoreMemType::L0_B,b);p.GetCoreMemSize(CoreMemType::L0_C,cc);
        std::map<std::string,std::map<std::string,std::string>> resources;
        bool ok=info->GetPlatformResWithLock(resources);
        Dl_info dso{};int dlok=::dladdr(reinterpret_cast<void*>(&Write),&dso);
        std::ostringstream out;
        out<<"{\"schema\":1,\"route\":"<<Quote(route)<<",\"m\":"<<m<<",\"k\":"<<k<<",\"n\":"<<n
           <<",\"platform_info_present\":true,\"GetCoreNum\":"<<p.GetCoreNum()<<",\"GetCoreNumAic\":"<<p.GetCoreNumAic()
           <<",\"GetCoreNumAiv\":"<<p.GetCoreNumAiv()<<",\"soc_enum\":"<<static_cast<uint32_t>(p.GetSocVersion())
           <<",\"arch\":"<<static_cast<uint32_t>(p.GetCurNpuArch())<<",\"platform_raw_core\":"<<info->GetCoreNum()
           <<",\"platform_raw_core_locked\":"<<info->GetCoreNumWithLock()<<",\"availableCores\":"<<available
           <<",\"blockDim\":"<<blocks<<",\"coreLimit\":"<<limit<<",\"workspace\":"<<workspace
           <<",\"capacity\":{\"UB\":"<<ub<<",\"L1\":"<<l1<<",\"L0A\":"<<a<<",\"L0B\":"<<b<<",\"L0C\":"<<cc<<"}"
           <<",\"host_dso\":"<<Quote(dlok&&dso.dli_fname?dso.dli_fname:"")<<",\"resource_map_ok\":"<<(ok?"true":"false")
           <<",\"resource_map\":{";
        bool first=true;
        for(const auto &group:resources){if(!first)out<<',';first=false;out<<Quote(group.first)<<":{";bool inner=true;
            for(const auto &item:group.second){if(!inner)out<<',';inner=false;out<<Quote(item.first)<<':'<<Quote(item.second);}out<<'}';}
        out<<"}}\n";std::string bytes=out.str();
        int fd=::open(path,O_WRONLY|O_APPEND|O_CLOEXEC|O_NOFOLLOW);if(fd<0)return false;
        struct stat st{};bool good=::fstat(fd,&st)==0&&S_ISREG(st.st_mode);int rc=-1;
        if(good){do{rc=::flock(fd,LOCK_EX);}while(rc<0&&errno==EINTR);good=rc==0;}
        size_t offset=0;
        while(good&&offset<bytes.size()){ssize_t count=::write(fd,bytes.data()+offset,bytes.size()-offset);
            if(count<0&&errno==EINTR)continue;if(count<=0){good=false;break;}offset+=static_cast<size_t>(count);}
        if(good&&::fsync(fd)!=0)good=false;
        if(rc==0&&::flock(fd,LOCK_UN)!=0)good=false;
        if(::close(fd)!=0)good=false;
        return good;
    }catch(...){return false;}
}
}
#endif
