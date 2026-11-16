#ifndef DFLASH_HOST_EVIDENCE_H
#define DFLASH_HOST_EVIDENCE_H
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace dflash_host_evidence {
// CPU Host storage only. No device Buffer/Tiling/Workspace modification.
class Batch {
    const char* path_;
    bool good_;
    size_t count_;
    std::string bytes_;
    bool Fail(const char* reason) noexcept {
        good_=false;
        std::fprintf(stderr,"DFLASH_GROUP_QUANT_EVIDENCE_ERROR: %s errno=%d\n",reason,errno);
        return false;
    }
public:
    Batch() noexcept : path_(std::getenv("DFLASH_GROUP_QUANT_EVIDENCE_PATH")),good_(true),count_(0) {}
    void Record(const char* kind,const char* format,...) noexcept {
        if (!good_) return;
        static const char* order[]={"LAUNCH","GATE_PREFETCH","GATE_COPY","SCALE","K_TILE","CUBE"};
        if (count_>=6 || std::strcmp(kind,order[count_])!=0) { Fail("record order");return; }
        va_list args;va_start(args,format);
        struct EndArgs { va_list &value; ~EndArgs() { va_end(value); } } endArgs{args};
        if (path_==nullptr) {
            // Exact original format and arguments when the environment is absent.
            if (std::vfprintf(stdout,format,args)<0) Fail("legacy stdout write");
            ++count_;return;
        }
        try {
            va_list measure;va_copy(measure,args);int n=std::vsnprintf(nullptr,0,format,measure);va_end(measure);
            if (n<=0 || n>16384) { Fail("record size");return; }
            std::vector<char> buffer(static_cast<size_t>(n)+1);
            const int written=std::vsnprintf(buffer.data(),buffer.size(),format,args);
            if (written!=n) { Fail("record formatting");return; }
            std::string line(buffer.data(),static_cast<size_t>(n));
            std::string prefix=std::string("DFLASH_GROUP_QUANT_")+kind+" ";
            if (line.compare(0,prefix.size(),prefix)!=0 || line.back()!='\n' ||
                line.find('\n')!=line.size()-1 || line[prefix.size()]!='{' || line[line.size()-2]!='}') {
                Fail("record framing");return;
            }
            // Format conversion of our own isolated buffer, never log repair.
            bytes_+=std::string("{\"record\":\"")+kind+"\",\"payload\":"+
                line.substr(prefix.size(),line.size()-prefix.size()-1)+"}\n";
            ++count_;
        } catch (...) { Fail("record allocation"); }
    }
    bool Commit() noexcept {
        if (!good_ || count_!=6) return Fail("incomplete callback");
        if (path_==nullptr) return true;
        if (path_[0]!='/') return Fail("sidecar path must be absolute and nonempty");
        int fd=::open(path_,O_WRONLY|O_APPEND|O_CLOEXEC|O_NOFOLLOW);
        if (fd<0) return Fail("open sidecar");
        struct stat st{};
        bool locked=false,ok=true;
        if (::fstat(fd,&st)!=0 || !S_ISREG(st.st_mode)) ok=Fail("sidecar must be regular");
        if (ok) {
            int rc;
            do { rc=::flock(fd,LOCK_EX); } while(rc<0 && errno==EINTR);
            if (rc<0) ok=Fail("lock sidecar");else locked=true;
        }
        size_t offset=0;
        // Hold the lock across all SIX lines, not just each individual write.
        while(ok && offset<bytes_.size()) {
            ssize_t n=::write(fd,bytes_.data()+offset,bytes_.size()-offset);
            if(n<0 && errno==EINTR) continue;
            if(n<=0) { ok=Fail("append sidecar");break; }
            offset+=static_cast<size_t>(n);
        }
        if(ok && ::fsync(fd)!=0) ok=Fail("sync sidecar");
        if(locked && ::flock(fd,LOCK_UN)!=0) ok=Fail("unlock sidecar");
        if(::close(fd)!=0) ok=Fail("close sidecar");
        return ok;
    }
};
} // namespace dflash_host_evidence
#endif
