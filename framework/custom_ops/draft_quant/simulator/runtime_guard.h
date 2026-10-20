#pragma once
#include <cstdlib>
#include <cstdio>
#include <dlfcn.h>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <limits.h>
#include <unistd.h>

inline std::string Resolved(const char *p) {
    char buffer[PATH_MAX];
    if (!p || !realpath(p,buffer)) throw std::runtime_error("cannot resolve simulator library");
    return buffer;
}
inline void VerifySimulatorRuntime(const std::string &directory) {
    const auto expected=Resolved(std::getenv("DFLASH_SIM_RUNTIME"));
    const auto profiler=Resolved(std::getenv("DFLASH_PROFILER_ROOT"));
    const auto driver=Resolved(std::getenv("DFLASH_SIM_DRIVER"));
    std::ifstream maps("/proc/self/maps");
    std::string all((std::istreambuf_iterator<char>(maps)),std::istreambuf_iterator<char>());
    std::ofstream(directory+"/process-maps.txt")<<all;
    if (all.find(expected)==std::string::npos || all.find(driver)==std::string::npos)
        throw std::runtime_error("simulator runtime/driver not mapped; no hardware fallback");
    std::ofstream proof(directory+"/runtime-symbols.tsv");
    for (const char *name : {"rtMalloc","rtMemcpy","rtStreamSynchronize","rtKernelLaunch"}) {
        void *symbol=dlsym(RTLD_DEFAULT,name); Dl_info info{};
        if (!symbol || !dladdr(symbol,&info)) throw std::runtime_error(std::string("unresolved runtime symbol: ")+name);
        const auto actual=Resolved(info.dli_fname);proof<<name<<"\t"<<actual<<"\t"<<(actual==expected?"camodel":"profiler_interposition")<<"\n";proof.flush();
        if(actual!=expected && actual.compare(0,profiler.size()+1,profiler+"/")!=0) throw std::runtime_error(std::string("runtime symbol is not camodel: ")+name+" -> "+actual);
    }
}
