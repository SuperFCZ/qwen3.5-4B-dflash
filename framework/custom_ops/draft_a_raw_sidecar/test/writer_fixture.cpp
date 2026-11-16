// CPU fixture only. Exercises the actual Host writer without any CANN/NPU call.
#include "host_evidence.h"
#include <fstream>
#include <thread>
#include <vector>
#include <atomic>
#include <iostream>
#include <chrono>

int main(int argc,char**argv) {
    std::vector<std::string> paths;
    for(int i=1;i<argc;++i)if(std::string(argv[i]).find("--output=")!=0)paths.push_back(argv[i]);
    std::atomic<int> failures{0};std::vector<std::thread> threads;
    for(size_t i=0;i<paths.size();++i)threads.emplace_back([&,i](){
        std::ifstream in(paths[i]);std::string line;dflash_host_evidence::Batch batch;
        if(!in){++failures;return;}
        while(std::getline(in,line)){
            const std::string prefix="DFLASH_GROUP_QUANT_";
            auto end=line.find(' ');auto kind=line.substr(prefix.size(),end-prefix.size());
            line+='\n';batch.Record(kind.c_str(),"%s",line.c_str());
            std::this_thread::yield();
        }
        if(!std::getenv("NO_WARN"))for(int n=0;n<31+(i<3?1:0);++n)std::printf("GMM Warning CPU fixture\n");
        if(!batch.Commit())++failures;
    });
    for(auto &t:threads)t.join();
    return failures.load()?1:0;
}
