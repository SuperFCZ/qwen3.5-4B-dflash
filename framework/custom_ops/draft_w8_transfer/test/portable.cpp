#include "impl.cpp"
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

template<class T> std::vector<T> read(const std::string& path, size_t n) {
    std::vector<T> data(n);
    std::ifstream f(path,std::ios::binary);
    if (!f.read(reinterpret_cast<char*>(data.data()),n*sizeof(T)) || f.peek()!=std::char_traits<char>::eof())
        throw std::runtime_error(path);
    return data;
}

int main(int argc,char** argv) {
    if(argc!=3) return 1;
    const std::string dir=std::string(argv[1])+"/";
    const int fail=std::stoi(argv[2]);
    auto x=read<half>(dir+(DFLASH_KV_INPUT==2?"packed":"nd"), RowsCount*K);
    auto codes=read<int8_t>(dir+"codes",K*N);
    auto scales=read<half>(dir+"scales",K*N/Group);
    auto gold=read<half>(dir+"gold",RowsCount*N);
    std::vector<half> y(RowsCount*N,half(-127));
    CpuTiling td{RowsCount,K,N,PanelK,262144-UserBytes,{N,PanelK,RowsCount,Columns,PanelK}};
    cpuOutputBase=y.data();cpuOutputWrites.assign(y.size(),0);cpuOutputOwners.assign(y.size(),-1);
    cpuBlockNum=8;matmul::cpuFailIteration=fail;
    for(cpuBlockIdx=0;cpuBlockIdx<(fail<0?8u:1u);++cpuBlockIdx)
        TEST_ENTRY(x.data(),codes.data(),scales.data(),y.data(),nullptr,&td);
    for(size_t i=0;i<y.size();++i) {
        const unsigned nt=(i%N)/Columns;
        const bool written=fail<0 || (nt%8==0 && int((nt/8+1)*Panels)<=fail);
        if (cpuOutputWrites[i]!=int(written))return 2;
        if (written && (std::memcmp(&y[i],&gold[i],2) || cpuOutputOwners[i]!=int(nt%8)))return 3;
        if (!written && y[i]!=half(-127))return 4;
    }
    if(fail>=0)return cpuMetrics.ends==1 && cpuMetrics.iterations==unsigned(fail)?0:5;
    constexpr unsigned calls=Tiles*Panels;
    constexpr unsigned casts=(Elements+128*255-1)/(128*255);
    constexpr unsigned aCopies=(Activation<RowsCount>::Whole?8:calls)*TEST_A_COPIES;
    if(cpuMetrics.iterations!=calls || cpuMetrics.partials!=Tiles*(Panels-1) || cpuMetrics.outputs!=Tiles ||
       cpuMetrics.ends!=8 || cpuMetrics.fullBarriers!=TEST_FENCES || cpuMetrics.mulCalls!=calls*Pieces*GroupsPerPart ||
       cpuMetrics.castCalls!=calls*Pieces*casts || cpuMetrics.copies!=aCopies+calls*(Pieces+1+Pieces/RawPieces))
        return 6;
    for(unsigned i=0;i<calls;++i)if(matmul::qCastsAtIterate.at(i)!=(i+1)*Pieces*casts)return 7;
    for(unsigned i=0;i<Tiles;++i) {
        const bool next=TEST_AHEAD && (i%(Tiles/8))+1<Tiles/8;
        if(matmul::qCastsAtOutput.at(i)!=(i+1)*Panels*Pieces*casts+(next?casts:0))return 8;
    }
    return 0;
}
