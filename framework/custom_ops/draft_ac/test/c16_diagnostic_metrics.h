// Host-side observation of already executed device outputs. No device math fallback.
#pragma once
#include <algorithm>
#include <array>
#include <limits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <stdexcept>
#include <vector>
namespace c16_diagnostic {
inline double DecodeHalf(uint16_t bits) {
    const double sign=(bits&0x8000)?-1.:1.; const unsigned exp=(bits>>10)&31, mant=bits&1023;
    if(exp==31) return mant?std::nan(""):sign*INFINITY;
    return sign*(exp?std::ldexp(1.+mant/1024.,int(exp)-15):std::ldexp(double(mant),-24));
}
inline double Half(uint16_t bits) {
    static const std::array<double,65536> table=[] {
        std::array<double,65536> result{};
        for(size_t i=0;i<result.size();++i) result[i]=DecodeHalf(uint16_t(i));
        return result;
    }();
    return table[bits];
}
inline uint16_t Bits(const std::vector<uint8_t>& x,size_t i) {
    return uint16_t(x[2*i])|(uint16_t(x[2*i+1])<<8);
}
struct Metrics {
    size_t count=0,mismatches=0,actualNonfinite=0,referenceNonfinite=0,nonfinitePairs=0;
    long double squares=0; std::vector<double> errors;
    void Add(uint16_t a,uint16_t b) {
        ++count; mismatches+=a!=b; const auto x=Half(a),y=Half(b);
        actualNonfinite+=!std::isfinite(x);referenceNonfinite+=!std::isfinite(y);
        if(!std::isfinite(x)||!std::isfinite(y)) { ++nonfinitePairs;return; }
        const double error=std::abs(x-y);errors.push_back(error);squares+=error*error;
    }
    void Write(std::ostream& out) {
        out<<"{\"elements\":"<<count<<",\"bit_mismatches\":"<<mismatches
           <<",\"mismatch_ratio\":"<<(count?double(mismatches)/count:0.)
           <<",\"actual_nonfinite\":"<<actualNonfinite<<",\"reference_nonfinite\":"<<referenceNonfinite
           <<",\"nonfinite_pairs\":"<<nonfinitePairs<<",\"finite_pairs\":"<<errors.size();
        if(errors.empty()) out<<",\"max_abs\":null,\"rmse\":null,\"p99_abs\":null";
        else {
            const auto maximum=*std::max_element(errors.begin(),errors.end());
            const auto index=size_t(std::ceil(.99*errors.size()))-1;
            std::nth_element(errors.begin(),errors.begin()+index,errors.end());
            out<<",\"max_abs\":"<<maximum<<",\"rmse\":"<<std::sqrt(double(squares/errors.size()))<<",\"p99_abs\":"<<errors[index];
        }
        out<<",\"error_domain\":\"finite pairs only; nonfinite counts never discarded\"}";
    }
};
inline void KV(std::ostream& out,const std::vector<uint8_t>& actual,const std::vector<uint8_t>& reference,
               const std::vector<uint8_t>& input,const std::vector<int64_t>& shape,size_t start,size_t valid,size_t gear) {
    if(shape.size()!=4||shape[2]<=0||shape[3]<=0||actual.size()!=reference.size()||actual.size()!=input.size()||actual.size()%2)
        throw std::runtime_error("diagnostic KV shape/size differs");
    size_t elements=1;
    for(auto d:shape) {
        if(d<=0 || elements>std::numeric_limits<size_t>::max()/size_t(d)) throw std::runtime_error("KV element overflow");
        elements*=size_t(d);
    }
    if(elements!=actual.size()/2) throw std::runtime_error("KV element inventory differs");
    const auto capacity=size_t(shape[2]),dim=size_t(shape[3]);
    if(start>capacity||gear>capacity-start||valid>gear) throw std::runtime_error("diagnostic KV bounds");
    Metrics all,prefix,written,padding,unchangedPrefix,unchangedTail;
    for(size_t i=0;i<actual.size()/2;++i) {
        const auto row=(i/dim)%capacity;const auto a=Bits(actual,i),b=Bits(reference,i);
        all.Add(a,b);
        if(row<start+valid) prefix.Add(a,b);
        if(row>=start && row<start+valid) written.Add(a,b);
        if(row>=start+valid && row<start+gear) padding.Add(a,b);
        if(row<start) unchangedPrefix.Add(a,Bits(input,i));
        if(row>=start+gear) unchangedTail.Add(a,Bits(input,i));
    }
    out<<"{\"against\":\"historical_native_reference_not_accepted_golden\",\"all\":";all.Write(out);
    out<<",\"valid_prefix\":";prefix.Write(out);out<<",\"written_valid\":";written.Write(out);
    out<<",\"written_padding\":";padding.Write(out);
    out<<",\"untouched_prefix_vs_input\":";unchangedPrefix.Write(out);
    out<<",\"untouched_tail_vs_input\":";unchangedTail.Write(out);out<<'}';
}
inline void Tokens(std::ostream& out,const std::vector<uint8_t>& actual,const std::vector<uint8_t>& reference) {
    if(actual.size()!=reference.size()||actual.size()%8) throw std::runtime_error("token reference size");
    size_t mismatches=0;for(size_t i=0;i<actual.size();i+=8) mismatches+=std::memcmp(actual.data()+i,reference.data()+i,8)!=0;
    out<<"{\"bitwise_equal\":"<<(mismatches?"false":"true")<<",\"mismatches\":"<<mismatches<<",\"actual_ids\":[";
    for(size_t i=0;i<actual.size();i+=8) { int64_t value;std::memcpy(&value,actual.data()+i,8);if(i)out<<',';out<<value; }
    out<<"]}";
}
}
