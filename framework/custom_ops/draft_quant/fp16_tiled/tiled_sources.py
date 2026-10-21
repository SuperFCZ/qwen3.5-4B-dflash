"""Audited derived sources in a NEW msopgen project, never edit W8 sources."""
import difflib
import hashlib
from pathlib import Path

OLD='DFlashGroupQuantLinear';NEW='DFlashGateFp16Tiled'
SNAKE='d_flash_gate_fp16_tiled'
PLAN_FIELDS={'dequant_mode':'baseline_dequant_mode','scale_mode':'baseline_scale_mode',
             'selected_scale':'baseline_scale_algorithm','raw_banks':'raw_banks_reserved',
             'weight_dequantizations_per_n_tile':'weight_tiles_per_n_tile'}


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('source anchor changed: '+old[:100])
    return text.replace(old,new,1)


def section(text,begin,end):
    if text.count(begin)!=1 or text.count(end)!=1:raise ValueError('ambiguous source section')
    a=text.index(begin);b=text.index(end,a)
    return text[a:b]


def rename(text):
    return text.replace(OLD,NEW).replace('d_flash_group_quant_linear_tiling.h',SNAKE+'_tiling.h')


def host_source(original):
    text=original
    first=section(text,'bool ValidWeightInput(', 'void PrintShape(')
    text=once(text,first,'')
    start=section(text,'    const auto *xShape = context->GetInputShape(0);','    const auto *yShape = context->GetOutputShape(0);')
    text=once(text,start,'''    const int64_t m=16, k=2560, n=19456;
    if (!ValidInput(context, 0, ge::DT_FLOAT16, ge::FORMAT_ND, {16,2560}, {16,2560}) ||
        !ValidInput(context, 1, ge::DT_FLOAT16, ge::FORMAT_ND,
                    {304,10,64,256}, {304,10,64,256})) {
        std::fprintf(stderr, "DFlashGateFp16Tiled requires FP16 ND X[16,2560], Wtiles[304,10,64,256]\\n");
        DumpTensor(context,0,false); DumpTensor(context,1,false);
        return ge::GRAPH_FAILED;
    }
''')
    infer=section(text,'    // Accept the GE logical origin','    auto *y = context->GetOutputShape(0);')
    text=once(text,infer,'''    const int64_t m=16, n=19456;
    if (!IsShape(context->GetInputShape(0), {16,2560}) ||
        !IsShape(context->GetInputShape(1), {304,10,64,256})) return GRAPH_FAILED;
''')
    text=once(text,'context->GetInputDataType(1) != DT_INT8 ||\n        context->GetInputDataType(2) != DT_FLOAT16','context->GetInputDataType(1) != DT_FLOAT16')
    text=once(text,'        this->Input("w_nz").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_FRACTAL_NZ});\n        this->Input("s").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});',
                   '        this->Input("w_tiles").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});')
    # All platform, UB, Cube tiling calls and capacity checks stay byte-identical.
    begin='    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());'
    end='    // Account using real tile sizes/depths'
    if section(text,begin,end)!=section(original,begin,end):raise ValueError('Cube Host recipe changed')
    text=rename(text)
    for tag in ('LAUNCH','GATE_PREFETCH','GATE_COPY','SCALE','K_TILE','CUBE'):
        text=once(text,'"DFLASH_GROUP_QUANT_'+tag+' ', '"DFLASH_GATE_FP16_PLAN_'+tag+' ')
    for old,new in PLAN_FIELDS.items():text=text.replace(r'\"'+old+r'\"',r'\"'+new+r'\"')
    text=once(text,'    return ge::GRAPH_SUCCESS;\n}', '''    std::fprintf(stdout, "DFLASH_GATE_FP16_PATH {\\"layout\\":\\"NT_KT_N64_K256\\",\\"cast_calls_per_k\\":0,\\"brcb_calls_per_k\\":0,\\"mul_calls_per_k\\":0,\\"gm_weight_bytes_per_k\\":32768,\\"pipe_all_per_k\\":3}\\n");
    return ge::GRAPH_SUCCESS;
}''')
    return text


def kernel_source(original):
    text=original
    text=once(text,section(text,'__aicore__ inline void CopyRawTile(', '}  // namespace'), '')
    text=text.replace('GM_ADDR x, GM_ADDR w_nz, GM_ADDR s, GM_ADDR y','GM_ADDR x, GM_ADDR w_tiles, GM_ADDR y')
    # Exact Gate shape is enforced by Host. Do not add a device shape guard
    # that would give the compiler new constant-propagation opportunities.
    text=once(text,'constexpr uint32_t kNzK0 = 32;\nconstexpr uint32_t kNzN0 = 16;\n','')
    text=once(text,'    AscendC::GlobalTensor<half> xGm, sGm, yGm;\n    AscendC::GlobalTensor<int8_t> qGm;',
                   '    AscendC::GlobalTensor<half> xGm, wTilesGm, yGm;')
    text=once(text,'    qGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(w_nz), globalN * globalK);\n    sGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(s), (globalK / kGroup) * globalN);',
                   '    wTilesGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(w_tiles), globalN * globalK);')
    # Remove inactive next-N initialization; only fixed Gate/Off may enter here.
    block=section(text,'        if (prefetch) {\n            CopyRawTile(qBanks,','        for (uint32_t kBegin = 0;')
    text=once(text,block,'')
    text=once(text,'    auto wLocal = weightBuf.Get<half>(kTileN * tileK);','    (void)qBanks; (void)sBanks;  // Raw reservations intentionally retained.\n    auto wLocal = weightBuf.Get<half>(kTileN * tileK);')
    start='            const uint32_t bank = prefetch ?'
    end='            if constexpr (StageA) mm.SetTensorA('
    preparation=section(text,start,end)
    waits=section(original,'                const auto scalarReady = pipe.FetchEventID(', '            }\n#if DFLASH_GROUP_QUANT_DEQUANT_MODE == 1')
    replacement='''            const bool hasNext = kBegin + tileK < globalK;
            const uint32_t offset=(nTile*(globalK/tileK)+kBegin/tileK)*kTileN*tileK;
            AscendC::DataCopy(wLocal, wTilesGm[offset], kTileN*tileK);
'''+waits+'''            // Retain the Off pre-Cube fence even with no vector dequantization.
            AscendC::PipeBarrier<PIPE_ALL>();
'''
    text=once(text,preparation,replacement)
    # Exact Matmul SDK calls, partial-sum control, final output and cleanup.
    tail=section(original,end,'extern "C" __global__')
    if section(text,end,'extern "C" __global__')!=tail:raise ValueError('Matmul/accumulator/synchronization tail changed')
    prefix=section(original,'    AscendC::TPipe pipe;','    AscendC::GlobalTensor<half> xGm,')
    if section(text,'    AscendC::TPipe pipe;','    AscendC::GlobalTensor<half> xGm,')!=prefix:raise ValueError('buffer allocation/SDK registration changed')
    text=text.replace('void d_flash_group_quant_linear(',f'void {SNAKE}(')
    text=text.replace('RunGroupQuantLinear<true>(x, w_nz, s, y, workspace, tilingData);','RunGroupQuantLinear<true>(x, w_tiles, y, workspace, tilingData);')
    text=text.replace('RunGroupQuantLinear<false>(x, w_nz, s, y, workspace, tilingData);','RunGroupQuantLinear<false>(x, w_tiles, y, workspace, tilingData);')
    if any(x in text for x in ('AscendC::Cast(', 'AscendC::Brcb(', 'AscendC::Mul(', 'AscendC::Muls(', 'CopyRawTile(')):raise ValueError('dequantization survived source adaptation')
    return text


def runner_source(original):
    text=original.replace('aclnnMmGetWorkspaceSize(r.x.desc,r.w.desc,r.y.desc,0,','aclnnDFlashGateFp16TiledGetWorkspaceSize(r.x.desc,r.w.desc,r.y.desc,')
    text=text.replace('aclnnMm(', 'aclnnDFlashGateFp16Tiled(').replace('"aclnnMmGetWorkspaceSize"','"aclnnDFlashGateFp16TiledGetWorkspaceSize"').replace('"aclnnMm"','"aclnnDFlashGateFp16Tiled"')
    text=once(text,section(text,'    void Make(int64_t rows,','    void Release()'),'''    void Make(const std::vector<int64_t>& shape,const std::vector<uint8_t>* values=nullptr) {
        size_t bytes=2;for(auto d:shape)bytes*=d;
        buffer.Allocate(bytes);if(values)buffer.CopyIn(*values);
        std::vector<int64_t> stride(shape.size(),1);
        for(size_t i=shape.size();i>1;--i)stride[i-2]=stride[i-1]*shape[i-1];
        desc=aclCreateTensor(shape.data(),shape.size(),ACL_FLOAT16,stride.data(),0,ACL_FORMAT_ND,shape.data(),shape.size(),buffer.data);
        if(!desc)throw std::runtime_error("aclCreateTensor failed");
    }
''')
    text=text.replace('r.x.Make(M,K,&x);r.w.Make(K,N,&w);r.y.Make(M,N);','r.x.Make({M,K},&x);r.w.Make({304,10,64,256},&w);r.y.Make({M,N});')
    text=text.replace('w_fp16_kn','w_fp16_tiles').replace('fp16_direct_mm','dflash_gate_fp16_tiled_test')
    text=text.replace('AscendCL native ACLNN','AscendCL custom FP16 ACLNN').replace(r'\"op\":\"aclnnMm\"',r'\"op\":\"DFlashGateFp16Tiled\"')
    text=text.replace(r'\"weight_layout\":\"ND_KN\",\"cube_math_type\":0',r'\"weight_layout\":\"NT_KT_N64_K256\",\"tile_n\":64,\"tile_k\":256')
    text=text.replace('// KEEP_DTYPE=0, FP16 inputs/output. No dtype downgrade or CPU path.','// Prepacked FP16 tiles, same W8 Off Cube recipe; no dequantization.')
    return text.replace('// Independent native FP16 ACLNN baseline; never linked to a custom OPP.','// Isolated custom FP16 tiled comparator; generated from the guarded native runner.')


def generate(here,project,runner_dir):
    runner_dir.mkdir(parents=True,exist_ok=False)
    originals={kind:(here/path).read_text() for kind,path in dict(host='op_host/d_flash_group_quant_linear.cpp',kernel='op_kernel/d_flash_group_quant_linear.cpp',tiling='op_host/d_flash_group_quant_linear_tiling.h',runner='fp16_compare/native_mm.cpp').items()}
    results=dict(host=host_source(originals['host']),kernel=kernel_source(originals['kernel']),
        tiling=rename(originals['tiling']).replace('D_FLASH_GROUP_QUANT_LINEAR_TILING_H','D_FLASH_GATE_FP16_TILED_TILING_H'),runner=runner_source(originals['runner']))
    files=dict(host=project/'op_host'/f'{SNAKE}.cpp',kernel=project/'op_kernel'/f'{SNAKE}.cpp',tiling=project/'op_host'/f'{SNAKE}_tiling.h',runner=runner_dir/'main.cpp')
    audit={}
    for kind,text in results.items():
        files[kind].write_text(text)
        diff=''.join(difflib.unified_diff(originals[kind].splitlines(True),text.splitlines(True),fromfile='accepted/'+kind,tofile='isolated/'+kind))
        (runner_dir/(kind+'.diff')).write_text(diff)
        audit[kind]=dict(original_sha256=hashlib.sha256(originals[kind].encode()).hexdigest(),derived_sha256=hashlib.sha256(text.encode()).hexdigest(),path=str(files[kind]),diff=str(runner_dir/(kind+'.diff')))
    return audit
