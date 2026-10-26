#!/usr/bin/env python3
"""One-shot, source-cited review writer. No build/install/device execution."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile

DEFAULT_REPO = Path(__file__).resolve().parents[4]

EXPECTED_PREFLIGHT = '312b1a0962ac6dce5a848243031a32f1898351b043faf60ba91aaabf54eab386'
ASC = 'aarch64-linux/asc/'
M = 'impl/adv_api/detail/matmul/'
# Audited content fingerprints, NOT identities copied from an old manifest.
# Every emitted path/hash/quote is obtained anew from the selected current Preflight.
SPECS = {
 'chip': (M+'feature_trait/matmul_chip_cap.h', '4274aec91fa3ff088aff38caa7a4f3922b51757f0b6710cd1a850769d9c727c8'),
 'utils': (M+'utils/matmul_utils.h', 'e9f2379b709666fccb6a1f8885a275c17827bfc469d4b0c625d2ab7a705c1eb4'),
 'mx': (M+'utils/mx_matmul_utils.h', '7b8de5eb9ad12e10016da2af6ffe661bff130afbf7725d1b70f42913daea4625'),
 'cfg': ('include/adv_api/matmul/tiling.h', 'a9a55c80deb72b6b2b8bc1561ca17b13889dd91fa40950a0f70a02b37659da1f'),
 'mmad': (M+'stage/compute/mmad_compute.h', 'c5f5dd39c23f845d4a8e4088030c6e173bf0273370d81152819432a952c66c5b'),
 'policy': (M+'policy/matmul_policy.h', '4f418fa553c628a83e2056fe7d2ef5b01097d2743588e729f78c20fe7f56b133'),
 'private': (M+'policy/matmul_private_modules.h', 'e1934286246684af49b155a77ea9952ce94fb63c4439f57ff568ee24312c2fbd'),
 'input': (M+'stage/copy_cube_in/base/copy_cube_in_using_ub.h', '4f57a5faf40eca08996956691ffe91be90f90ab4500707bb28ce43b85dce1b8e'),
 'tile': (M+'stage/copy_cube_in/copy_tile_to_cube/copy_tile_to_cube_using_ub.h', '8fed124b939a1a2477622e1414362a95222c37dbfb24ad895494cacb58b33726'),
 'nd': (M+'stage/copy_cube_in/copy_tile_to_cube/data_copy_wrapper_using_ub_nd.h', '6d5315d222c1f4357f45adbc00ad76380ec0d7d6d3bcffd570ff6cd9867c18d6'),
 'qutils': (M+'stage/copy_cube_out/quant/quant_processor_utils.h', '31467a960b31e5ad1bffb9b459f507f5a6b835fd236771af6f59b137cf9fdf1c'),
 'qbase': (M+'stage/copy_cube_out/quant/quant_processor_intf.h', 'd280aeb808633dd80d58feba31e70e28b0d46b3abe6ffb47b02eea17cb3b31fb'),
 'qcopy': (M+'stage/copy_cube_out/quant/quant_processor_datacopy.h', '10438170dde8cb131bf3ba9443955d101d5e7936aa27dc38127677cb1b898c7c'),
 'output': (M+'stage/copy_cube_out/copy_cube_out_datacopy.h', 'e572b74a29a2d09cc66f79fba676400e4900ef5ceadddf3f6abb99746ab14aad'),
 'workspace': (M+'resource/local_workspace/matmul_local_workspace.h', 'dbffdce3b74c3e18414c1c5e89e4a4260ec64654eeccc58f172ed89f7ae6c08b'),
 'iterate': (M+'matmul_impl.h', '4782d9976fbb555d01497a444b77ed05438f6829500fe557653644cc3d4f8187'),
 'hostbase': ('include/adv_api/matmul/matmul_tiling_base.h', '44dea3b20ddeea8987471f8f226cfeadbdc65140f746ee1f3bab33b8b8e3350b'),
 'hostapi': ('include/adv_api/matmul/matmul_tiling.h', 'b861ae2f87297fd744e7ec274840171882d2d875caa91ae5f4bef2d6f1f3d0c2'),
}
CITATIONS = {
 'chip': [('chip',71,101),('iterate',52,66)],
 'integer_cube': [('mx',44,47),('mx',235,273),('utils',120,122),('private',109,114),('mmad',26,66)],
 'nd_io': [('policy',34,48),('cfg',36,97),('cfg',400,400),('input',34,38),('input',77,112),('tile',61,89),('nd',136,174),('nd',253,320),('qutils',23,47),('qcopy',28,31),('qbase',73,76),('output',206,229),('output',245,290)],
 'tiling': [('hostbase',256,266),('hostbase',314,332),('hostbase',451,451),('hostapi',31,51),('cfg',36,97),('cfg',400,400)],
 'buffers': [('workspace',157,183),('workspace',214,244),('workspace',248,318),('nd',253,320),('input',61,112),('iterate',65,79)],
}
FINDINGS = {
 'chip': '静态契约支持：2002 选择 CHIP_TYPE_200 和 v200 IterateAll。绑定当前 CANN9/310P3 源码；不将工具版本查询或符号存在当作设备运行证据。',
 'integer_cube': '静态契约支持：INT8 A/B 位宽相同，GetTransBDataType 保留 INT8 B；GetL0DataType 通用模板保留 INT8，GetMmDstType<int8_t> 为 INT32；普通非稀疏路径调用 Mmad。此结论不替代 INT64 oracle 的设备逐元素验证。',
 'nd_io': '静态契约支持：CFG_NORM.doNorm=true、enVecND2NZ=false、enableUBReuse=true、enableL1CacheUB=false。非转置 A ND 使用 OnTheFly；B INT8 ND 使用 TransData 布局路径，非按权重反量化。INT8/INT32 不满足 IsQuantSenario，使用 NoQuant；C ND 走 OnTheFly。M16/K128/N64 满足该固定候选的块对齐前提，实际布局和 readonly/guards 尚未实测。',
 'tiling': '静态接口和平台来源契约支持：Host 从 TilingContext 构造 PlatformAscendC 并传入 MatmulApiTiling；INT8 A/B ND、INT32 C、无 bias，固定 M16/N64/K128。GetTiling 返回码、实际 singleCore/base 及深度容量必须由未修改的 Host 拒绝式检查验证；此处不宣称 GetTiling 已运行或必然成功。',
 'buffers': '静态契约支持：enVecND2NZ=false 选择相应 LocalWorkspace 特化，2002 接受外部 UB workspace。固定候选无 bias、CO2=4096B、A 临时=0、B 双转换缓冲=16384B，总需求20480B；前提是 Host 返回已要求的固定 tiling。存在输入队列、搬运事件及 IterateAll 复用等待，候选末尾保留 PIPE_ALL/End。实际平台容量、完成性、Guard 和重复性仍须运行验证。',
}
RAW = {
 'raw/host_common.h':'235f4880419a276f260b8ee9511b04b13b7b59fab9102e154cac465a753e03f7',
 'raw/d_flash_w8_a8_accumulator_kernel.cpp':'cdb3d831b77c0b1a6cb227bc6099e4a7f28b2febe656a8ecc01a10d3d9a90084',
}

def identity(path):
    p=Path(path).resolve(strict=True);h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())

def verify(item):
    if identity(item['path'])!=item:raise ValueError('identity mismatch: '+item['path'])
    return Path(item['path'])

def generate(root,repo):
    root=root.resolve(strict=True);directory=(repo/'framework/custom_ops/draft_w8a8_fixpipe').resolve(strict=True)
    pf=root/'preflight/preflight.json';template=root/'preflight/review-template.json';output=root/'review-raw.json'
    if output.exists():raise FileExistsError('review-raw.json already exists; it will not be overwritten')
    pf_id=identity(pf);template_id=identity(template)
    if pf_id['sha256']!=EXPECTED_PREFLIGHT:raise ValueError('wrong current Preflight SHA256')
    p=json.loads(pf.read_text());r=json.loads(template.read_text())
    if p.get('raw_only') is not True or p.get('soc')!='Ascend310P3' or p.get('status')!='UNRESOLVED' or p.get('collection')!='FIXED_FILES_COLLECTED':raise ValueError('Preflight scope/status mismatch')
    if r.get('preflight')!=pf_id or r.get('status')!='UNRESOLVED' or r.get('raw_only') is not True or set(r.get('common',{}))!=set(CITATIONS):raise ValueError('current template mismatch')
    source_ids={s['path']:s for s in p['sources']}
    for s in p['sources']:verify(s)
    for name in ('w8_sdk.py','w8_common.py'):
        f=directory/name
        if source_ids.get(str(f))!=identity(f):raise ValueError('formal validator not bound to current Preflight')
    for rel,digest in RAW.items():
        item=source_ids.get(str(directory/rel))
        if not item or verify(item) is None or item['sha256']!=digest:raise ValueError('reviewed raw candidate changed: '+rel)
    indexed={}
    for f in p['files']:
        a,b=verify(f['source']),verify(f['snapshot'])
        if f['source']['sha256']!=f['snapshot']['sha256'] or f['source']['bytes']!=f['snapshot']['bytes']:raise ValueError('SDK source/snapshot mismatch')
        if f['source']['path'] in indexed:raise ValueError('duplicate SDK source identity')
        indexed[f['source']['path']]=f
    for tool in p['tools'].values():verify(tool['file'])
    checked={}
    for key,(relative,audited_hash) in SPECS.items():
        path=str((Path(p['cann_root'])/ASC/relative).resolve(strict=True))
        f=indexed.get(path)
        if not f:raise ValueError('required SDK file missing from CURRENT Preflight: '+path)
        if f['source']['sha256']!=audited_hash:raise ValueError('current SDK content differs from reviewed excerpt: '+path)
        checked[key]=(f,verify(f['snapshot']).read_text(encoding='utf-8').splitlines())
    for claim,ranges in CITATIONS.items():
        citations=[]
        for key,lo,hi in ranges:
            f,lines=checked[key]
            if not 1<=lo<=hi<=len(lines):raise ValueError('invalid citation range: '+key)
            citations.append(dict(path=f['source']['path'],sha256=f['source']['sha256'],start=lo,end=hi,quote='\n'.join(lines[lo-1:hi])))
        r['common'][claim]=dict(status='SUPPORTED',finding=FINDINGS[claim],citations=citations)
    r.update(status='SOURCE_REVIEWED',reviewer='Codex: fixed Raw ND K128 static source-contract review; exact current snapshot citations validated',
             generator=identity(Path(__file__)),current_template=template_id,candidate_sources=[source_ids[str(directory/x)] for x in RAW],
             scope='Static source contract only; SOURCE_REVIEWED is not Host runtime or NPU PASS',
             host_tiling_runtime='NOT_RUN',installation='NOT_RUN',npu='NOT_RUN',performance='NOT_RUN',
             guards='NOT_RUN',readonly='NOT_RUN',repeat='NOT_RUN',numerical_oracle='NOT_RUN',
             runtime_obligations=['GetTiling return code and exact base/singleCore','actual UB/L1/L0 and buffer depth checks','six-case INT64 oracle equality','guards/readonly/poison/repeat'])
    sys.path.insert(0,str(directory))
    sdk=importlib.import_module('w8_sdk')
    if Path(sdk.__file__).resolve()!=directory/'w8_sdk.py':raise ValueError('unexpected formal validator import')
    common=importlib.import_module('w8_common')
    if Path(common.__file__).resolve()!=directory/'w8_common.py':raise ValueError('unexpected validator helper import')
    payload=json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    # Temporary candidate disappears on any validation error. No existing artifact is edited.
    with tempfile.TemporaryDirectory(prefix='.raw-review-check-',dir=root) as temp:
        candidate=Path(temp)/'candidate.json';candidate.write_text(payload,encoding='utf-8');candidate.chmod(0o600)
        sdk.reviewed(pf,candidate,'nd',raw_only=True)
        verify(pf_id);verify(template_id)
        for f in p['files']:verify(f['source']);verify(f['snapshot'])
        os.link(candidate,output,follow_symlinks=False)  # atomic, exclusive, same filesystem
    print('SOURCE_REVIEWED (static contract only):',output)
    print('SHA256:',identity(output)['sha256'])
    print('Host Tiling runtime / installation / NPU: NOT_RUN; Build and Smoke were NOT invoked.')
    return output

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--repo',type=Path,default=DEFAULT_REPO)
    args=ap.parse_args()
    try:generate(args.root,args.repo)
    except Exception as e:
        issue=dict(status='UNRESOLVED',error=f'{type(e).__name__}: {e}',host_tiling_runtime='NOT_RUN',npu='NOT_RUN',build_invoked=False)
        if args.root.is_dir():
            with tempfile.NamedTemporaryFile(mode='w',prefix='review-unresolved-',suffix='.json',dir=args.root,delete=False,encoding='utf-8') as f:
                json.dump(issue,f,ensure_ascii=False,indent=2);print('UNRESOLVED evidence:',f.name)
        print(issue['error'],file=sys.stderr);raise SystemExit(1)
