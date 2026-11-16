"""Private B/R build worker. Only invoked explicitly after precheck."""
import sys
from pathlib import Path
from types import SimpleNamespace
import raw_common as c


def main(request):
    r=c.read(request);root=Path(r['output']);root.mkdir(parents=True,exist_ok=False)
    result=dict(status='BLOCKED',npu='NOT_RUN')
    try:
        c.verify_tree(r['evidence']);c.verify_tree(r['sources'])
        from opp import build,preflight
        cfg=r['evidence'][r['variant']]['config']
        actual=build(SimpleNamespace(a_build_config=c.verify(cfg),c_release=Path(r['evidence']['H']['state']['args']['c_release']),output_dir=root/'joint'))
        if actual['a_config']!=r['evidence'][r['variant']]['configuration']:raise ValueError('built wrong A configuration')
        # Verify actual generated Host/Kernel headers, not just request flags.
        for sub in ('op_host','op_kernel'):
            p=root/'joint/CustomOp'/sub/'d_flash_group_quant_linear_build_config.h'
            if c.sha(p)!=actual['a_config']['header_sha256']:raise ValueError('generated header differs')
        copied=[]
        for sub in ('op_host','op_kernel'):
            for source in (c.QUANT/sub).iterdir():
                if source.suffix not in ('.h','.cpp') or source.name=='d_flash_group_quant_linear_build_config.h':continue
                target=root/'joint/CustomOp'/sub/source.name
                if c.sha(target)!=c.sha(source):raise ValueError('generated A source differs: '+str(target))
                copied.append(c.identity(target))
        api=Path(actual['install'])/'vendors/customize/op_api/include/aclnn_d_flash_group_quant_linear.h'
        if c.sha(api)!=r['evidence'][r['variant']]['api_header']['sha256']:raise ValueError('new ACLNN API differs from accepted Child')
        pf=preflight(Path(actual['install']))
        if pf!=actual['preflight']:raise ValueError('installed OPP differs after build')
        result.update(status='PRIVATE_OPP_COMPILED_NOT_NPU_VALIDATED',opp=actual,
            copied_a_sources=copied,header_files=[c.identity(root/'joint/CustomOp'/s/'d_flash_group_quant_linear_build_config.h') for s in ('op_host','op_kernel')],
            build_report=c.identity(root/'joint/opp.json'))
    except Exception as e:result['error']=f'{type(e).__name__}: {e}'
    c.write(root/'report.json',result)
    return int(result['status']=='BLOCKED')


if __name__=='__main__':raise SystemExit(main(Path(sys.argv[1])))
