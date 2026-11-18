import os
from pathlib import Path
import select
import shlex
import signal
import socket
import struct
import subprocess
import sys
import time
from profile_common import HERE,FLAGS,c,flow,resume,require


def sender(conn):
    size=struct.calcsize('3i')
    data,items,flags,_=conn.recvmsg(1,socket.CMSG_SPACE(size))
    require(data==b'?' and not(flags & socket.MSG_CTRUNC),'no original Runner gate token')
    creds=[struct.unpack('3i',val) for level,typ,val in items if level==socket.SOL_SOCKET and typ==socket.SCM_CREDENTIALS]
    require(len(creds)==1 and creds[0][1]==os.getuid(),'missing/wrong Runner SCM_CREDENTIALS')
    return creds[0][0]


def run(out,b,gear,r):
    case=b['cases'][gear];spec=case['spec'];selected=b['selected'];env=b['env']
    c.verify_opp(selected['opp']);flow.host.envs.assert_invocation(env,selected['env'])
    runner=selected['om']['runner'];require(c.dependencies(runner,out/'runner-ldd.log',env)==selected['om']['deps'],'Runner loader drift')
    records=resume.combined(selected,b['cann'],b['extras'],c.read(resume.POLICY))
    def maps(path):return resume.host.dsos.maps_rows(path.read_text())
    folder=out/'application';folder.mkdir()
    for side in ('inputs','outputs'):
        (folder/(side+'.tsv')).write_text(''.join(x['name']+' '+x['dtype']+' '+' '.join(map(str,x['shape']))+'\n' for x in spec['abi'][side]))
    for side in ('inputs','golden'):
        dest=folder/side;dest.mkdir()
        for name,item in spec[side].items():(dest/(name+'.bin')).symlink_to(c.verify(item))
    require(c.identity(b['msprof_alias'])==b['msprof'],'msprof retargeted')
    c.command([b['msprof_alias'],'--help'],out/'msprof-help.log',env)
    helptext=(out/'msprof-help.log').read_text()
    require(all(flag.split('=')[0] in helptext for flag in FLAGS),'missing CLI capability; inspect help')
    inspect_cmd=[str(c.verify(b['atc'])),'--mode=1','--om='+str(c.verify(case['om'])),'--json='+str(out/'om-graph.json')]
    c.write(out/'graph-inspect-command.json',inspect_cmd)
    with (out/'graph-inspect.log').open('xb') as inspection:
        subprocess.run(inspect_cmd,cwd=out,env=dict(env,ASCEND_WORK_PATH=str(out),AI_RUN_DIR=str(out)),stdout=inspection,stderr=subprocess.STDOUT,check=True)
    r['graph']=c.identity(out/'om-graph.json')
    parent,child=socket.socketpair()
    with parent,child,(out/'msprof.log').open('xb') as log:
        parent.setsockopt(socket.SOL_SOCKET,socket.SO_PASSCRED,1);fd=child.fileno()
        argv=list(map(str,[c.verify(runner),0,c.verify(case['om']),folder,folder/'inputs',folder/'golden','benchmark',fd]))
        endpoint=os.readlink('/proc/self/fd/'+str(fd))
        request=out/'request.json';c.write(request,dict(argv=argv,endpoint=endpoint))
        cmd=[b['msprof_alias'],*FLAGS,'--output='+str(out/'profiler'),'--application='+shlex.join([sys.executable,'-I','-B',str(HERE/'exec_shim.py'),str(request)])]
        c.write(out/'msprof-command.json',dict(argv=cmd,env=env,pass_fds=[fd],runner=runner,om=case['om']))
        with subprocess.Popen(cmd,pass_fds=(fd,),cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True) as proc:
            child.close()
            try:
                deadline=time.monotonic()+600
                while not select.select([parent],[],[],1)[0]:
                    require(proc.poll() is None and time.monotonic()<deadline,'msprof exited/timed out before original gate (see msprof.log)')
                pid=sender(parent);shim=c.read(folder/'shim.json');task=Path('/proc')/str(pid)
                require(shim==dict(pid=pid,fd=fd,endpoint=endpoint,argv=argv) and os.readlink(task/'fd'/str(fd))==endpoint and
                        c.identity(task/'exe')==runner and (task/'cmdline').read_bytes().split(b'\0')[:-1]==[os.fsencode(arg) for arg in argv],'exec/PID/FD/argv identity differs')
                actual_env=dict(entry.split('=',1) for entry in os.fsdecode((task/'environ').read_bytes()).split('\0') if entry)
                for key in ('ASCEND_CUSTOM_OPP_PATH','ASCEND_OPP_PATH','LD_LIBRARY_PATH','LD_PRELOAD','LD_AUDIT',
                            'ASCEND_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES'):
                    require(actual_env.get(key)==env.get(key),'profiled Runner environment drift: '+key)
                live=maps(task/'maps');require(live==maps(folder/'pre-maps.txt'),'live and Runner pre-maps differ')
                resume.check_maps(live,records)
                r['runner_pid']=pid;r['launch_authorized']=True;parent.sendall(b'G');parent.close()
                require(proc.wait(timeout=1800)==0,'msprof/application failure')
            finally:
                parent.close()
                if proc.poll() is None:
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    resume.check_maps(maps(folder/'post-maps.txt'),records)
    row=dict(execution=c.identity(folder/'execution.json'),outputs={name:[c.identity(folder/(name+'-'+suffix)) for suffix in ('actual-0.bin','actual-1.bin','benchmark-last.bin','postcheck.bin')] for name in flow.OUTPUTS})
    validation=flow.verify_run(row,spec,0,True)
    require(validation['workspace_bytes']==case['workspace'] and validation['model_weight_bytes']==case['weights'],'profile allocation drift')
    r.update(validation=validation,output_evidence=row)
