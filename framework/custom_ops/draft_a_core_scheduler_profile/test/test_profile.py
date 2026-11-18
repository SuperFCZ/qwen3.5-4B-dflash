import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import attribution as a
from collect_profile import sender
from profile_common import c


class ProfileContracts(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.events=[dict(ph='M',name='process_name',pid=100,args={'name':'Ascend Hardware'})]
        for i in range(38):
            for name,dt,duration,pid in [('aclmdlExecuteAsync',0,2,51),('mm',3,5,100),('aclrtSynchronizeStream',4,6,51)]:
                self.events.append(dict(name=name,ph='X',pid=pid,tid=22,ts=100*i+dt,dur=duration,
                           args=dict(TaskType='AI_CORE',TaskId='5',OpType='MatMul') if pid==100 else {}))
        self.trace=self.root/'msprof_0.json'
        (self.root/'op_summary_0.csv').write_text('Op Name,OP Type,Task Duration(us)\nmm,MatMul,999\n')
        self.graph=self.root/'graph.json';self.graph.write_text('{"op":[{"name":"mm","type":"MatMul"}]}')

    def parse(self,pid=51):
        self.trace.write_text(json.dumps(self.events))
        return a.recover(self.root,pid)[0]

    def test_provisional_only_selected_calls(self):
        t=self.parse();result=a.aggregate(self.root,t,self.graph,{'median_ms':31.,'p95_ms':32.})
        self.assertEqual(result['attribution'],'PROVISIONAL_HOST_ENCLOSURE')
        row=result['operator_types'][0]
        self.assertEqual((row['category'],row['task_count'],row['sum_us']),('Unmapped MatMul',30,150))
        self.assertEqual([x['call'] for x in result['measured_calls']],list(range(7,37)))
        self.assertEqual(result['top20_kernels'][0]['sum_us'],150)

    def test_pid_and_missing_window(self):
        with self.assertRaisesRegex(ValueError,'runner-bound'):self.parse(999)
        self.events.pop()
        with self.assertRaises(ValueError):self.parse()

    def test_device_steps_and_duration_metadata(self):
        self.events.append(dict(ph='M',pid=100,tid=99,name='thread_name',args={'name':'Step Trace(Model ID:1)'}))
        for i in range(38):
            self.events[2+i*3]['args'].update(ModelId=1,StreamId=2)
            self.events.append(dict(ph='X',pid=100,tid=99,name='Iteration '+str(i+1),cat='Iteration Time',
                ts=str(i*100+2),dur='7.0000',args={'Iteration ID':i+1,'Iteration End':str(i*100+9)}))
        self.assertEqual(self.parse()['device_assignment'],'DEVICE_STEPS_VERIFIED')
        self.events[-1]['dur']='15.0000'
        invalid=self.parse()
        self.assertNotEqual(invalid['device_assignment'],'DEVICE_STEPS_VERIFIED')
        self.assertTrue(invalid['assignment_evidence']['unresolved_tasks'])

    def test_crossing_and_duplicate(self):
        extra=dict(self.events[2]);extra['ts']=702;extra['dur']=9
        self.events.extend([dict(self.events[23]),extra])
        result=a.aggregate(self.root,self.parse(),self.graph,{'median_ms':31.,'p95_ms':32.})
        self.assertTrue(result['step_unassigned_reasons'])
        self.assertEqual(len(result['duplicate_events']),2)
        self.assertEqual(result['operator_types'][0]['task_count'],29)

    def test_categories(self):
        for kind,expected in [('SoftmaxV2','Softmax'),('DFlashGroupQuantLinear','GroupQuantLinear'),('MatMul','Unmapped MatMul'),
                              ('RmsNorm','Norm'),('KVCacheUpdate','KV Cache Update'),('FlashAttention','Attention'),('Add','Elementwise')]:
            self.assertEqual(a.category(kind,'unknown',set()),expected)

    def test_shim_missing_fd(self):
        q=self.root/'request.json';c.write(q,dict(argv=['/bin/false','0','model',str(self.root),'99'],endpoint='socket:[0]'))
        proc=subprocess.run([sys.executable,'-B',HERE/'exec_shim.py',q],capture_output=True)
        self.assertNotEqual(proc.returncode,0);self.assertFalse((self.root/'shim.json').exists())

    @unittest.skipUnless(sys.platform=='linux','Linux gate credentials')
    def test_real_shim_exec_pid_and_socket(self):
        import shutil
        from textwrap import dedent
        if not shutil.which('cc'):self.skipTest('cc absent')
        source=self.root/'fixture.c';source.write_text(dedent('''
        #include <unistd.h>
        #include <stdlib.h>
        int main(int argc,char** argv){int fd=atoi(argv[argc-1]);char b;return !(write(fd,"?",1)==1&&read(fd,&b,1)==1&&b=='G');}
        '''))
        runner=self.root/'fixture';subprocess.run(['cc',source,'-o',runner],check=True)
        p,ch=socket.socketpair()
        with p,ch:
            p.settimeout(5);p.setsockopt(socket.SOL_SOCKET,socket.SO_PASSCRED,1)
            fd=ch.fileno();q=self.root/'request.json'
            c.write(q,dict(argv=[str(runner),'0','unused',str(self.root),str(fd)],endpoint=os.readlink('/proc/self/fd/'+str(fd))))
            with subprocess.Popen([sys.executable,'-I','-B',HERE/'exec_shim.py',q],pass_fds=(fd,)) as proc:
                ch.close()
                try:
                    self.assertEqual(sender(p),proc.pid);self.assertEqual(Path('/proc',str(proc.pid),'exe').resolve(),runner)
                    self.assertEqual(c.read(self.root/'shim.json')['pid'],proc.pid)
                    p.sendall(b'G');self.assertEqual(proc.wait(timeout=5),0)
                finally:p.close()
