"""Linux only. Never opens a replacement control channel."""
import json
import os
import socket
import sys
from pathlib import Path


def main(path):
    q=json.loads(path.read_text());argv=q['argv'];fd=int(argv[-1]);folder=Path(argv[3])
    if not os.get_inheritable(fd) or os.readlink('/proc/self/fd/'+str(fd))!=q['endpoint']:
        raise RuntimeError('msprof did not inherit the original gate FD')
    endpoint=socket.socket(fileno=fd)
    if endpoint.family!=socket.AF_UNIX or endpoint.type!=socket.SOCK_STREAM:raise ValueError('wrong gate socket')
    endpoint.detach()
    with (folder/'shim.json').open('x') as report:
        json.dump(dict(pid=os.getpid(),fd=fd,endpoint=q['endpoint'],argv=argv),report)
    with (folder/'runner.log').open('xb',buffering=0) as log:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
    os.execv(argv[0],argv)


if __name__=='__main__':main(Path(sys.argv[1]))
