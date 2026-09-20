"""Exercise real C++ transport against valid, stale, duplicate, and stalled servers."""
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
class PolicyTransportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='dst-transport-');cls.root=Path(cls.tmp.name);cls.binary=cls.root/'probe'
        subprocess.run(['g++','-std=c++14','-O1','-I/usr/include/eigen3','-I'+str(ROOT/'src/global_planner/exploration_manager/include'),
          str(ROOT/'tests/policy_client_probe.cpp'),str(ROOT/'src/global_planner/exploration_manager/src/policy_client.cpp'),'-o',str(cls.binary)],check=True,capture_output=True)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def run_case(self,result,delay=0.):
        address=str(self.root/'service.sock');server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);server.bind(address);server.listen(1)
        def respond():
            conn,_=server.accept()
            with conn:
                conn.recv(4096);time.sleep(delay)
                try:conn.sendall(json.dumps(result).encode()+b'\n')
                except BrokenPipeError:pass
        worker=threading.Thread(target=respond);worker.start()
        try:return subprocess.run([str(self.binary),address],capture_output=True,text=True,timeout=3)
        finally:worker.join();server.close();Path(address).unlink()
    def test_valid(self):
        p=self.run_case(dict(ok=True,request_id=7,graph_version=7,route=[2,0,1]));self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(p.stdout,'0 3 1 2 ')
    def test_stale(self):
        p=self.run_case(dict(ok=True,request_id=6,graph_version=7,route=[0]));self.assertNotEqual(p.returncode,0);self.assertIn('stale',p.stderr)
    def test_duplicate(self):
        p=self.run_case(dict(ok=True,request_id=7,graph_version=7,route=[0,0]));self.assertNotEqual(p.returncode,0)
    def test_bounded_timeout(self):
        p=self.run_case(dict(ok=True,request_id=7,graph_version=7,route=[0]),delay=.3);self.assertNotEqual(p.returncode,0);self.assertIn('timeout',p.stderr)
if __name__=='__main__':unittest.main()
