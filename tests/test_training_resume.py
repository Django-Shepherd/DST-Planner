"""Exercise the real training CLI: pause/resume must exactly reproduce CPU updates."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
from test_learning import fixture
from dst_planner.features import observation,FEATURE_CONFIG

class ResumeTest(unittest.TestCase):
    def test_exact_cpu_resume_and_contract(self):
        with tempfile.TemporaryDirectory(prefix='dst-resume-test-') as temp:
            root=Path(temp);o=observation(fixture())
            samples=[dict(obs=o,next_obs=o,action=0,expert=[1,0,2],reward=-2.,terminated=False,truncated=False),
                     dict(obs=o,next_obs=o,action=1,expert=[1,2,0],reward=97.,terminated=True,truncated=False)]
            dataset=dict(train=samples,validation=samples,bc_train=samples,bc_validation=samples,feature_config=FEATURE_CONFIG)
            torch.save(dataset,root/'data.pt')
            base=[sys.executable,'-m','dst_planner.train','--dataset',str(root/'data.pt'),'--device','cpu',
                  '--steps','8','--batch-size','4','--micro-batch','2','--hidden','32','--eval-every','4','--threads','2']
            env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]/'learning'))
            for name,extra in [('full',[]),('split',['--stop-after','4']),('split',['--resume',str(root/'split/last.pt')])]:
                result=subprocess.run(base+['--output',str(root/name)]+extra,env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                if '--stop-after' in extra:self.assertFalse((root/name/'completion.json').exists())
            full=torch.load(root/'full/last.pt',map_location='cpu');split=torch.load(root/'split/last.pt',map_location='cpu')
            for branch in ['model','target']:
                for key,value in full[branch].items():self.assertTrue(torch.equal(value,split[branch][key]),key)
            self.assertEqual(full['scheduler'],split['scheduler'])
            self.assertEqual(full['rank_rng'][0]['python'],split['rank_rng'][0]['python'])
            self.assertTrue(torch.equal(full['rank_rng'][0]['torch'],split['rank_rng'][0]['torch']))
            for key,state in full['optimizer']['state'].items():
                for field,value in state.items():self.assertTrue(torch.equal(value,split['optimizer']['state'][key][field]))
            altered=base.copy();altered[altered.index('--batch-size')+1]='8'
            result=subprocess.run(altered+['--output',str(root/'split'),'--resume',str(root/'split/last.pt')],env=env,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Resume must preserve batch',result.stderr)

if __name__=='__main__':unittest.main(verbosity=2)
