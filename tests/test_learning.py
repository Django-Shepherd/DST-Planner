"""Contract, masking, permutation, gradient, and actor export checks."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'learning'))
from dst_planner.features import observation,collate,FEATURE_CONFIG
from dst_planner.model import DST,ModelConfig
from dst_planner.train import losses
from dst_planner.predict import Predictor,export


def fixture():
    return dict(feature_version=2,gain_definition='observed_voxel_raycast_v1',graph_version=7,decision_id=7,
                position=[0,0,1],current_node_id=0,nodes=[
                    dict(id=i,position=p,is_current=i==0) for i,p in enumerate([[0,0,1],[1,0,1],[0,2,1],[3,3,1]])],
                edges=[[0,1,1.],[1,0,1.],[0,2,2.],[2,0,2.],[2,3,3.],[3,2,3.]],
                candidates=[dict(id=i,node_id=i+1,position=p) for i,p in enumerate([[1,0,1],[0,2,1],[3,3,1]])],
                information_gain_m3=[10,20,30],reachability_dimension=4,
                reachable=[0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0],expert_route=[0,2,1,3])

class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_teacher_information_is_not_input(self):
        a=fixture();b=copy.deepcopy(a);b['expert_route']=[0,3,1,2];b['cost_matrix']=[999]*16;b['selected_candidate_id']=2
        self.assertTrue(np.array_equal(observation(a).nodes,observation(b).nodes))
        self.assertTrue(np.array_equal(observation(a).reachable,observation(b).reachable))
    def test_finite_gradients_all_branches(self):
        torch.manual_seed(17);m=DST(ModelConfig(hidden=32,graph_layers=1,history_layers=1,diffusion_steps=4));t=copy.deepcopy(m).eval().requires_grad_(False)
        o=observation(fixture());samples=[dict(obs=o,next_obs=o,action=0,expert=[1,0,2],reward=-2.,terminated=False,truncated=False),
                                        dict(obs=o,next_obs=o,action=1,expert=[1,2,0],reward=97.,terminated=True,truncated=False)]
        loss,logs=losses(m,t,samples,samples,'cpu');self.assertTrue(torch.isfinite(loss));loss.backward()
        for name in ['encoder','actor','critic','diffusion']:
            grads=[p.grad for p in getattr(m,name).parameters() if p.grad is not None]
            self.assertTrue(grads,name);self.assertTrue(all(torch.isfinite(g).all() for g in grads),name)
            self.assertGreater(sum(float(g.abs().sum()) for g in grads),0,name)
        self.assertTrue(all(p.grad is None for p in t.parameters()))
    def test_short_sequence_and_no_legal_action(self):
        m=DST(ModelConfig(hidden=32,graph_layers=1,history_layers=1)).eval();o=observation(fixture());b=collate([o])
        b['reachable'][:,0,2:]=False;b['reachable'][:,1,1:]=False
        with torch.no_grad():seq,_,mask=m.actor.rollout(m.encoder(b),b)
        self.assertEqual(seq.tolist(),[[0,-1,-1,-1,-1]]);self.assertEqual(int(mask.sum()),1)
        b['reachable'].zero_()
        with torch.no_grad():seq,logp,mask=m.actor.rollout(m.encoder(b),b)
        self.assertFalse(mask.any());self.assertTrue(torch.isfinite(logp).all())
    def test_graph_permutation_equivariance(self):
        torch.manual_seed(11);m=DST(ModelConfig(hidden=32,graph_layers=2,history_layers=1)).eval()
        o=observation(fixture());perm=np.array([2,0,3,1]);inverse=np.argsort(perm);q=copy.deepcopy(o)
        q.nodes=o.nodes[perm];q.edges=inverse[o.edges];q.current=int(inverse[o.current]);q.candidates=inverse[o.candidates]
        with torch.no_grad():
            a=collate([o]);b=collate([q]);ha=m.encoder(a);hb=m.encoder(b)
            la,_=m.actor.logits(m.actor.context(ha,a),a,torch.empty(1,0,dtype=torch.long))
            lb,_=m.actor.logits(m.actor.context(hb,b),b,torch.empty(1,0,dtype=torch.long))
        torch.testing.assert_close(la,lb,atol=2e-5,rtol=2e-5)
    def test_export_matches_training_actor(self):
        torch.manual_seed(13);m=DST(ModelConfig(hidden=32,graph_layers=1,history_layers=1)).eval();r=fixture();b=collate([observation(r)])
        with torch.no_grad():expected=m.actor.rollout(m.encoder(b),b)[0][0].tolist()
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'train.pt';out=Path(d)/'actor.pt'
            torch.save(dict(model=m.state_dict(),model_config=asdict(m.config),feature_config=FEATURE_CONFIG,step=1,dataset_sha256='test'),src)
            export(src,out);predictor=Predictor(out);result=predictor.predict(r)
            self.assertEqual(result['route'],[x for x in expected if x>=0])
            ck=torch.load(out,weights_only=True);self.assertNotIn('critic',ck);self.assertNotIn('diffusion',ck)

if __name__=='__main__':unittest.main()
