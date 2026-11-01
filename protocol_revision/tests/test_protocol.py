import unittest,sys,random,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from same_rule import global_matching,greedy,candidates,NS
from extraction_legacy import extract_runs
class ProtocolTests(unittest.TestCase):
    def test_global_avoids_greedy_loss(self):
        r=[(0,100),(200,300)];p=[(-10,20),(50,200)]
        self.assertEqual(len(greedy(r,p,60)),1);self.assertEqual(len(global_matching(r,p,60)),2)
    def test_maximum_overlap_after_cardinality(self):
        r=[(0,10),(8,20)];p=[(0,9),(10,20)]
        self.assertEqual(global_matching(r,p,0),[(0,0),(1,1)])
    def test_inclusive_contact_and_just_outside(self):
        self.assertEqual(global_matching([(0,10)],[(70,80)],60),[(0,0)])
        self.assertEqual(global_matching([(0,10)],[(71,80)],60),[])
    def test_no_boundary_closeness_requirement(self):
        self.assertEqual(global_matching([(10,20)],[(0,200)],60),[(0,0)])
    def test_split_gap(self):
        e=extract_runs([0,10,50,60],[1,1,1,0],10,30,100)
        self.assertEqual([(x.start,x.end) for x in e],[(0,20),(50,60)])
    def test_gap_exact_threshold_does_not_split(self):
        e=extract_runs([0,30,40],[1,1,0],10,30,100)
        self.assertEqual([(x.start,x.end) for x in e],[(0,40)])
    def test_final_sample_capped(self):
        e=extract_runs([0,10],[0,1],10,30,15)
        self.assertEqual([(x.start,x.end) for x in e],[(10,15)])
    def test_negative_closes_at_cadence(self):
        e=extract_runs([0,20],[1,0],10,30,100)
        self.assertEqual(e[0].end,10)
    def test_midnight_is_not_split(self):
        day=86400*NS;e=extract_runs([day-10*NS,day,day+10*NS],[1,1,0],10*NS,30*NS,day+20*NS)
        self.assertEqual(len(e),1)
    def test_global_matches_exhaustive_oracle(self):
        rng=random.Random(20261101)
        for case in range(100):
            r=[(s:=rng.randrange(30),s+rng.randrange(1,15)) for _ in range(4)]
            p=[(s:=rng.randrange(30),s+rng.randrange(1,15)) for _ in range(4)]
            edges=candidates(r,p,3);weights={(i,j):w for w,i,j in edges};best=(0,0)
            def walk(i,used,n,w):
                nonlocal best
                if i==len(r):best=max(best,(n,w));return
                walk(i+1,used,n,w)
                for j in range(len(p)):
                    if j not in used and (i,j) in weights:walk(i+1,used|{j},n+1,w+weights[i,j])
            walk(0,set(),0,0);got=global_matching(r,p,3)
            self.assertEqual((len(got),sum(weights[x] for x in got)),best)
if __name__=='__main__':unittest.main()
