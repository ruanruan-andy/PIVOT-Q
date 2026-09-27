"""CPU-only regression checks for paper selectors, no model imports needed."""
import ast
import random
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pivot_q'))
from sparse_selection import select_sparse, temporal_phase

# Extract unchanged legacy functions without importing GR00T/PEFT.
tree=ast.parse((Path(__file__).resolve().parents[1]/'pivot_q/selector.py').read_text())
names={'percentile_ranks','select_phase','_legacy_select_states'}
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+functions,type_ignores=[])
ns=dict(torch=torch,random=random,SimpleNamespace=SimpleNamespace)
exec(compile(ast.fix_missing_locations(module),'<legacy>','exec'),ns)
legacy,ranks,phase=[ns[n] for n in ['_legacy_select_states','percentile_ranks','select_phase']]
cfg=SimpleNamespace(phase_bins=4,priority_per_phase=4,random_per_phase=0,min_temporal_gap=4,
    alpha_q=.5,beta_r=.5,weight_min=.5,weight_max=2.)
for length in [16,17,31,52,72,103,220,280,300,520]:
    g=torch.Generator().manual_seed(length)
    q,r=torch.rand(length,generator=g),torch.rand(length,generator=g)
    rng=random.Random(0)
    cfg.selection_method='pivot_q'; cfg.selection_weighting='vulnerability'
    a=legacy(q,r,cfg,rng); b=select_sparse(q,r,cfg,rng,legacy,ranks,phase)
    for key in vars(a):
        assert torch.equal(getattr(a,key),getattr(b,key)) if isinstance(getattr(a,key),torch.Tensor) else getattr(a,key)==getattr(b,key),key
    for method in ['random_sparse','uniform_sparse']:
        cfg.selection_method=method; cfg.selection_weighting='uniform'
        before=rng.getstate()
        a=select_sparse(q,r,cfg,rng,legacy,ranks,phase)
        b=select_sparse(q*0,r*0,cfg,rng,legacy,ranks,phase)
        assert a.indices==b.indices and rng.getstate()==before
        assert len(set(a.indices))==16 and a.phase_counts==[4]*4
        assert torch.equal(a.weights/a.weights.sum(),torch.full((16,),1/16))
        for p,gap in enumerate(a.phase_effective_gaps):
            ix=[i for i in a.indices if min(3,i*4//length)==p]
            assert all(j-i>=gap for i,j in zip(ix,ix[1:]))
        cfg.selection_weighting='vulnerability'
        c=select_sparse(q,r,cfg,rng,legacy,ranks,phase)
        assert c.indices==a.indices and torch.isfinite(c.weights).all()
assert temporal_phase([0,2,5,9,12,16],4,4)[0]==[0,5,12,16]
print('PASS: legacy equality, quotas, gaps, weighting, repeatability, RNG isolation and sparse valid indices')
