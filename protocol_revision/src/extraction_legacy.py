"""Continuous UTC event scoring; exact nanoseconds and rational IoU costs.

Maximum-cardinality one-to-one matching, then maximum total IoU. Equal-cost
solutions are resolved by fixed sorted vertex/edge iteration (not epsilon).
No fitted clock correction or physical-truth interpretation is implied.
"""
from dataclasses import dataclass
from fractions import Fraction
from collections import deque

NS = 1_000_000_000

@dataclass(frozen=True, order=True)
class Event:
    start: int
    end: int
    id: str
    def __post_init__(self):
        if self.end <= self.start:
            raise ValueError(f'Non-positive interval: {self.id}')
    @property
    def duration(self): return self.end - self.start

def iou(a, b):
    overlap = max(0, min(a.end, b.end) - max(a.start, b.start))
    return Fraction(overlap, a.duration + b.duration - overlap)

def eligible(a, b, mode, value):
    if mode == 'iou':
        v = Fraction(str(value))
        if not 0 < v <= 1: raise ValueError('IoU threshold must be in (0,1].')
        return iou(a,b) >= v
    if mode == 'endpoints':
        v = Fraction(str(value)) * NS
        if v < 0: raise ValueError('Endpoint tolerance must be nonnegative.')
        return abs(b.start-a.start) <= v and abs(b.end-a.end) <= v
    raise ValueError(f'Unknown mode {mode!r}')

def merge_intervals(events, prefix='REF'):
    merged=[]; membership={}
    for e in sorted(events):
        if merged and e.start <= merged[-1].end:
            p=merged[-1]; merged[-1]=Event(p.start,max(p.end,e.end),p.id)
            membership[p.id].append(e.id)
        else:
            new=Event(e.start,e.end,f'{prefix}{len(merged)+1:04d}')
            merged.append(new); membership[new.id]=[e.id]
    return merged,membership

def extract_runs(times_ns, positives, cadence_ns, max_gap_ns, window_end, prefix='SRAG'):
    """Prior baseline event envelopes, no calendar cuts, split acquisition gaps.

Boundary support ends at min(next negative observation, last+median cadence).
Ordinary positive sample runs span observed sample times. Conservative support
coverage is computed separately; envelopes are not all directly observed time.
"""
    if cadence_ns <= 0 or max_gap_ns < cadence_ns: raise ValueError('Invalid cadence/gap')
    if len(times_ns) != len(positives): raise ValueError('Lengths differ')
    out=[]; begin=None; last=None; prev=None
    def close(end):
        nonlocal begin
        end=min(end,window_end)
        if begin is not None and end > begin:
            out.append(Event(int(begin),int(end),f'{prefix}{len(out)+1:04d}'))
        begin=None
    for t,p in zip(times_ns,positives):
        t=int(t)
        if prev is not None and t <= prev: raise ValueError('Times must be strictly increasing')
        if begin is not None and t-last > max_gap_ns: close(last+cadence_ns)
        if p:
            if begin is None: begin=t
            last=t
        elif begin is not None: close(min(t,last+cadence_ns))
        prev=t
    if begin is not None: close(last+cadence_ns)
    return out

def score(reference, predictions, mode, value):
    """Exact min-cost maximum flow, component-wise; stable deterministic ties.

Fraction costs avoid floating-point tie perturbations. At equal distance retain
first predecessor. Vertex order: source, sorted reference, sorted prediction,
then sink. Edge insertion is sorted by (reference,prediction). Successive
shortest residual augmentations stop only when no augmenting path remains.
"""
    # Validate configuration even when one input side is empty.
    if mode not in {'iou', 'endpoints'}: raise ValueError(f'Unknown mode {mode!r}')
    v=Fraction(str(value))
    if mode=='iou' and not 0<v<=1: raise ValueError('IoU threshold must be in (0,1].')
    if mode=='endpoints' and v<0: raise ValueError('Endpoint tolerance must be nonnegative.')
    refs=sorted(reference); preds=sorted(predictions)
    if len({e.id for e in refs}) != len(refs) or len({e.id for e in preds}) != len(preds):
        raise ValueError('Event IDs must be unique within each side')
    candidates=[(i,j,iou(a,b)) for i,a in enumerate(refs) for j,b in enumerate(preds)
                if eligible(a,b,mode,value)]
    # Connected components reduce cost without changing global optimum.
    adjacency={}
    for i,j,_ in candidates:
        adjacency.setdefault(('r',i),set()).add(('p',j));adjacency.setdefault(('p',j),set()).add(('r',i))
    components=[]; seen=set()
    for vertex in sorted(adjacency):
        if vertex in seen: continue
        todo=[vertex]; comp=set();seen.add(vertex)
        while todo:
            u=todo.pop();comp.add(u)
            for v in sorted(adjacency[u]):
                if v not in seen:seen.add(v);todo.append(v)
        components.append(comp)
    chosen=[]
    for comp in components:
        ri=sorted(v[1] for v in comp if v[0]=='r'); pj=sorted(v[1] for v in comp if v[0]=='p')
        rmap={v:k+1 for k,v in enumerate(ri)}; pmap={v:k+1+len(ri) for k,v in enumerate(pj)}
        sink=len(ri)+len(pj)+1; n=sink+1; graph=[[] for _ in range(n)];tracked=[]
        def edge(u,v,cost):
            f=[v,len(graph[v]),1,cost]; b=[u,len(graph[u]),0,-cost]
            graph[u].append(f);graph[v].append(b);return f
        for i in ri:edge(0,rmap[i],Fraction(0))
        for i,j,q in candidates:
            if i in rmap and j in pmap: tracked.append((i,j,edge(rmap[i],pmap[j],-q)))
        for j in pj:edge(pmap[j],sink,Fraction(0))
        while True:
            dist=[None]*n;prev=[None]*n;dist[0]=Fraction(0)
            for _ in range(n-1):
                changed=False
                for u in range(n):
                    if dist[u] is None:continue
                    for k,e in enumerate(graph[u]):
                        v,_,cap,cost=e
                        if cap and (dist[v] is None or dist[u]+cost < dist[v]):
                            dist[v]=dist[u]+cost;prev[v]=(u,k);changed=True
                if not changed:break
            if dist[sink] is None:break
            v=sink
            while v:
                u,k=prev[v];e=graph[u][k];e[2]-=1;graph[v][e[1]][2]+=1;v=u
        chosen.extend((i,j) for i,j,e in tracked if e[2]==0)
    pairs=sorted((refs[i],preds[j]) for i,j in chosen)
    matched_r={r.id for r,p in pairs};matched_p={p.id for r,p in pairs}
    nr,np=len(refs),len(preds);tp=len(pairs)
    precision=tp/np if np else None;recall=tp/nr if nr else None
    f1=2*tp/(nr+np) if nr+np else None
    return {'pairs':pairs,'unmatched_reference':[x for x in refs if x.id not in matched_r],
            'unmatched_prediction':[x for x in preds if x.id not in matched_p],
            'metrics':{'references':nr,'predictions':np,'TP':tp,'FP':np-tp,'FN':nr-tp,
                       'precision':precision,'recall':recall,'F1':f1,
                       'eligible_edges':len(candidates),
                       'eligible_components':len(components),
                       'ambiguous_components':sum(any(len(adjacency[v])>1 for v in c) for c in components),
                       'matched_IoU_sum':float(sum((iou(a,b) for a,b in pairs),Fraction(0)))}}
