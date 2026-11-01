"""Independent reconstruction of Hitaeshi's public candidate/greedy rule.

Uses exact integer nanoseconds. Input order is preserved for index tie breaks.
The global alternative is explicitly DIFFERENT: max cardinality then expanded
intersection duration, not a replacement silently applied to the vendor code.
"""
NS=1_000_000_000

def validate(ref,pred,tol):
    if tol<0:raise ValueError('Tolerance must be nonnegative')
    for side in [ref,pred]:
        if any(e<=s for s,e in side):raise ValueError('Intervals must have positive duration')

def candidates(ref,pred,tol=60*NS):
    validate(ref,pred,tol)
    # Inclusive endpoint contact is eligible, unlike strict half-open overlap.
    return [(max(0,min(b+tol,d)-max(a-tol,c)),i,j)
            for i,(a,b) in enumerate(ref) for j,(c,d) in enumerate(pred)
            if max(a-tol,c)<=min(b+tol,d)]

def metrics(pairs,nr,np):
    tp=len(pairs);fp=np-tp;fn=nr-tp
    p=tp/np if np else None;r=tp/nr if nr else None
    return {'tp':tp,'fp':fp,'fn':fn,'precision':p,'recall':r,'f1':2*tp/(nr+np) if nr+np else None}

def greedy(ref,pred,tol=60*NS):
    used_r=set();used_p=set();pairs=[]
    for overlap,i,j in sorted(candidates(ref,pred,tol),key=lambda x:(-x[0],x[1],x[2])):
        if i not in used_r and j not in used_p:
            pairs.append((i,j));used_r.add(i);used_p.add(j)
    return sorted(pairs)

def global_matching(ref,pred,tol=60*NS):
    edges=candidates(ref,pred,tol);adj={}
    for w,i,j in edges:
        adj.setdefault(('r',i),set()).add(('p',j));adj.setdefault(('p',j),set()).add(('r',i))
    seen=set();components=[]
    for v in sorted(adj):
        if v in seen:continue
        stack=[v];seen.add(v);comp=set()
        while stack:
            u=stack.pop();comp.add(u)
            for v2 in sorted(adj[u]):
                if v2 not in seen:seen.add(v2);stack.append(v2)
        components.append(comp)
    pairs=[]
    for comp in components:
        ri=sorted(x[1] for x in comp if x[0]=='r');pj=sorted(x[1] for x in comp if x[0]=='p')
        rm={v:k+1 for k,v in enumerate(ri)};pm={v:k+1+len(ri) for k,v in enumerate(pj)}
        sink=len(ri)+len(pj)+1;g=[[] for _ in range(sink+1)];tracked=[]
        def add(u,v,cost):
            f=[v,len(g[v]),1,cost];b=[u,len(g[u]),0,-cost];g[u].append(f);g[v].append(b);return f
        for i in ri:add(0,rm[i],0)
        for w,i,j in edges:
            if i in rm and j in pm:tracked.append((i,j,add(rm[i],pm[j],-w)))
        for j in pj:add(pm[j],sink,0)
        while True:
            dist=[None]*len(g);prev=[None]*len(g);dist[0]=0
            for _ in range(len(g)-1):
                changed=False
                for u in range(len(g)):
                    if dist[u] is None:continue
                    for k,e in enumerate(g[u]):
                        v,rev,cap,cost=e
                        if cap and (dist[v] is None or dist[u]+cost<dist[v]):
                            dist[v]=dist[u]+cost;prev[v]=(u,k);changed=True
                if not changed:break
            if dist[sink] is None:break
            v=sink
            while v:
                u,k=prev[v];e=g[u][k];e[2]-=1;g[v][e[1]][2]+=1;v=u
        pairs.extend((i,j) for i,j,e in tracked if e[2]==0)
    return sorted(pairs)
