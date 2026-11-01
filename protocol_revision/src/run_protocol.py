"""Local protocol revision, 2026-11-01. No upstream v0.2 claim."""
from pathlib import Path
import argparse,csv,json,hashlib,platform
import pandas as pd
import numpy as np
from same_rule import NS,candidates,global_matching,greedy,metrics
from extraction_legacy import extract_runs
ROOT=Path(__file__).resolve().parents[1]
def write(rows,path,columns=None):
    pd.DataFrame(rows,columns=columns).to_csv(path,index=False)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reference-events',type=Path,required=True);ap.add_argument('--prediction-events',type=Path,required=True);ap.add_argument('--raw',type=Path);ap.add_argument('--out',type=Path,default=ROOT/'results');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    R=pd.read_csv(a.reference_events);P=pd.read_csv(a.prediction_events)
    for df in [R,P]:
        if not {'event_id','start_utc','end_utc'}<=set(df.columns):raise ValueError('Required columns: event_id,start_utc,end_utc')
        if df.event_id.isna().any() or df.event_id.duplicated().any():raise ValueError('Event IDs must be present and unique')
        for col in ['start_utc','end_utc']:
            for value in df[col]:
                if pd.isna(value) or pd.Timestamp(value).tzinfo is None:raise ValueError('Timestamps must be finite and timezone-aware')
    refs=[(pd.Timestamp(x).value,pd.Timestamp(y).value) for x,y in zip(R.start_utc,R.end_utc)]
    preds=[(pd.Timestamp(x).value,pd.Timestamp(y).value) for x,y in zip(P.start_utc,P.end_utc)]
    pairs=global_matching(refs,preds,60*NS);old=greedy(refs,preds,60*NS)
    rows=[]
    for i,j in pairs:
        s,e=refs[i];p,q=preds[j];ov=max(0,min(e,q)-max(s,p));expanded=max(0,min(e+60*NS,q)-max(s-60*NS,p))
        rows.append({'reference_id':R.event_id.iloc[i],'prediction_id':P.event_id.iloc[j],'reference_start_utc':R.start_utc.iloc[i],'reference_end_utc':R.end_utc.iloc[i],'prediction_start_utc':P.start_utc.iloc[j],'prediction_end_utc':P.end_utc.iloc[j],'onset_offset_s':(p-s)/NS,'end_offset_s':(q-e)/NS,'original_overlap_s':ov/NS,'expanded_overlap_s':expanded/NS,'both_boundaries_within_60s':abs(p-s)<=60*NS and abs(q-e)<=60*NS})
    write(rows,a.out/'matched_pairs.csv', ['reference_id','prediction_id','reference_start_utc','reference_end_utc','prediction_start_utc','prediction_end_utc','onset_offset_s','end_offset_s','original_overlap_s','expanded_overlap_s','both_boundaries_within_60s'])
    offsets=[]
    for key in ['onset_offset_s','end_offset_s']:
        x=np.array([r[key] for r in rows]);absx=np.abs(x)
        if not len(x):
            offsets.append({'boundary':key,'n':0,'median_signed_s':None,'mean_absolute_s':None,'median_absolute_s':None,'q05_signed_s':None,'q95_signed_s':None,'within_60s':0});continue
        offsets.append({'boundary':key,'n':len(x),'median_signed_s':float(np.median(x)),'mean_absolute_s':float(absx.mean()),'median_absolute_s':float(np.median(absx)),'q05_signed_s':float(np.quantile(x,.05)),'q95_signed_s':float(np.quantile(x,.95)),'within_60s':int((absx<=60).sum())})
    write(offsets,a.out/'boundary_offsets_summary.csv')
    usedr={i for i,j in pairs};usedp={j for i,j in pairs}
    unmatched=[{'side':side,'event_id':row.event_id,'start_utc':row.start_utc,'end_utc':row.end_utc} for side,df,used in [('reference',R,usedr),('prediction',P,usedp)] for i,row in df.iterrows() if i not in used]
    write(unmatched,a.out/'unmatched_events.csv',['side','event_id','start_utc','end_utc'])
    edges=candidates(refs,preds,60*NS)
    summary={'date':'2026-11-01','scope':'User-supplied event arrays; descriptive agreement only','rule':'Expand each reference by 60 s each side; inclusive intersection/contact eligibility','objective':'Maximum cardinality, then maximum summed expanded overlap in integer nanoseconds','tie_break':'Deterministic ordered residual paths; no claim of unique matching in arbitrary tied inputs','metrics':metrics(pairs,len(refs),len(preds)),'reference_events':len(refs),'prediction_events':len(preds),'same_pair_ids_as_prior_greedy':pairs==old,'eligible_edges':len(edges),'competing_reference_vertices':sum(sum(i==k for _,i,j in edges)>1 for k in range(len(refs))),'competing_prediction_vertices':sum(sum(j==k for _,i,j in edges)>1 for k in range(len(preds))),'both_boundaries_within_60s':sum(r['both_boundaries_within_60s'] for r in rows),'boundary_offsets':offsets,'raw_extraction_verified':False,'upstream_v02_retrieved':False,'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__}
    if a.raw:
        h=hashlib.sha256(a.raw.read_bytes()).hexdigest()
        if h!='351af6c6734b6e70bc835b11156c2d0b087f64f71cbcc8d72c0ca7dddaed8f1f':raise ValueError('Raw source hash mismatch')
        raw=pd.read_csv(a.raw);t=pd.to_datetime(raw.UTC,utc=True,format='ISO8601').astype('int64').to_numpy();start=pd.Timestamp('2020-10-15',tz='UTC').value;end=pd.Timestamp('2020-11-08',tz='UTC').value
        mask=(t>=start)&(t<end);t=t[mask];pos=raw.SAA.to_numpy()[mask]
        if not np.all(np.diff(t)>0) or not set(pos)<= {0,1}:raise ValueError('Invalid sample series')
        cad=int(np.median(np.diff(t)));g=30*NS;extracted=extract_runs(t,pos,cad,g,end)
        assert [(e.start,e.end) for e in extracted]==preds,'Re-extracted SRAG intervals differ'
        audit=[]
        for e in extracted:
            ix=np.flatnonzero((t>=e.start)&(t<e.end)&(pos==1));last=int(ix[-1]);nxt=int(t[last+1]) if last+1<len(t) else None
            reason='analysis_end' if e.end==end else ('acquisition_gap' if nxt is not None and nxt-int(t[last])>g else ('negative_sample' if nxt is not None and pos[last+1]==0 else 'last_observed_sample'))
            audit.append({'event_id':e.id,'positive_samples':len(ix),'last_positive_utc':pd.Timestamp(int(t[last]),tz='UTC').isoformat(),'next_observation_utc':pd.Timestamp(nxt,tz='UTC').isoformat() if nxt else None,'end_utc':pd.Timestamp(e.end,tz='UTC').isoformat(),'termination_reason':reason,'terminal_extension_s':(e.end-int(t[last]))/NS,'gap_threshold_s':30,'cadence_cap_s':cad/NS})
        write(audit,a.out/'event_extraction_audit.csv')
        gaps=[{'previous_utc':pd.Timestamp(int(t[k]),tz='UTC').isoformat(),'next_utc':pd.Timestamp(int(t[k+1]),tz='UTC').isoformat(),'step_s':int(t[k+1]-t[k])/NS} for k in np.flatnonzero(np.diff(t)>g)]
        write(gaps,a.out/'acquisition_gaps.csv',['previous_utc','next_utc','step_s'])
        summary.update(raw_extraction_verified=True,raw_sha256=h,samples=len(t),cadence_s=cad/NS,gap_threshold_s=30,acquisition_gaps=len(gaps),first_sample_positive=bool(pos[0]),last_sample_positive=bool(pos[-1]),terminal_policy='End=min(next negative sample, last positive + median cadence, analysis end); gap >30 s closes before gap; no midnight split; terminal sample capped at cadence and analysis end')
    summary['input_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.reference_events,a.prediction_events]}
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
