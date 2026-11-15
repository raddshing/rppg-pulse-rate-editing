"""M0/M1/M2 and the pre-M3 clock-group gate, using only bytes/labels/metadata."""
from src.st5c.base import *
from src.st5c.avi_packets import packet_metadata
import numpy as np
import pandas as pd
import re


def timestamp_table(path):
    values=[];times=[]
    for line in path.read_text().splitlines():
        parts=line.split()
        if not parts:continue
        try:value=float(parts[0])
        except ValueError:value=np.nan
        values.append(value)
        if len(parts)!=3 or not np.isfinite(value):
            times.append(np.iinfo(np.int64).min);continue
        text=parts[1]+'T'+parts[2]
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?',text):
            try:stamp=np.datetime64(text,'us').astype(np.int64)
            except ValueError:stamp=np.iinfo(np.int64).min
        else:stamp=np.iinfo(np.int64).min
        times.append(stamp)
    return np.asarray(values,float),np.asarray(times,np.int64)


def describe(values,prefix):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    out={prefix+'_n':len(a)}
    for q,name in [(0,'min'),(.25,'p25'),(.5,'median'),(.75,'p75'),(1,'max')]:
        out[prefix+'_'+name]=float(np.quantile(a,q)) if len(a) else np.nan
    return out


def clock_groups(rawtimes,meta_first):
    invalid=int(np.sum(rawtimes==np.iinfo(np.int64).min))
    if invalid or meta_first==np.iinfo(np.int64).min:
        return dict(**describe([],'group_samples'),**describe([],'group_interval_seconds'),group_count=np.nan,
            group_interval_median_ge0_1=False,raw_start_minus_meta_start_seconds=np.nan,raw_end_minus_meta_start_seconds=np.nan,
            clock_a_seconds=np.nan,clock_b_seconds_per_sample=np.nan,clock_max_absolute_residual_seconds=np.nan,
            raw_clock_irregular=True,group_time_monotone=False,raw_sample_count=len(rawtimes),raw_timestamp_invalid_rows=invalid)
    starts=np.r_[0,np.flatnonzero(np.diff(rawtimes)!=0)+1]
    lengths=np.diff(np.r_[starts,len(rawtimes)]);interval=np.diff(rawtimes[starts])/1e6
    t=(rawtimes[starts]-meta_first)/1e6
    if len(starts)>=2:
        b,a=np.polyfit(starts.astype(float),t,1);residual=t-(a+b*starts)
    else:a=b=np.nan;residual=np.array([np.nan])
    finite=np.isfinite(residual);maximum=float(np.max(np.abs(residual[finite]))) if finite.any() else np.nan
    return dict(**describe(lengths,'group_samples'),**describe(interval,'group_interval_seconds'),
        group_count=len(starts),group_interval_median_ge0_1=bool(len(interval) and np.median(interval)>=.1),
        raw_start_minus_meta_start_seconds=float((rawtimes[0]-meta_first)/1e6),
        raw_end_minus_meta_start_seconds=float((rawtimes[-1]-meta_first)/1e6),clock_a_seconds=float(a),clock_b_seconds_per_sample=float(b),
        clock_max_absolute_residual_seconds=maximum,raw_clock_irregular=bool(not np.isfinite([a,b,maximum]).all() or b<=0 or maximum>.1),
        group_time_monotone=bool(np.all(interval>0)),raw_sample_count=len(rawtimes),raw_timestamp_invalid_rows=0)


def second_column(sync,index,elapsed,raw):
    if sync.ndim!=2 or sync.shape[1]!=2:return dict(status='unconfirmed_non_two_column_syntax')
    values=sync[:,1];n=len(values);integer=np.isfinite(values).all() and np.all(values==np.rint(values))
    result=dict(status='parsed',numeric=True,integer=bool(integer),finite_n=int(np.isfinite(values).sum()),
        first_value=float(values[0]),last_value=float(values[-1]),nondecreasing=bool(np.all(np.diff(values)>=0)),strictly_increasing=bool(np.all(np.diff(values)>0)),
        **describe(values,'value'),**describe(np.diff(values),'successive_difference'))
    best=None
    if len(index)>=n:
        for k in range(len(index)-n+1):
            differences=values-index[k:k+n];constants,counts=np.unique(differences,return_counts=True);j=int(np.argmax(counts))
            candidate=(int(counts[j]),k,float(constants[j]))
            if best is None or candidate[0]>best[0]:best=candidate
        result.update(meta_index_plus_constant_best_exact_fraction=best[0]/n,meta_index_plus_constant_k=best[1],meta_index_plus_constant_c=best[2])
        difference=values-elapsed[:n]
        result.update(as_relative_seconds_minus_meta_k0_median=float(np.median(difference)),as_relative_seconds_minus_meta_k0_variance=float(np.var(difference)))
    else:result.update(meta_index_plus_constant_best_exact_fraction=np.nan,meta_index_plus_constant_k=np.nan,meta_index_plus_constant_c=np.nan,
        as_relative_seconds_minus_meta_k0_median=np.nan,as_relative_seconds_minus_meta_k0_variance=np.nan)
    for origin in [0,1]:
        if integer:
            position=np.rint(values).astype(np.int64)-origin;valid=(position>=0)&(position<len(raw))
            exact=np.zeros(n,bool);exact[valid]=sync[valid,0]==raw[position[valid]]
            result.update({f'raw_sample_index_origin{origin}_in_range_fraction':float(valid.mean()),
                f'raw_sample_index_origin{origin}_exact_amplitude_fraction':float(exact.mean()),
                f'raw_sample_index_origin{origin}_exact_amplitude_fraction_valid':float(exact[valid].mean()) if valid.any() else np.nan})
        else:
            for suffix in ['in_range_fraction','exact_amplitude_fraction','exact_amplitude_fraction_valid']:result[f'raw_sample_index_origin{origin}_'+suffix]=np.nan
    return result


def load_sources():
    source_hash('reports/st5b/st5b_mcd_sources.json')
    sources=json.loads((ROOT/'reports/st5b/st5b_mcd_sources.json').read_text());root=Path(sources['original_destination'])
    check('MCD approved original root under paths.bulk',root==Path(PATHS['bulk'])/'raw/MCD-rPPG')
    check('MCD exactly6003 immutable source files',len(sources['files'])==6003)
    manifest={r['path']:r for r in sources['files']}
    source_hash(root/'db.csv');db=pd.read_csv(root/'db.csv',dtype={'patient_id':str})
    db=db[db.view.eq('front')&db.camera.eq('FullHDwebcam')].sort_values(['patient_id','step'])
    check('MCD exactly1200 unique subject/state records',len(db)==1200 and not db.duplicated(['patient_id','step']).any())
    return root,manifest,db


def main():
    check('Packet parser source committed',sha(ROOT/'src/st5c/avi_packets.py') in (ROOT/'reports/ST4_INSTR02B_PREFIX.md').read_text())
    root,manifest,db=load_sources();started=time.monotonic();frames=[];groups=[];columns=[]
    if (ROOT/'outputs/st5c_mcd_framecount.csv').exists():
        check('Resume only interrupted, not completed MCD counting',not (REPORTS/'st5c_mcd_M1_summary.json').exists())
        frames=pd.read_csv(ROOT/'outputs/st5c_mcd_framecount.csv',dtype={'subject_id':str},float_precision='round_trip').to_dict('records')
        groups=pd.read_csv(ROOT/'outputs/st5c_mcd_clock_groups.csv',dtype={'subject_id':str},float_precision='round_trip').to_dict('records')
        columns=pd.read_csv(ROOT/'outputs/st5c_mcd_sync_col2.csv',dtype={'subject_id':str},float_precision='round_trip').to_dict('records')
        check('Three partial MCD tables have identical record membership',[r['recording_id'] for r in frames]==[r['recording_id'] for r in groups]==[r['recording_id'] for r in columns])
        for r in groups:r.setdefault('raw_timestamp_invalid_rows',0)
        for r in frames:r.setdefault('meta_timestamp_invalid_rows',0)
    completed={r['recording_id'] for r in frames}
    save_json(REPORTS/'st5c_mcd_provenance.json',dict(repository='wengziheng/mcd_rppg',revision='91b967a5f2bab735d35f81722ff95e629358753b',
        confirmation_received_by_PI='2026-10-15',confirmation='Data provider confirmed this is the new official repository; GitHub README predates the move.',
        evidence='reports/st8/instruction_approval.json',old_repository_compared=False,HuggingFace_accessed=False,
        received_original_root=str(root),license='CC-BY-4.0',citation='Egorov et al., ACM MM 2025',DOI='10.1145/3746027.3758255'))
    for done,row in enumerate(db.to_dict('records'),1):
        if f'{row["patient_id"]}_{row["step"]}' in completed:continue
        base=dict(subject_id=row['patient_id'],state=row['step'],recording_id=f'{row["patient_id"]}_{row["step"]}',
            population='MCD-rPPG pinned front FullHDwebcam1200recordings',ground_truth_path=str(root/row['ppg_sync']),frame_timestamp_path=str(root/row['meta']),raw_PPG_path=str(root/row['ppg']))
        for name in ['video','meta','ppg_sync','ppg']:
            p=root/row[name];check('Only exact immutable allowed MCD file',row[name] in manifest and p.stat().st_size==int(manifest[row[name]]['bytes']))
        index,times=timestamp_table(root/row['meta']);raw,rawtimes=timestamp_table(root/row['ppg']);sync=np.loadtxt(root/row['ppg_sync'],ndmin=2)
        elapsed=(times.astype(float)-float(times[0]))/1e6
        elapsed[times==np.iinfo(np.int64).min]=np.nan
        parsed=packet_metadata(root/row['video']);nf=parsed['positive_packet_count'];ns=len(sync);nm=len(index)
        category='F0' if nf==ns==nm else 'F1' if nf==ns and nf!=nm else 'F2' if nf==nm and nf!=ns else 'other'
        frames.append(dict(**base,video_path=str(root/row['video']),video_sha256=manifest[row['video']]['sha256'],**parsed,
            N_frames=nf,N_sync=ns,N_meta=nm,frame_count_class=category,frames_minus_sync=nf-ns,frames_minus_meta=nf-nm,
            meta_timestamp_invalid_rows=int(np.sum(times==np.iinfo(np.int64).min))))
        groups.append(dict(**base,**clock_groups(rawtimes,times[0])))
        columns.append(dict(**base,**second_column(sync,index,elapsed,raw)))
        if done%25==0 or done==1200:
            write_rows('outputs/st5c_mcd_framecount.csv',frames);write_rows('outputs/st5c_mcd_clock_groups.csv',groups)
            write_rows('outputs/st5c_mcd_sync_col2.csv',columns)
            progress('mcd_packets',done,1200,started);budget_check((time.monotonic()-started)/done*(1200-done)+7200)
    counts={name:sum(r['frame_count_class']==name for r in frames) for name in ['F0','F1','F2','other']}
    coarse=sum(r['group_interval_median_ge0_1'] for r in groups)
    summary=dict(status='passed' if coarse<=120 else 'failed_clock_group_gate',recordings=1200,frame_count_classes=counts,
        ambiguous_frame_count_records=sum(r['frame_count_ambiguous'] for r in frames),
        raw_group_median_interval_ge0_1_records=coarse,raw_group_median_interval_ge0_1_fraction=coarse/1200,
        raw_timestamp_unparseable_records=sum(r['raw_timestamp_invalid_rows']>0 for r in groups),
        pre_M3_clock_group_gate_passed=coarse<=120,video_pixels_decoded=False,
        clock_group_median_intervals=describe([r['group_interval_seconds_median'] for r in groups],'seconds'),
        clock_group_median_sizes=describe([r['group_samples_median'] for r in groups],'samples'))
    save_json(REPORTS/'st5c_mcd_M1_summary.json',summary)
    save_json(REPORTS/'st5c_mcd_sync_col2.json',dict(status='complete',recordings=len(columns),table='outputs/st5c_mcd_sync_col2.csv',used_for_mapping=False,
        hypotheses=dict(meta_index='maximize exact constant-difference fraction across all legal meta offsets; first k and smallest c break ties',
        raw_sample_number='zero/one-based integer index into raw PPG; exact amplitude equality, invalid indices count as nonmatches',
        time='numeric seconds relative to first meta timestamp, compared at k0; descriptive only')))
    check('Raw PPG clock groups permit M3',coarse<=120,scope='S5-3M2')
    return summary


if __name__=='__main__':finish('mcd_packets',main)
