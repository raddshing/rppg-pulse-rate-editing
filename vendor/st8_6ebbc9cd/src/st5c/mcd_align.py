"""Prespecified label-label offsets, timestamped windows and sufficiency counts."""
from src.st5c.mcd_packets import *
from src.st5c.signals import common_hr,reference_estimators,classify
from scipy.signal import butter,sosfiltfilt


def correlation(a,b):
    if len(a)<3 or not np.isfinite(a).all() or not np.isfinite(b).all():return np.nan
    x=a-a.mean();y=b-b.mean();den=np.sqrt(np.dot(x,x)*np.dot(y,y))
    return float(np.dot(x,y)/den) if den>0 else np.nan


def best_delta(x,y):
    values=[]
    for d in range(-20,21):
        a,b=(x[:-d],y[d:]) if d>0 else (x[-d:],y[:d]) if d<0 else (x,y)
        values.append(correlation(a,b))
    if not np.isfinite(values).any():return np.nan,np.nan
    j=int(np.nanargmax(values));return (j-20)/100,float(values[j])


def compare_offset(times,sync,raw_times,raw,k,sos):
    assigned=times[k:k+len(sync)];low=max(assigned[0],raw_times[0]);high=min(assigned[-1],raw_times[-1])
    if high<=low:return None
    grid=np.arange(int(np.ceil(low*100)),int(np.floor(high*100))+1,dtype=float)/100
    if len(grid)<100:return None
    check('100Hz interpolation stays within both labels',grid[0]>=low-1e-12 and grid[-1]<=high+1e-12)
    x=sosfiltfilt(sos,np.interp(grid,assigned,sync));y=sosfiltfilt(sos,np.interp(grid,raw_times,raw))
    delta,peak=best_delta(x,y)
    return dict(k=k,r0=correlation(x,y),delta_seconds=delta,peak_correlation=peak,grid=grid,x=x,y=y)


def offset_alignment(meta_times,sync,raw_times,raw):
    sos=butter(4,[.7,3.5],btype='bandpass',fs=100,output='sos');candidates=[];best=None
    for k in range(len(meta_times)-len(sync)+1):
        value=compare_offset(meta_times,sync,raw_times,raw,k,sos)
        if value is None:continue
        candidates.append({key:value[key] for key in ['k','r0','delta_seconds','peak_correlation']})
        if np.isfinite(value['r0']) and (best is None or value['r0']>best['r0']):best=value
    if best is None:return dict(alignment_status='no_overlap',reason='no_common_finite100Hz_support'),candidates
    k=best['k'];grid=best['grid'];edges=np.linspace(grid[0],grid[-1],4);thirds=[]
    for j in range(3):
        take=(grid>=edges[j])&((grid<edges[j+1]) if j<2 else (grid<=edges[j+1]))
        delta,_=best_delta(best['x'][take],best['y'][take]);thirds.append(delta)
    verified=best['r0']>=.8 and abs(best['delta_seconds'])<=.05+1e-12 and all(np.isfinite(d) and abs(d-best['delta_seconds'])<=.05+1e-12 for d in thirds)
    by={r['k']:r['r0'] for r in candidates};outside=[r['r0'] for r in candidates if abs(r['k']-k)>2 and np.isfinite(r['r0'])]
    assigned=meta_times[k:k+len(sync)];inside=(assigned>=raw_times[0])&(assigned<=raw_times[-1])
    position=np.rint((assigned[inside]-raw_times[0])/(raw_times[1]-raw_times[0])).astype(int)
    position=np.clip(position,0,len(raw)-1)
    integer_scale=bool(np.all(raw==np.rint(raw)) and np.all(sync==np.rint(sync)) and raw.min()>=0 and raw.max()<=255 and sync.min()>=0 and sync.max()<=255)
    exact=float(np.mean(sync[inside]==raw[position])) if integer_scale and inside.any() else np.nan
    return dict(alignment_status='verified' if verified else 'unverified',reason='' if verified else 'prespecified_correlation_or_lag_checks',
        k_star=k,r0=best['r0'],delta_seconds=best['delta_seconds'],peak_correlation=best['peak_correlation'],
        r0_k_minus1=by.get(k-1,np.nan),r0_k_plus1=by.get(k+1,np.nan),
        outside_pm2_max_r0=max(outside) if outside else np.nan,outside_pm2_r0_gap=best['r0']-max(outside) if outside else np.nan,
        third1_delta_seconds=thirds[0],third2_delta_seconds=thirds[1],third3_delta_seconds=thirds[2],
        integer_encoding_0_255_comparable=integer_scale,exact_nearest_raw_amplitude_fraction=exact,exact_comparison_rows=int(inside.sum()),
        exact_comparison_outside_support_rows=int((~inside).sum()),common_start_seconds=float(grid[0]),common_end_seconds=float(grid[-1]),common_grid_samples=len(grid)),candidates


def aligned_windows(base,mapped_times,sync,raw_times,raw):
    result=[];edge_excluded=0;start=mapped_times[0];end=mapped_times[-1]
    for j in range(int(np.floor((end-start+.1)/10))):
        begin=start+j*10;stop=begin+10;grid=begin+np.arange(300)/30
        left=np.min(abs(mapped_times-begin));right=np.min(abs(mapped_times-stop))
        if left>.1+1e-12 or right>.1+1e-12 or grid[0]<mapped_times[0]-1e-12 or grid[-1]>mapped_times[-1]+1e-12:
            edge_excluded+=1;continue
        interval=(mapped_times[:-1]<stop)&(mapped_times[1:]>begin)
        gaps=np.diff(mapped_times)[interval];maximum=float(gaps.max()) if len(gaps) else np.nan
        gap=bool(len(gaps) and np.any(gaps>.1+1e-12));wave=np.interp(grid,mapped_times,sync)
        raw_wave=np.interp(grid,raw_times,raw) if raw_times[0]<=grid[0] and raw_times[-1]>=grid[-1] else None
        for layer in BANDS:
            hr=common_hr(wave,layer);more=reference_estimators(wave,hr);kind=classify(hr,more['h_ACF'],more['h_IBI'])
            rawhr=common_hr(raw_wave,layer) if raw_wave is not None else np.nan
            result.append(dict(**base,window_id=base['recording_id']+'|'+str(j),window_index=j,layer=layer,
                start_seconds=j*10,end_seconds=(j+1)*10,meta_relative_start_seconds=float(begin),first_mapped_frame_seconds=float(start),
                grid_samples=300,left_endpoint_nearest_frame_seconds=float(left),right_endpoint_nearest_frame_seconds=float(right),
                maximum_intersecting_frame_gap_seconds=maximum,gap_excluded=gap,primary_included=not gap,
                h_FFT=hr,h_ACF=more['h_ACF'],h_IBI=more['h_IBI'],reference_class=kind,agreement=kind=='agreement',
                raw_PPG_HR=rawhr,raw_PPG_difference=rawhr-hr,raw_PPG_gt1grid=bool(abs(rawhr-hr)>GRID+1e-6) if np.isfinite(rawhr) else False,
                raw_PPG_available=bool(np.isfinite(rawhr)),position_bin='0-30' if j*10<30 else '30-60' if j*10<60 else '60-120' if j*10<120 else 'ge120'))
    return result,edge_excluded


def summaries(inventory,windows,alignments):
    frame=pd.DataFrame(windows);bins=[-np.inf,70,85,100,110,120,130,140,np.inf];labels=['lt70','70-85','85-100','100-110','110-120','120-130','130-140','ge140'];rows_out=[]
    def counts(q):
        return dict(n_windows=len(q),n_subjects=int(q.subject_id.nunique()) if len(q) else 0,
            n_agreement=int(q.agreement.sum()) if len(q) else 0,n_agreement_subjects=int(q[q.agreement].subject_id.nunique()) if len(q) else 0)
    suff={}
    for population in ['primary_gap_excluded','descriptive_gap_included']:
        subset=frame[frame.primary_included] if population=='primary_gap_excluded' and len(frame) else frame
        for layer in BANDS:
            for state in ['before','after']:
                q=subset[subset.layer.eq(layer)&subset.state.eq(state)] if len(subset) else subset
                for low,high,label in zip(bins[:-1],bins[1:],labels):
                    z=q[q.h_FFT.ge(low)&q.h_FFT.lt(high)] if len(q) else q
                    rows_out.append(dict(population=population,kind='HR_bin',layer=layer,state=state,bin=label,**counts(z)))
                for pos in ['0-30','30-60','60-120','ge120']:
                    z=q[q.h_FFT.ge(100)&q.position_bin.eq(pos)] if len(q) else q
                    rows_out.append(dict(population=population,kind='ge100_position',layer=layer,state=state,bin=pos,**counts(z)))
        high=subset[subset.layer.eq('HAR')&subset.h_FFT.ge(100)&subset.agreement] if len(subset) else subset
        n=len(high);subjects=int(high.subject_id.nunique()) if n else 0
        suff[population]=dict(n_HAR_ge100_agreement_windows=n,n_subjects=subjects,windows_ge50=n>=50,subjects_ge20=subjects>=20,both=n>=50 and subjects>=20)
    write_rows('outputs/st5c_mcd_hr_bins2.csv',rows_out)
    save_json(REPORTS/'st5c_mcd_sufficiency2.json',dict(status='complete',primary=suff['primary_gap_excluded'],descriptive_gap_included=suff['descriptive_gap_included'],scientific_selection_judgment=False))
    save_json(REPORTS/'st5c_mcd_mapping2.json',dict(status='complete',recordings=len(inventory),mapping_confirmed_records=sum(r['mapping_confirmed'] for r in inventory),
        alignment_status_counts=pd.Series([r['alignment_status'] for r in alignments]).value_counts().to_dict(),
        unresolved_reason_counts=pd.Series([r['mapping_reason'] for r in inventory if not r['mapping_confirmed']]).value_counts().to_dict(),
        complete_windows_per_layer=len(windows)//2,gap_excluded_windows_per_layer=int(frame[frame.layer.eq('HAR')].gap_excluded.sum()) if len(frame) else 0,
        no_external_pixels_decoded=True,no_HF_access=True,no_video_POS_or_model_used_for_alignment=True))


def main():
    initial=json.loads((REPORTS/'st5c_mcd_M1_summary.json').read_text());check('M1 completed and preM3 gate passed',initial['status']=='passed' and initial['pre_M3_clock_group_gate_passed'])
    root,manifest,db=load_sources();packet=pd.read_csv(ROOT/'outputs/st5c_mcd_framecount.csv',dtype={'subject_id':str},float_precision='round_trip').set_index('recording_id')
    groups=pd.read_csv(ROOT/'outputs/st5c_mcd_clock_groups.csv',dtype={'subject_id':str},float_precision='round_trip').set_index('recording_id')
    check('No completed MCD alignment overwrite',not (ROOT/'outputs/st5c_mcd_alignment.csv').exists())
    alignments=[];inventory=[];windows=[];offsets=[];started=time.monotonic()
    for done,row in enumerate(db.to_dict('records'),1):
        rid=f'{row["patient_id"]}_{row["step"]}';p=packet.loc[rid];g=groups.loc[rid]
        base=dict(subject_id=row['patient_id'],state=row['step'],recording_id=rid,population='MCD-rPPG received_front_FullHDwebcam_timestamp_verified',
            ground_truth_path=str(root/row['ppg_sync']),raw_PPG_path=str(root/row['ppg']),frame_timestamp_path=str(root/row['meta']))
        index,timestamps=timestamp_table(root/row['meta']);raw,rawstamps=timestamp_table(root/row['ppg']);sync=np.loadtxt(root/row['ppg_sync'],ndmin=2)[:,0]
        meta=(timestamps.astype(float)-float(timestamps[0]))/1e6
        rawtimes=float(g.clock_a_seconds)+float(g.clock_b_seconds_per_sample)*np.arange(len(raw))
        if bool(g.raw_clock_irregular) or not np.isfinite(raw).all():
            result=dict(alignment_status='raw_clock_irregular',reason='malformed_original_timestamp' if g.raw_timestamp_invalid_rows else 'raw_clock_residual_or_nonfinite_signal')
        elif int(p.meta_timestamp_invalid_rows)>0 or not np.all(np.diff(meta)>0) or len(meta)<len(sync) or not np.isfinite(sync).all():
            result=dict(alignment_status='unverified',reason='meta_or_sync_structure')
        else:
            result,candidates=offset_alignment(meta,sync,rawtimes,raw)
            offsets.extend(dict(recording_id=rid,**r) for r in candidates)
        alignment=dict(**base,raw_start_minus_meta_start_seconds=float(g.raw_start_minus_meta_start_seconds),raw_clock_a_seconds=float(g.clock_a_seconds),
            raw_clock_b_seconds_per_sample=float(g.clock_b_seconds_per_sample),raw_clock_max_residual_seconds=float(g.clock_max_absolute_residual_seconds),**result)
        alignments.append(alignment)
        confirmed=result['alignment_status']=='verified' and p.frame_count_class in ['F0','F1','F2'] and not bool(p.frame_count_ambiguous)
        reason='' if confirmed else 'packet_count_ambiguity' if bool(p.frame_count_ambiguous) else 'frame_count_other' if p.frame_count_class=='other' else result['alignment_status']
        inv=dict(**base,N_frames=int(p.N_frames),N_sync=int(p.N_sync),N_meta=int(p.N_meta),frame_count_class=p.frame_count_class,
            alignment_status=result['alignment_status'],mapping_confirmed=confirmed,mapping_reason=reason,db_pulse=float(row['pulse']),
            n_complete_windows=0,n_primary_windows=0,n_gap_excluded_windows=0,n_endpoint_excluded_windows=0)
        if confirmed:
            k=int(result['k_star']);mapped=meta[k:k+len(sync)];local,edge=aligned_windows(base,mapped,sync,rawtimes,raw);windows.extend(local)
            inv.update(k_star=k,first_labelled_video_frame=0 if p.frame_count_class in ['F0','F1'] else k,
                first_labelled_meta_row=k,n_complete_windows=len(local)//2,n_primary_windows=sum(r['primary_included'] and r['layer']=='HAR' for r in local),
                n_gap_excluded_windows=sum(r['gap_excluded'] and r['layer']=='HAR' for r in local),n_endpoint_excluded_windows=edge)
            for layer in BANDS:
                z=[r for r in local if r['layer']==layer];primary=[r for r in z if r['primary_included']]
                mean=float(np.mean([r['h_FFT'] for r in z])) if z else np.nan;mean_primary=float(np.mean([r['h_FFT'] for r in primary])) if primary else np.nan
                inv.update({'mean_complete_HR_'+layer:mean,'mean_primary_HR_'+layer:mean_primary,'db_pulse_minus_mean_complete_HR_'+layer:float(row['pulse'])-mean,
                    'raw_PPG_gt1grid_all_'+layer:sum(r['raw_PPG_gt1grid'] for r in z),'raw_PPG_gt1grid_primary_'+layer:sum(r['raw_PPG_gt1grid'] for r in primary)})
        inventory.append(inv)
        if done%25==0 or done==1200:
            pd.DataFrame(alignments).to_csv(ROOT/'outputs/st5c_mcd_alignment.csv',index=False)
            pd.DataFrame(inventory).to_csv(ROOT/'outputs/st5c_mcd_inventory2.csv',index=False)
            pd.DataFrame(windows).to_csv(ROOT/'outputs/st5c_mcd_windows2.csv',index=False)
            pd.DataFrame(offsets).to_csv(ROOT/'outputs/st5c_mcd_alignment_offsets.csv',index=False)
            progress('mcd_alignment',done,1200,started);budget_check((time.monotonic()-started)/done*(1200-done)+3600)
    summaries(inventory,windows,alignments)
    return dict(recordings=len(inventory),window_rows=len(windows),mapping_confirmed=sum(r['mapping_confirmed'] for r in inventory),video_pixels_decoded=False)


if __name__=='__main__':finish('mcd_alignment',main)
