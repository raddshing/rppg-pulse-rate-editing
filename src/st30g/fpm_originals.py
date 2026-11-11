"""FP-m raw72 fixed-HC training originals; exact RF596 boundaries."""
from src.st30p.ops import *
configure('st30g')
import numpy as np
import cv2
import contextlib
import io


def main():
    assert cv2.__version__=='4.5.2'
    plan=sorted(rows('outputs/st12c_a_original_manifest.csv'),key=lambda r:int(r['chunk_order']))
    core=json.loads((ROOT/'reports/st30r/core_facts.json').read_text())
    assert sha('outputs/st12c_a_original_manifest.csv')==core['source_hashes']['outputs/st12c_a_original_manifest.csv']
    assert len(plan)==596
    projected=596*(160*72*72*3*4+128+160*4+128)
    capacity(add_bulk=projected+3*1024**2)
    index={r['recording_id']:r for r in rows('data/index/pure_index_st1.csv')}
    guard={r['path']:r for r in rows('outputs/st12_input_hash_checks.csv')}
    repo=ROOT/'third_party/rPPG-Toolbox';sys.path.insert(0,str(repo));os.chdir(repo)
    from dataset.data_loader.BaseLoader import BaseLoader
    loader=object.__new__(BaseLoader)
    from src.st9g.preprocess import standardized_label
    manifest=[];checks=[];detections=[];started=time.monotonic()
    for done,rid in enumerate(sorted({r['recording_id'] for r in plan}),1):
        rr=index[rid];files=sorted(Path(rr['path']).glob('*.png'));group=[r for r in plan if r['recording_id']==rid]
        assert len(files)==int(rr['frame_count'])
        def intact(p):
            p=Path(p);s=p.stat();g=guard[str(p)]
            assert all(int(g[k])==v for k,v in dict(bytes=s.st_size,device=s.st_dev,inode=s.st_ino,
                mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns).items()),str(p)
        for p in files:intact(p)
        intact(rr['label_path'])
        label=json.loads(Path(rr['label_path']).read_text());raw=np.array([v['Value']['waveform'] for v in label['/FullPackage']],float)
        y=standardized_label(np.interp(np.linspace(1,len(raw),len(files)),np.linspace(1,len(raw),len(raw)),raw))
        bgr=cv2.imread(str(files[0]));assert bgr is not None;first=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        msg=io.StringIO()
        with contextlib.redirect_stdout(msg):box=np.asarray(loader.face_detection(first,'HC',True,1.5),dtype=int)
        detections.append(dict(recording_id=rid,first_frame_HC=True,coefficient=1.5,
            fallback='No Face Detected' in msg.getvalue(),multiple='More than one' in msg.getvalue(),coordinates_exported=False))
        x0,y0,w,h=box
        directory=Path(PATHS['bulk'])/'st30g/FPm/original'/rid;directory.mkdir(parents=True,exist_ok=False)
        buffer=[];lookup={int(r['start_frame']):r for r in group}
        for i,p in enumerate(files[:len(group)*160]):
            frame=first if i==0 else cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB)
            crop=frame[max(y0,0):min(y0+h,frame.shape[0]),max(x0,0):min(x0+w,frame.shape[1])]
            buffer.append(cv2.resize(crop,(72,72),interpolation=cv2.INTER_AREA))
            if len(buffer)<160:continue
            start=i-159;r=lookup[start]
            capacity(add_bulk=160*72*72*3*4+2048)
            xp=directory/('chunk_'+r['chunk_index']+'_input.npy');yp=directory/('chunk_'+r['chunk_index']+'_label.npy')
            np.save(xp,np.ascontiguousarray(buffer,dtype=np.float32));np.save(yp,np.ascontiguousarray(y[start:start+160],dtype=np.float32))
            item=dict(r,model='FPm',view_index='FPm',input_path=str(xp),label_path=str(yp),
                input_bytes=xp.stat().st_size,label_bytes=yp.stat().st_size,input_sha256=sha(xp),label_sha256=sha(yp),
                stored_format='THWC_float32_Raw',population='PURE01-08_RF596_boundaries;fixed_recording_first_HC1.5;72INTER_AREA;whole_recording_Standardized_label')
            manifest.append(item)
            for p2,k in [(xp,'input'),(yp,'label')]:
                s=p2.stat();checks.append(dict(path=str(p2),sha256=item[k+'_sha256'],bytes=s.st_size,device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns))
            buffer=[]
        assert not buffer
        for p in files:intact(p)
        intact(rr['label_path'])
        write_rows('outputs/st30g_FPm_original_manifest.csv',manifest)
        write_rows('reports/st30g/FPm_original_guards.csv',checks)
        write_rows('outputs/st30g_FPm_original_detection.csv',detections)
        save('reports/st30g/FPm_original_progress.json',dict(status='running',recordings=done,total=47,chunks=len(manifest),elapsed_seconds=time.monotonic()-started,updated_utc=now()))
    assert len(manifest)==596 and sum(int(r['input_bytes'])+int(r['label_bytes']) for r in manifest)==projected
    save('reports/st30g/FPm_original_source_bindings.json',dict(source_hashes={p:sha(p) for p in [
        'outputs/st12c_a_original_manifest.csv','data/index/pure_index_st1.csv','outputs/st12_input_hash_checks.csv',
        'third_party/rPPG-Toolbox/dataset/data_loader/BaseLoader.py','src/st9g/preprocess.py','src/st30g/fpm_originals.py']},
        source_metadata_before_after_matches=True,new_pixel_export=False,new_geometry_export=False))
    return dict(status='complete',chunks=len(manifest),recordings=47,bytes=projected,
        manifest_sha256=sha('outputs/st30g_FPm_original_manifest.csv'),training_calls=0)


if __name__=='__main__':phase('st30g','FPm_originals',main,heavy=True)
