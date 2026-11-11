"""Prepare actual author-config RhythmFormer originals after the fixed opening."""
import os
os.environ['ST12_NAMESPACE']='st12c'
from src.st12.compat import bind
bind()
from src.st9e.native import *
from src.st9g.preprocess import standardized_data,standardized_label
from src.st12c.rhythm_view import RhythmViewWorker
from src.st12 import gpu


def main():
    path=REPORTS/'st12c_a_settings.json';settings=json.loads(path.read_text())
    check('Rhythm settings committed',sha(path) in (ROOT/'reports/ST4_INSTR05B_PREFIX.md').read_text())
    for name,digest in settings['source_hashes'].items():check('Exact Rhythm dependencies',sha(ROOT/name)==digest)
    check('Fixed RF source plan',sha(ROOT/settings['chunks_path'])==settings['chunks_sha256'])
    plan=rows(settings['chunks_path']);_,pure,allowed=indices();audit={r['path']:r for r in rows('outputs/st12_input_hash_checks.csv')}
    worker=RhythmViewWorker();manifest=[];started=time.monotonic()
    try:
        for done,rid in enumerate(settings['model']['training_recordings'],1):
            budget_check();gpu.capacity();files=frame_files(pure[rid]);label=contact(pure[rid],'PURE',allowed)
            for p in files:
                st=p.stat();r=audit[path_key(p)]
                check('Original source unchanged since S12 input hash',all(int(r[k])==v for k,v in dict(bytes=st.st_size,device=st.st_dev,inode=st.st_ino,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns).items()))
            native=read_pure(files,0,len(files));view,_=worker.view(native);del native
            data=standardized_data(view);labels=standardized_label(label);del view
            directory=Path(PATHS['bulk'])/'st12c/a/original'/rid;directory.mkdir(parents=True,exist_ok=True)
            for r in [x for x in plan if x['recording_id']==rid]:
                start=int(r['start_frame']);x=directory/('chunk_'+r['chunk_index']+'_input.npy');y=directory/('chunk_'+r['chunk_index']+'_label.npy')
                check('Never overwrite RF originals',not x.exists() and not y.exists())
                np.save(x,np.ascontiguousarray(data[start:start+160],dtype=np.float32));np.save(y,np.ascontiguousarray(labels[start:start+160],dtype=np.float32))
                manifest.append(dict(r,input_path=str(x),label_path=str(y),input_bytes=x.stat().st_size,label_bytes=y.stat().st_size,
                    input_sha256=sha(x),label_sha256=sha(y),stored_format='THWC_float32',population='authorPURE0–0.8; whole-recording Standardized'))
            del data,labels,label
            write_rows('outputs/st12c_a_original_manifest.csv',manifest);progress('rhythm_originals',done,len(settings['model']['training_recordings']),started)
    finally:worker.close()
    check('All planned RF originals',len(manifest)==settings['model']['chunks'])
    return dict(chunks=len(manifest),bytes=sum(r['input_bytes']+r['label_bytes'] for r in manifest),
        manifest_sha256=sha(ROOT/'outputs/st12c_a_original_manifest.csv'),GPU_seconds=0,validation_calls=0)


if __name__=='__main__':finish('rhythm_originals',main)
