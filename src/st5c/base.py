"""Scoped paths, frozen-source checks and owned phase accounting for instruction02B."""
import csv,datetime as dt,fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PATHS=json.loads((ROOT/'paths.yaml').read_text())
NAMESPACE='st5c'
CACHE=Path(PATHS['cache'])/NAMESPACE
TMP=Path(PATHS['cache'])/'tmp'/NAMESPACE
REPORTS=ROOT/'reports'/NAMESPACE
for directory in (CACHE,TMP,REPORTS):directory.mkdir(parents=True,exist_ok=True)
os.environ.update(TMPDIR=str(TMP),MPLCONFIGDIR=str(CACHE/'matplotlib'),XDG_CACHE_HOME=str(CACHE/'xdg'),
    CUDA_CACHE_PATH=str(CACHE/'cuda'),TORCH_EXTENSIONS_DIR=str(CACHE/'torch_extensions'),
    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
GRID=3.515625
BANDS={'HAR':(.6,3.3),'L0':(.75,2.5)}
MODELS=('deepphys','efficientphys','factorizephys','physformer','physnet','rhythmformer','tscan')
RW15=sorted([f'st1_mmpd_m03_rw_{m}_{a}_all_s100' for m in MODELS for a in ('pm','um')]+['st1_mmpd_m03_rw_ibvpnet_pm_all_s100'])
READ_HASHES={}
_MANIFEST=None
class RequiredStop(RuntimeError):
    def __init__(self,message,scope=None):self.scope=scope or NAMESPACE;self.check=message;super().__init__(message)
def check(name,condition,expected=None,scope=None):
    if not bool(condition if expected is None else condition==expected):raise RequiredStop(name,scope)
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def rows(path):
    with (ROOT/path).open(newline='') as f:return list(csv.DictReader(f))
def write_rows(path,records,fields=None):
    records=list(records);p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(records[0]));w.writeheader();w.writerows(records)
def append_row(path,record):
    with (ROOT/path).open('a',newline='') as f:
        fcntl.flock(f,fcntl.LOCK_EX);w=csv.DictWriter(f,fieldnames=list(record))
        if f.tell()==0:w.writeheader()
        w.writerow(record);f.flush()
def save_json(path,value):
    p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    with p.open('w') as f:f.write(raw);f.flush()
def source_hash(path):
    global _MANIFEST
    p=ROOT/path
    try:key=str(p.relative_to(ROOT))
    except ValueError:key=str(p)
    if _MANIFEST is None:
        _MANIFEST={}
        for name in ['outputs/st8_input_hash_checks.csv','outputs/st8_st6_manifest.csv']:
            for r in rows(name):_MANIFEST[r['path']]=r['sha256']
        old='outputs/st4_st3exit_manifest.csv'
        check('Frozen inherited manifest unchanged',sha(ROOT/old)==_MANIFEST[old])
        for r in rows(old):_MANIFEST.setdefault(r['path'],r['sha256'])
        inherited='reports/st4/st4_a_pilot_input_hashes.json'
        check('Frozen pilot dependency index unchanged',sha(ROOT/inherited)==_MANIFEST[inherited])
        for inherited_key,value in json.loads((ROOT/inherited).read_text()).items():_MANIFEST.setdefault(inherited_key,value)
    check('Input has frozen checksum',key in _MANIFEST)
    digest=sha(p);check('Frozen source unchanged',digest==_MANIFEST[key]);READ_HASHES[key]=digest;return digest
def prefix_guard():
    check('S8-0 completed',json.loads((ROOT/'reports/st8/st8_s0_completion.json').read_text())['status']=='complete')
    name='reports/ST4_INSTR02B_PREFIX.md'
    a=subprocess.run(['git','ls-files','--error-unmatch',name],cwd=ROOT,capture_output=True)
    b=subprocess.run(['git','diff','--quiet','HEAD','--',name],cwd=ROOT)
    check('Dependent prefix committed',a.returncode==0 and b.returncode==0)
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def cpu_phase_seconds():
    total=0.
    for scope in ['st8','st7b','st5c','st6d']:
        for p in (ROOT/'reports'/scope).glob('*_phase.json'):
            try:r=json.loads(p.read_text())
            except (json.JSONDecodeError,FileNotFoundError):continue
            if r.get('resource')!='CPU':continue
            total+=float(r.get('wall_seconds',max(0,time.time()-r['started_epoch'])))
    return total
def budget_check(projected_additional_seconds=0):
    used=cpu_phase_seconds();check('Summed CPU phase wall below48h',used+projected_additional_seconds<172800)
    check('Working cache below150GiB',not (ROOT/'reports/st8/st8_disk_limit.json').exists())
    return used
def progress(stage,done,total,started,status='running'):
    elapsed=time.monotonic()-started;remain=elapsed/done*(total-done) if done else None
    r=dict(stage=stage,status=status,done=int(done),total=int(total),elapsed_seconds=round(elapsed,3),remaining_seconds=None if remain is None else round(remain,3),updated_utc=now())
    save_json(REPORTS/f'{NAMESPACE}_{stage}_progress.json',r);print(json.dumps(r),flush=True)
def process_identity(pid):
    p=Path('/proc')/str(pid);fields=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=int(pid),process_start_ticks=fields[19],executable=str((p/'exe').resolve()),command=' '.join(x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x))
def ledger(job,event,pid=None):append_row(f'outputs/{NAMESPACE}_process_ledger.csv',dict(job=job,**process_identity(pid or os.getpid()),recorded_utc=now(),event=event))
def delete_temp(path):
    p=Path(path);check('Delete only consumed new scoped scratch',any(d.resolve() in p.resolve().parents for d in (CACHE,TMP)))
    append_row(f'outputs/{NAMESPACE}_deleted_temp.csv',dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)));p.unlink()
def finish(stage,function,resource='CPU'):
    started=time.monotonic();epoch=time.time();ledger(stage,'started');code=0
    phase=REPORTS/f'{NAMESPACE}_{stage}_phase.json'
    save_json(phase,dict(status='running',resource=resource,started_epoch=epoch,started_utc=now()))
    try:
        budget_check();commit=prefix_guard();result=dict(function() or {})
        if 'status' in result:result['check_status']=result.pop('status')
        result.update(status='complete',execution_commit=commit)
    except RequiredStop as e:
        code=2;result=dict(status='stopped_required_check',check=str(e),scope=e.scope)
        save_json(REPORTS/f'{NAMESPACE}_{stage}_required_stop.json',dict(at_utc=now(),**result))
    except Exception as e:
        import traceback
        code=1;result=dict(status='implementation_error',error_type=type(e).__name__,stack=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name) for f in traceback.extract_tb(e.__traceback__)])
    wall=time.monotonic()-started;result.update(process_wall_seconds=wall,finished_utc=now())
    save_json(REPORTS/f'{NAMESPACE}_{stage}_completion.json',result);save_json(REPORTS/f'{NAMESPACE}_{stage}_input_hashes.json',READ_HASHES)
    save_json(REPORTS/f'{NAMESPACE}_{stage}_exit.json',dict(exit_code=code,finished_utc=now()))
    save_json(phase,dict(status=result['status'],resource=resource,started_epoch=epoch,wall_seconds=wall,finished_utc=now()))
    ledger(stage,'completed' if code==0 else 'stopped');print(json.dumps(dict(stage=stage,status=result['status'])),flush=True)
    if code:raise SystemExit(code)
