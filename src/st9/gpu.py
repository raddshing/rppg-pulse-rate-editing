"""Owned GPU child lifetime accounts and identity-checked heavy-CPU pauses."""
from src.st9.ops import *
from src.st9.checkpoint_guard import require_checkpoint_capacity
import signal
from contextlib import contextmanager

CAPS = {'S7':14400., 'A2':18000., 'A2-ref':9000., 'B-T+L-dev':216000.}


DIRECTORY = ROOT / 'reports/st9/gpu_accounts'


def record_seconds(record, monotonic_now=None, epoch_now=None):
    value=float(record['GPU_seconds'])
    if record['status'] in ['starting','running']:
        if 'started_monotonic' in record:
            elapsed=(time.monotonic() if monotonic_now is None else monotonic_now)-record['started_monotonic']
        else:
            began=dt.datetime.fromisoformat(record['started_utc']).timestamp()
            elapsed=(time.time() if epoch_now is None else epoch_now)-began
        value=max(value,elapsed)
    return value


def accounts():
    out=[]
    for path in DIRECTORY.glob('*.json'):
        # Older already-running launchers wrote in place. New records are atomic.
        for attempt in range(5):
            try:
                out.append(json.loads(path.read_text()));break
            except json.JSONDecodeError:
                if attempt==4:raise
                time.sleep(.02)
    return out


def account_total(account):
    return sum(record_seconds(r) for r in accounts() if r['account']==account)


def write_account(path,value):
    temporary=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


@contextmanager
def allocation_lock():
    DIRECTORY.mkdir(exist_ok=True)
    with (DIRECTORY/'.allocation.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


def owned_cpu_roots():
    result = []
    for ns in ['st9e','st9s','st9a','st9g','st9l','st9v']:
        ledger_path = ROOT / f'outputs/{ns}_process_ledger.csv'
        if not ledger_path.exists():
            continue
        events = rows(ledger_path)
        for phase in (ROOT / 'reports' / ns).glob('*_phase.json'):
            r = json.loads(phase.read_text())
            if r.get('status') != 'running' or r.get('resource') != 'CPU':
                continue
            job = phase.name[len(ns)+1:-len('_phase.json')]
            starts = [x for x in events if x['job'] == job and x['event'] == 'started']
            if not starts:
                continue
            expected = starts[-1]
            pid = int(expected['pid'])
            if pid == os.getpid() or not (Path('/proc') / str(pid)).exists():
                continue
            actual = process_identity(pid)
            check('Pause only exact owned CPU root identity', all(str(actual[k]) == str(expected[k]) for k in actual))
            check('Pause only instruction03 command', 'src.st9' in actual['command'] or '/src/st9' in actual['command'])
            result.append(actual)
    return result


def descendants(pid):
    parents = {}
    for path in Path('/proc').glob('[0-9]*'):
        try:
            fields = (path / 'stat').read_text().rsplit(')',1)[1].split()
            parents[int(path.name)] = int(fields[1])
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    found = []
    pending = [pid]
    while pending:
        parent = pending.pop()
        children = [p for p, pp in parents.items() if pp == parent]
        found.extend(children)
        pending.extend(children)
    return found


def pause_cpu(job):
    entries = []
    for root in owned_cpu_roots():
        # Stop the root first so it cannot create further workers while enumerating.
        os.kill(root['pid'], signal.SIGSTOP)
        entries.append(dict(**root, event='paused', at_utc=now()))
        for pid in descendants(root['pid']):
            try:
                identity = process_identity(pid)
                os.kill(pid, signal.SIGSTOP)
                entries.append(dict(**identity, event='paused', at_utc=now()))
            except (FileNotFoundError, ProcessLookupError):
                pass
    continuing=[]
    for value in accounts():
        if value.get('status') in ['starting','running'] and value.get('physical_GPU')==0:
            check('Only authorized A2 GPU0 may continue during S7 timing',value['account'] in ['A2','A2-ref'])
            continuing.append(dict(job=value['job'],account=value['account'],identity=value['identity']))
    save_json(ROOT / 'reports/st9' / (job + '_CPU_pause.json'), dict(job=job, processes=entries,
        authorized_GPU0_generation_continues=continuing,loopback_viewers_untouched=True))
    return entries


def resume_cpu(job, entries):
    out = []
    for r in reversed(entries):
        try:
            actual = process_identity(r['pid'])
            check('Resume only exact owned paused identity', all(str(actual[k]) == str(r[k]) for k in actual))
            os.kill(r['pid'], signal.SIGCONT)
            out.append(dict(**actual, event='resumed', at_utc=now()))
        except (FileNotFoundError, ProcessLookupError):
            out.append(dict(pid=r['pid'], event='already_exited', at_utc=now()))
    save_json(ROOT / 'reports/st9' / (job + '_CPU_resume.json'), dict(job=job, processes=out))


def run_gpu(job, account, gpu, environment, module, args=(), limit=None, pause_heavy=False, reserve_provider=None):
    prefix_guard()
    require_checkpoint_capacity()
    destination = DIRECTORY / (job + '.json')
    child = None
    identity = None
    used = 0.
    capped = False
    aggregate_cap = False
    priority_reserve = 0.
    priority_cap = False
    disk_stop = None
    paused = []
    log_path = REPORTS / (job + '_worker.log')
    with allocation_lock():
        prior = account_total(account)
        priority_reserve = float(reserve_provider()) if reserve_provider is not None else 0.
        check('Nonnegative finite priority GPU reserve', priority_reserve >= 0 and priority_reserve < float('inf'))
        allowance = min(CAPS[account] - prior - priority_reserve, limit if limit is not None else CAPS[account])
        check('Positive GPU allowance', allowance > 20)
        check('Never overwrite a GPU attempt account', not destination.exists())
        for active in accounts():
            check('One owned child per physical GPU',not(active.get('status') in ['starting','running'] and active.get('physical_GPU')==gpu))
        began=time.monotonic();epoch=time.time()
        fixed=dict(job=job,account=account,physical_GPU=gpu,prior_GPU_seconds=prior,allowance_seconds=allowance,
            cap_seconds=CAPS[account],started_monotonic=began,
            initial_priority_reserve_seconds=priority_reserve,
            started_utc=dt.datetime.fromtimestamp(epoch,dt.timezone.utc).isoformat())
        write_account(destination,dict(**fixed,status='starting',GPU_seconds=0.,identity=None))
    try:
        paused = pause_cpu(job) if pause_heavy else []
        environment_vars = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
            ST9_GPU_ACCOUNT=account, ST9_GPU_ALLOWANCE=str(allowance), ST9_GPU_STARTED_EPOCH=str(epoch),
            ST9_PHYSICAL_GPU=str(gpu))
        with log_path.open('ab') as log:
            child = subprocess.Popen([str(Path(PATHS['environments'][environment]) / 'bin/python'), '-B', '-s', '-m', module, *args],
                cwd=ROOT, env=environment_vars, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            identity = process_identity(child.pid)
            ledger(job, 'started', child.pid)
            while True:
                used = time.monotonic() - began
                write_account(destination, dict(**fixed,status='running',GPU_seconds=used,
                    identity=identity, includes_loading_CPU_waits_failures=True))
                code = child.poll()
                if code is not None:
                    break
                try:
                    require_checkpoint_capacity()
                except RequiredStop as error:
                    disk_stop = str(error)
                current=[r for r in accounts() if r['account']==account]
                active=sum(r['status'] in ['starting','running'] for r in current)
                remaining=CAPS[account]-sum(record_seconds(r) for r in current)
                priority_reserve = float(reserve_provider()) if reserve_provider is not None else 0.
                check('Nonnegative finite priority GPU reserve', priority_reserve >= 0 and priority_reserve < float('inf'))
                # Sum both GPU lifetimes while running, including loading/waits.
                # Reserve bounded polling/terminate/kill latency for every child.
                aggregate_cap=remaining<=25*active+5
                priority_cap=reserve_provider is not None and remaining-priority_reserve<=25*active+5
                if used >= allowance - 5 or aggregate_cap or priority_cap or disk_stop:
                    check('Only exact owned GPU child stopped at resource bound', process_identity(child.pid) == identity)
                    capped = used >= allowance - 5 or aggregate_cap or priority_cap
                    child.terminate()
                    try:
                        child.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=10)
                    break
                try:
                    child.wait(timeout=min(10, max(.1,allowance-used-5)))
                except subprocess.TimeoutExpired:
                    pass
    finally:
        if child is not None and child.poll() is None:
            check('Only exact owned GPU child stopped on launcher exit', process_identity(child.pid) == identity)
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
        used = time.monotonic() - began
        other=sum(record_seconds(r) for r in accounts() if r['account']==account and r['job']!=job)
        write_account(destination, dict(status='closed', job=job, account=account, physical_GPU=gpu,
            started_utc=fixed['started_utc'],started_monotonic=began,
            GPU_seconds=used, cumulative_account_GPU_seconds=other+used, cap_seconds=CAPS[account],
            allowance_seconds=allowance, exit_code=None if child is None else child.returncode,
            stopped_at_cap=capped,stopped_for_aggregate_cap=aggregate_cap,identity=identity,
            stopped_for_disk=disk_stop is not None,disk_stop_check=disk_stop,
            stopped_for_priority_reserve=priority_cap,remaining_priority_reserve_seconds=priority_reserve,
            includes_complete_child_lifetime=True,closed_utc=now()))
        if pause_heavy:
            resume_cpu(job, paused)
    check('GPU child completed within allowance', child is not None and child.returncode == 0 and not capped)
    return dict(GPU_seconds=used, cumulative_account_GPU_seconds=account_total(account), account=account,
        physical_GPU=gpu, worker_exit_code=child.returncode, paused_CPU_processes=len(paused))
