"""Add explicitly named new-stage dependencies to the scoped immutable inventory."""
from src.st12.ops import *


def _register_unlocked(paths, scope, receipt):
    """Hash each requested file only; inherited manifest contents are metadata."""
    expected=rows('outputs/st12_inputs_expected.csv');known={r['path']:r for r in expected}
    authority={}
    for name in ['reports/st1_exit_manifest.csv','outputs/st10_input_hash_checks.csv','outputs/st11_input_hash_checks.csv','outputs/st12_st11_manifest.csv']:
        for r in rows(name):authority[path_key(r['path'])]=dict(r,authority=name)
    checks=[]
    for p in sorted(set(map(path_key,paths))):
        check('Explicit dependency excludes sealed path','/sealed/' not in p)
        actual=ROOT/p;digest=sha(actual);size=actual.stat().st_size
        prior=known.get(p,authority.get(p))
        matches=prior is None or (digest==prior['sha256'] and size==int(prior['bytes']))
        checks.append(dict(path=p,bytes=size,sha256=digest,comparison_available=prior is not None,passed=matches,
                           authority=prior.get('authority','existing_scoped_input') if prior else 'new_preuse_baseline'))
        check('Scoped dependency unchanged: '+p,matches,scope=scope)
        if p not in known:
            row=dict(path=p,bytes=size,sha256=digest,scope=scope,authority=checks[-1]['authority']);expected.append(row);known[p]=row
    write_rows('outputs/st12_inputs_expected.csv',expected)
    save_json(receipt,dict(scope=scope,checks=checks,checked_utc=now(),manifest_sha256=sha(ROOT/'outputs/st12_inputs_expected.csv')))
    return checks


def register(paths, scope, receipt):
    with (ROOT/'reports/st12/.input_supplement.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        try:return _register_unlocked(paths,scope,receipt)
        finally:fcntl.flock(handle,fcntl.LOCK_UN)
