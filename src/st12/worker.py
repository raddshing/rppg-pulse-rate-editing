"""Minimal worker receipts preserving exceptions and technical interruptions."""
from src.st12.ops import *
import signal


class TechnicalInterruption(Exception):pass


def run(function,result_path,stage):
    started=time.monotonic()
    def interrupted(sig,_):raise TechnicalInterruption('External signal '+str(sig))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    code=0
    try:
        result=dict(function() or {});result.setdefault('status','complete')
    except TechnicalInterruption as exc:
        code=143;result=dict(status='technical_interruption',reason=str(exc))
    except ResourceWait as exc:
        code=3;result=dict(status='resource_wait',reason=str(exc))
    except RequiredStop as exc:
        code=2;result=dict(status='stopped_required_check',reason=str(exc),scope=exc.scope)
    except Exception as exc:
        import traceback
        code=1;result=dict(status='implementation_error',error_type=type(exc).__name__,
            stack=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name) for f in traceback.extract_tb(exc.__traceback__)])
    result.update(worker_seconds=time.monotonic()-started,finished_utc=now(),stage=stage)
    save_json(result_path,result)
    if code:raise SystemExit(code)
