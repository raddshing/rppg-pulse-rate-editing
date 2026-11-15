"""GPU inference server of the Stage-5 evaluation (stage 2). Copy of audit/infer_server.py (the same archived create/predict of
src/st13/sealed_model.py at commit b81f6c9e6, the same architecture configs, cuDNN deterministic, TF32 off for matmul, the same
per-model random state) with three additions for brief 02 6.45 / 6.48:
  - 'stream': the input stream a model reads (default: its view). A stream is one prepared input (e.g. 'RhythmFormer' and
    'RhythmFormer@blur15'); a predict request names a stream and runs every model of that stream.
  - 'batch': chunks per forward pass, 4 (default; the October batching) or 1 (matched FactorizePhys batch-1 sensitivity).
  - 'run_id' may be null (no check of the run name stored in the checkpoint; the file is checked by SHA-256 before loading).
  A model that fails to load is reported with its error and not added (automatic defect, brief 02 6.48 마 4).
  Every reply carries the CUDA caching-allocator OOM count (torch.cuda.memory_stats()['num_ooms']). Rehearsal 11-14: when
  memory runs short, cuDNN (v8 API, torch 2.1.2, cuDNN 8.9.2) catches the workspace OOM and runs another deterministic plan
  with different rounding, without an error, and keeps that plan for later calls; the count is the only trace, so the client
  stops on any nonzero count. The cache is kept between requests (no empty_cache, as audit/infer_server.py) so that memory
  reserved by the first batches stays with this process.
  Brief 02 6.67 다 3: every reply also carries 'ooms_by_key', the increase of that count during each checkpoint's load
  (load reply) or forward passes (predict reply), so that the client records the allocator OOMs per checkpoint.
Protocol over stdin/stdout (one JSON line each, payload float32 THWC bytes):
  {"load": [{"key", "view", "checkpoint", "run_id", "stream", "batch", "md_inference", "rng_seed"}]} -> {"ok", "loaded", "failed"}
  {"predict": stream, "shape": [n,T,H,W,C], "rng": "reset"|"carry"} + bytes -> {"ok","keys","shape"} + float32 bytes
  {"close": true}
"""
import os  # noqa: E402  (first: thread settings precede numpy/torch imports)
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENCV_FOR_THREADS_NUM='16')
import contextlib  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

V = Path(os.environ['STAGE5_VENDOR'])  # the archived source tree (src/, configs/, third_party/rPPG-Toolbox)
REPO = V / 'third_party' / 'rPPG-Toolbox'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(V))
os.chdir(REPO)
import torch  # noqa: E402
import yaml  # noqa: E402
from yacs.config import CfgNode  # noqa: E402

torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = True
with contextlib.redirect_stdout(sys.stderr):
    from src.st13.sealed_model import create, predict  # noqa: E402

SETTINGS = json.loads((V / 'configs' / 'inference_settings.json').read_text())


def rng_get():
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state().clone(),
                cuda=[x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])


def rng_put(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def cfg_for(view, md_inference=None):
    cfg = CfgNode(yaml.safe_load((V / 'configs' / f'{view}_architecture.yaml').read_text()))
    cfg.defrost()
    cfg.DEVICE = 'cuda:0'
    cfg.NUM_OF_GPU_TRAIN = 1
    cfg.TEST.DATA.PREPROCESS = CfgNode(SETTINGS[view]['preprocessing'])
    if view in ('FPm', 'FactorizePhys') and md_inference is not None:  # FSAM at inference (A3 item 3)
        cfg.MODEL.FactorizePhys.MD_INFERENCE = bool(md_inference)
    cfg.freeze()
    return cfg


def num_ooms():
    return int(torch.cuda.memory_stats().get('num_ooms', 0)) if torch.cuda.is_initialized() else 0


def main():
    out = sys.stdout.buffer
    models, initial, states = {}, {}, {}
    for line in sys.stdin.buffer:
        req = json.loads(line)
        if req.get('close'):
            break
        if 'load' in req:
            loaded, failed, load_ooms = [], [], {}
            for m in req['load']:
                n0 = num_ooms()  # brief 02 6.67 다 3: allocator OOMs per checkpoint
                try:
                    view = m['view']
                    name = SETTINGS[view]['architecture_name']
                    cfg = cfg_for(view, m.get('md_inference'))
                    seed = int(m.get('rng_seed', 100))
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    np.random.seed(seed)
                    random.seed(seed)
                    with contextlib.redirect_stdout(sys.stderr):
                        model = create(name, cfg)
                    state = torch.load(m['checkpoint'], map_location='cpu')
                    if m.get('run_id') is not None:
                        assert state['run_id'] == m['run_id'], (state['run_id'], m['run_id'])
                    weights = state['model']
                    if name == 'FactorizePhys' and not next(iter(weights)).startswith('module.'):
                        weights = {'module.' + k: v for k, v in weights.items()}
                    model.load_state_dict(weights, strict=True)
                    model.eval()
                except Exception as e:  # automatic defect: load failure (the model is not used)
                    failed.append(dict(key=m['key'], error=repr(e)[:300]))
                    load_ooms[m['key']] = num_ooms() - n0
                    continue
                batch = int(m.get('batch', 4))
                assert batch in (1, 4)
                models[m['key']] = dict(model=model, name=name, cfg=cfg, view=view, stream=m.get('stream', view), batch=batch)
                initial[m['key']] = rng_get()
                states[m['key']] = rng_get()
                loaded.append(dict(key=m['key'], run_id=state.get('run_id'), epoch=int(state['epoch'])))
                load_ooms[m['key']] = num_ooms() - n0
            out.write((json.dumps(dict(ok=True, loaded=loaded, failed=failed, num_ooms=num_ooms(), ooms_by_key=load_ooms)) + '\n').encode())
            out.flush()
            continue
        stream = req['predict']
        shape = tuple(req['shape'])
        raw = sys.stdin.buffer.read(int(np.prod(shape)) * 4)
        data = np.frombuffer(raw, np.float32).reshape(shape)
        keys = [k for k, v in models.items() if v['stream'] == stream]
        mode = req.get('rng', 'reset')
        assert mode in ('reset', 'carry')
        if mode == 'reset':
            for k in keys:
                states[k] = {a: (b.clone() if hasattr(b, 'clone') else b) for a, b in initial[k].items()}
                states[k]['cuda'] = [x.clone() for x in initial[k]['cuda']]
        result = np.empty((len(keys), shape[0], shape[1]), np.float32)
        ooms_by_key = {k: 0 for k in keys}
        with torch.no_grad():
            for begin in range(0, shape[0], 4):
                chunk = data[begin:begin + 4]
                fmts = {SETTINGS[models[k]['view']]['data_format'] for k in keys}
                assert len(fmts) <= 1  # one stream = one view
                fmt = fmts.pop() if fmts else None
                arr = chunk.transpose(0, 1, 4, 2, 3) if fmt == 'NDCHW' else chunk.transpose(0, 4, 1, 2, 3)
                tensor = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to('cuda:0')  # as audit/infer_server.py
                for i, k in enumerate(keys):
                    m = models[k]
                    n0 = num_ooms()
                    if m['batch'] == 4:
                        rng_put(states[k])
                        result[i, begin:begin + len(chunk)] = predict(m['model'], m['name'], tensor.clone(), m['cfg']).detach().cpu().numpy()
                        states[k] = rng_get()
                    else:  # batch 1: one chunk per forward pass
                        for j in range(len(chunk)):
                            rng_put(states[k])
                            result[i, begin + j:begin + j + 1] = predict(m['model'], m['name'], tensor[j:j + 1].clone(), m['cfg']).detach().cpu().numpy()
                            states[k] = rng_get()
                    ooms_by_key[k] += num_ooms() - n0
        stats = torch.cuda.memory_stats()
        out.write((json.dumps(dict(ok=True, keys=keys, shape=list(result.shape), num_ooms=int(stats.get('num_ooms', 0)), ooms_by_key=ooms_by_key,
                                   reserved_mb=round(torch.cuda.memory_reserved() / 2**20))) + '\n').encode())
        out.write(result.tobytes())
        out.flush()


if __name__ == '__main__':
    main()
