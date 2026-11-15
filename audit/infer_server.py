"""GPU inference server (repro environment, as in the draft's environment lock).

Uses the archived create/predict (src/st13/sealed_model.py, commit b81f6c9e6) and the
draft's architecture configs. Protocol over stdin/stdout:
  {"load": [{"key", "view", "checkpoint", "run_id"}]}            -> {"ok", "loaded"}
  {"predict": view, "shape": [n,T,H,W,C], "rng": "reset"|"carry"} + float32 THWC bytes
                                                                 -> {"ok","keys","shape"} + float32 bytes
  {"close": true}
Random state (as the draft's st30e/infer_group.py): every model gets its own RNG state, taken right
after seed 100 + create + weight load, and each batch runs with that model's state (FactorizePhys uses
random numbers at inference). "carry" continues each model's state across calls (draft protocol with
one global chunk order); "reset" restarts every call from the post-load state (deterministic per call).
Before 2026-11-10 23:00 the models shared one RNG stream; outputs of 2.3 were produced that way.
"""
import os  # noqa: E402  (first: thread settings must precede numpy/torch/cv2 imports; children inherit)
# brief 02 6.14/6.16: OPS2_THREADS=one (default) fixes BLAS/OpenMP to 1 thread; OPS2_THREADS=default states the
# library defaults of the earlier unrestricted runs on this 16-CPU server explicitly (OpenMP 8, OpenBLAS 16,
# MKL 8). OpenCV (pthreads) keeps its unrestricted default of 16 threads in both modes, also under CPU affinity.
os.environ.update({'one': dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                   'default': dict(OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='16', MKL_NUM_THREADS='8')}[
    os.environ.get('OPS2_THREADS', 'one')], OPENCV_FOR_THREADS_NUM='16')
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
V = WORK / 'vendor' / 'st8_6ebbc9cd'
REPO = V / 'third_party' / 'rPPG-Toolbox'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(V))
os.chdir(REPO)
import contextlib  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from yacs.config import CfgNode  # noqa: E402

torch.set_num_threads(1)
torch.backends.cudnn.deterministic = os.environ.get('OPS2_CUDNN_DETERMINISTIC', '1') == '1'  # brief 02 6.34: '0' only for the batch/determinism check
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
    if view in ('FPm', 'FactorizePhys') and os.environ.get('OPS2_DIAG_MD_INFERENCE') in ('0', '1'):  # diagnostic only
        cfg.MODEL.FactorizePhys.MD_INFERENCE = os.environ['OPS2_DIAG_MD_INFERENCE'] == '1'
    if view in ('FPm', 'FactorizePhys') and md_inference is not None:  # brief 02 6.19 B8 (FSAM off at inference)
        cfg.MODEL.FactorizePhys.MD_INFERENCE = bool(md_inference)
    cfg.freeze()
    return cfg


def main():
    out = sys.stdout.buffer
    models = {}
    initial, states = {}, {}
    for line in sys.stdin.buffer:
        req = json.loads(line)
        if req.get('close'):
            break
        if 'load' in req:
            loaded = []
            for m in req['load']:
                view = m['view']
                name = SETTINGS[view]['architecture_name']
                cfg = cfg_for(view, m.get('md_inference'))
                seed = int(m.get('rng_seed', 100))  # brief 02 6.19 B8: inference RNG repeats; default 100
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                with contextlib.redirect_stdout(sys.stderr):
                    model = create(name, cfg)
                state = torch.load(m['checkpoint'], map_location='cpu')
                assert state['run_id'] == m['run_id'], (state['run_id'], m['run_id'])
                weights = state['model']
                if name == 'FactorizePhys' and not next(iter(weights)).startswith('module.'):
                    weights = {'module.' + k: v for k, v in weights.items()}
                model.load_state_dict(weights, strict=True)
                model.eval()
                models[m['key']] = (model, name, cfg, view, SETTINGS[view]['data_format'])
                initial[m['key']] = rng_get()
                states[m['key']] = rng_get()
                loaded.append(dict(key=m['key'], run_id=state['run_id'], epoch=int(state['epoch'])))
            out.write((json.dumps(dict(ok=True, loaded=loaded)) + '\n').encode())
            out.flush()
            continue
        view = req['predict']
        shape = tuple(req['shape'])
        raw = sys.stdin.buffer.read(int(np.prod(shape)) * 4)
        data = np.frombuffer(raw, np.float32).reshape(shape)
        keys = [k for k, v in models.items() if v[3] == view]
        mode = req.get('rng', 'reset')
        assert mode in ('reset', 'carry')
        if mode == 'reset':
            for k in keys:
                states[k] = {a: (b.clone() if hasattr(b, 'clone') else b) for a, b in initial[k].items()}
                states[k]['cuda'] = [x.clone() for x in initial[k]['cuda']]
        result = np.empty((len(keys), shape[0], shape[1]), np.float32)
        with torch.no_grad():
            for begin in range(0, shape[0], 4):
                chunk = data[begin:begin + 4]
                fmt = SETTINGS[view]['data_format']
                arr = chunk.transpose(0, 1, 4, 2, 3) if fmt == 'NDCHW' else chunk.transpose(0, 4, 1, 2, 3)
                tensor = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to('cuda:0')
                for i, k in enumerate(keys):
                    model, name, cfg, _, _ = models[k]
                    rng_put(states[k])
                    result[i, begin:begin + len(chunk)] = predict(model, name, tensor.clone(), cfg).detach().cpu().numpy()
                    states[k] = rng_get()
        ooms = int(torch.cuda.memory_stats().get('num_ooms', 0))  # 11-14: cuDNN silently changes plan after an allocator OOM
        if ooms:
            print(f'WARNING allocator OOM count {ooms}: outputs of this server may depend on free GPU memory', file=sys.stderr, flush=True)
        out.write((json.dumps(dict(ok=True, keys=keys, shape=list(result.shape), num_ooms=ooms)) + '\n').encode())
        out.write(result.tobytes())
        out.flush()


if __name__ == '__main__':
    main()
