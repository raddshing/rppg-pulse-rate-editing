"""Frozen PURE generator functions with ST6 ownership and one prescribed A override."""
from src.st30p.ops import *
configure('st30g')
import ast
import copy
import math
import numpy as np
import cv2
from fractions import Fraction
from scipy.signal import butter,filtfilt,sosfiltfilt,resample_poly,windows,periodogram,find_peaks
from src.st11x.future import frozen_reference_functions
from src.st11x.pathways import BANDS,GRID
from src.st3.readers import pos as inherited_pos,chrom as inherited_chrom
from src.eval.har import detrend
from src.st9g.preprocess import diff_normalize_data,diff_normalize_label,standardized_data,standardized_label

REPORTS=ROOT/'reports/st30g';NAMESPACE='st30g'
common_hr=frozen_reference_functions()['common_hr'];filter4=frozen_reference_functions()['filter4']


def check(name,value,**kwargs):
    if not value:raise RuntimeError(name)


def source_hash(path):
    settings=json.loads((REPORTS/'A_pool_settings.json').read_text())
    assert sha(path)==settings['source_hashes'][str(path)],str(path)


def ledger(kind,event,pid):
    append('reports/st30g/children.csv',dict(at_utc=now(),kind=kind,event=event,pid=pid))


def load(path,names):
    tree=ast.parse((ROOT/path).read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ROOT/path),'exec'),globals())


load('src/st9e/native.py',['contact','frame_files','read_pure','reflected_indices','parameters','resample','editor_band','normalize','skin_mask','tracking_options'])
load('src/st9e/inherited.py',['ppg_vectors','green','band_reader_pos','band_reader_chrom','stochastic_round'])
load('src/st9e/editor.py',['bands_for','face_masks','notch_all','weighted_rgb','calibration','edited','injection_waveform'])
load('src/st9e/generator.py',['raw_target_label','reflection_used'])
load('src/st9e/temporal.py',['shifted_start'])
load('src/st9e/model_view.py',['ViewWorker','crop_native'])
load('src/st12c/rhythm_view.py',['RhythmViewWorker'])

# Keep every inherited arithmetic statement. The only scientific insertion is
# the directly prescribed fixed amplitude, after its old contextual calculation.
tree=ast.parse((ROOT/'src/st9e/editor.py').read_text())
node=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='edited'))
node.name='edited_with_fixed_A';node.args.args.append(ast.arg(arg='fixed_A'));node.args.defaults.append(ast.Constant(value=None))
position=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
    and isinstance(n.value.func,ast.Name) and n.value.func.id=='check')
node.body.insert(position,ast.parse('if fixed_A is not None:\n    A = float(fixed_A)\n').body[0])
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(ROOT/'src/st9e/editor.py')+':ST6_fixed_A','exec'),globals())


def snapshot(path):
    s=Path(path).stat();return dict(bytes=s.st_size,device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


def verify():
    settings=json.loads((REPORTS/'A_pool_settings.json').read_text())
    assert sha(REPORTS/'A_pool_settings.json') in (ROOT/'reports/ST6_INSTR01_PREFIX.md').read_text()
    for p,h in settings['source_hashes'].items():assert sha(p)==h,p
    assert sha('outputs/st30g_A_targets.csv')==settings['targets_sha256']
    return settings
