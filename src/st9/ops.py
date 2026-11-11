"""Public stand-in for the project's operational module (paths, guards, constants).

Paths come from environment variables; numeric constants (GRID, BANDS) are the project's.
"""
import csv, datetime as dt, hashlib, json, os, time
from pathlib import Path
import numpy as np
ROOT = Path(os.environ.get('PULSE_EDIT_ROOT', Path(__file__).resolve().parents[2]))
NAMESPACE = os.environ.get('ST9_NAMESPACE', 'st9e')
CACHE = Path(os.environ.get('PULSE_EDIT_CACHE', ROOT / 'cache')) / NAMESPACE
TMP = CACHE / 'tmp'
REPORTS = ROOT / 'reports' / NAMESPACE
GRID = 3.515625
BANDS = {'HAR': (.6, 3.3), 'L0': (.75, 2.5)}


class RequiredStop(RuntimeError):
    pass


def check(message, condition):
    if not condition:
        raise RequiredStop(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with open(ROOT / path, newline='') as f:
        return list(csv.DictReader(f))


def write_rows(path, data, fields=None):
    data = list(data)
    fields = fields or (list(data[0]) if data else [])
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(data)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=1, default=str) + '\n')


def finish(name, fn):
    print(json.dumps(fn(), default=str)[:2000])
