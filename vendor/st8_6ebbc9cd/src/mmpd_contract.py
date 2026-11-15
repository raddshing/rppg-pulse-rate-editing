"""OPS2 stub replacing src/mmpd_contract.py.

The original reads the full MMPD index (including sealed MMPD-B metadata).
OPS2 never calls it; only grid._prepare/_estimate are used.
"""


def approved_metadata(*args, **kwargs):
    raise RuntimeError('MMPD metadata access is disabled in the OPS2 workspace')


def public_analysis_registry(*args, **kwargs):
    raise RuntimeError('MMPD registry access is disabled in the OPS2 workspace')
