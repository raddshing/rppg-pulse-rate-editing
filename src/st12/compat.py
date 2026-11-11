"""Bind frozen scientific helpers to the instruction05B operational namespace."""
import sys
from src.st12 import ops


def bind():
    previous = sys.modules.get('src.st9.ops')
    if previous is not None and previous is not ops:
        raise RuntimeError('Bind instruction05B paths before importing frozen helpers')
    sys.modules['src.st9.ops'] = ops
    return ops
