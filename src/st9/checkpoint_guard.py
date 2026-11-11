"""Instruction03 bulk-free write stop, independent of model and sampling state."""
import shutil
from src.st9.ops import PATHS, ROOT, check


def require_checkpoint_capacity():
    # Read only filesystem capacity; consume no Python/NumPy/Torch RNG state.
    # The 60-second sampler separately monitors total new bulk/cache usage.
    free = shutil.disk_usage(PATHS['bulk']).free
    check('Bulk free at least50GB immediately before checkpoint write',
          free >= 50_000_000_000, scope='bulk_free_write_stop')
    check('No sampled resource stop before checkpoint write',
          not (ROOT / 'reports/st9/st9_disk_limit.json').exists(),
          scope='sampled_disk_write_stop')
    return free
