"""Verify the precommitted author-loss restoration and preserve failed evidence."""
from src.st12.ops import *


def verify_repair():
    path = ROOT / 'reports/st12c/st12c_a_loss_implementation_repair.json'
    repair = json.loads(path.read_text())
    check('Author-loss restoration committed before use', sha(path) in
          (ROOT / 'reports/ST4_INSTR05B_PREFIX.md').read_text())
    for name, digest in repair['source_hashes'].items():
        check('Exact author-loss restoration source', sha(ROOT / name) == digest)
    for name, digest in repair['preserved_evidence_hashes'].items():
        check('Original failed attempt and timing evidence retained', sha(ROOT / name) == digest)
    check('Restore prescribed author training without new settings',
          repair['scientific_settings_unchanged'] and repair['same_author_training_contract'] and
          repair['completed_training_steps_before_failure'] == 0 and
          repair['completed_training_epochs_before_failure'] == 0)
    repair['receipt_sha256'] = sha(path)
    return repair
