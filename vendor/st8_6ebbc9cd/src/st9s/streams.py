"""Separate original, actual pool-index and selector streams for instruction03."""
import random
import numpy as np


def seed_globals(seed, torch_module):
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)


def seed_worker(worker_id, seed):
    import torch
    seed_globals(seed + worker_id, torch)


class SourceStreams:
    def __init__(self, n, pool_size, seed, weights=None):
        self.n = int(n)
        self.pool_size = int(pool_size)
        self.original = np.random.default_rng([seed, 0])
        self.augmentation = np.random.default_rng([seed, 1])
        self.selector = np.random.default_rng([seed, 2])
        self.order = np.empty(0, int)
        self.position = 0
        self.weights = None if weights is None else np.asarray(weights) / np.sum(weights)

    def next_original(self, count):
        out = []
        while len(out) < count:
            if self.position == len(self.order):
                self.order = self.original.permutation(self.n)
                self.position = 0
            take = min(count - len(out), len(self.order) - self.position)
            out.extend(self.order[self.position:self.position + take])
            self.position += take
        return np.asarray(out, int)

    def batch(self, arm, batch_size=4):
        if arm == '0':
            return self.next_original(batch_size), np.empty(0, int), []
        if arm == 'vii':
            if self.weights is None:
                raise ValueError('LDS requires frozen weights')
            return self.original.choice(self.n, batch_size, p=self.weights), np.empty(0, int), []
        if self.pool_size < 1 or batch_size % 2:
            raise ValueError('Augmented batches require an actual pool and even batch size')
        original = self.next_original(batch_size // 2)
        aug = self.augmentation.integers(0, self.pool_size, batch_size // 2)
        methods = list(np.where(self.selector.random(len(aug)) < .5, 'i', 'ii-h')) if arm == 'iii' else [arm] * len(aug)
        return original, aug, methods


def audit_streams(n, pool_size, weights, seed=0):
    streams = {a:SourceStreams(n, pool_size, seed, weights) for a in ['0','i','ii-h','iv-h','vii','iii']}
    reference_original = np.random.default_rng([seed, 0])
    expected_original = reference_original.permutation(n)[:8]
    expected_aug = np.random.default_rng([seed, 1]).integers(0, pool_size, 8)
    expected_selector = np.where(np.random.default_rng([seed, 2]).random(8) < .5, 'i', 'ii-h').tolist()
    expected_lds = np.random.default_rng([seed, 0]).choice(n, 8, p=np.asarray(weights) / np.sum(weights))
    observations = {}
    for a, stream in streams.items():
        batches = [stream.batch(a) for _ in range(2 if a in ['0','vii'] else 4)]
        observations[a] = dict(original=np.concatenate([x[0] for x in batches]).tolist(),
            augmentation=np.concatenate([x[1] for x in batches]).tolist(), methods=sum([x[2] for x in batches], []))
    assertions = dict(
        original_only_permutation=all(observations[a]['original'] == expected_original.tolist() for a in ['0','i','ii-h','iv-h','iii']),
        original_only_LDS_choice=observations['vii']['original'] == expected_lds.tolist(),
        augmentation_only_actual_pool_integers=all(observations[a]['augmentation'] == expected_aug.tolist() for a in ['i','ii-h','iv-h','iii']),
        same_pool_size_same_indices=len({tuple(observations[a]['augmentation']) for a in ['i','ii-h','iv-h']}) == 1,
        selector_only_iii=observations['iii']['methods'] == expected_selector,
        target_draw_removed=not hasattr(SourceStreams, 'target'))
    untouched_selector = np.random.default_rng([seed, 2]).bit_generator.state
    untouched_aug = np.random.default_rng([seed, 1]).bit_generator.state
    assertions['unused_selectors_untouched'] = all(streams[a].selector.bit_generator.state == untouched_selector for a in ['0','i','ii-h','iv-h','vii'])
    assertions['unaugmented_pool_stream_untouched'] = all(streams[a].augmentation.bit_generator.state == untouched_aug for a in ['0','vii'])
    return dict(seed=seed, original_count=n, actual_pool_size=pool_size, assertions=assertions,
        passed=all(assertions.values()), observations=observations)
