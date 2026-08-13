from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator

import torch
from torch.utils.data import Sampler

from jepa_arkit.training.dataset import MotionWindowDataset


class IdentityBalancedBatchSampler(Sampler[list[int]]):
    """Deterministic batches with bounded windows per identity."""

    def __init__(
        self,
        dataset: MotionWindowDataset,
        *,
        batch_size: int,
        min_identities: int,
        max_identity_share: float,
        seed: int,
        drop_last: bool = True,
    ) -> None:
        if batch_size <= 0 or min_identities <= 0:
            raise ValueError("batch_size and min_identities must be positive")
        if not 0 < max_identity_share <= 1:
            raise ValueError("max_identity_share must be in (0, 1]")
        self.dataset = dataset
        self.batch_size = batch_size
        self.min_identities = min_identities
        self.max_per_identity = max(1, int(batch_size * max_identity_share))
        self.seed = seed
        self.drop_last = drop_last
        grouped: dict[str, list[int]] = defaultdict(list)
        for window_index, (record_index, _) in enumerate(dataset.windows):
            grouped[dataset.records[record_index].face_identity_id].append(window_index)
        if len(grouped) < min_identities:
            raise ValueError(
                f"Dataset has {len(grouped)} identities; sampler requires {min_identities}"
            )
        self.grouped = dict(grouped)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        queues: dict[str, list[int]] = {}
        for identity, indices in self.grouped.items():
            order = torch.randperm(len(indices), generator=generator).tolist()
            queues[identity] = [indices[index] for index in order]
        identities = sorted(queues)
        identity_order = torch.randperm(len(identities), generator=generator).tolist()
        identities = [identities[index] for index in identity_order]
        cursor = 0
        while any(queues.values()):
            available = [identity for identity in identities if queues[identity]]
            if len(available) < self.min_identities:
                break
            chosen = [
                available[(cursor + index) % len(available)]
                for index in range(self.min_identities)
            ]
            cursor = (cursor + self.min_identities) % len(available)
            batch: list[int] = []
            per_identity = 0
            while len(batch) < self.batch_size and any(queues[identity] for identity in chosen):
                per_identity += 1
                if per_identity > self.max_per_identity:
                    break
                for identity in chosen:
                    if queues[identity] and len(batch) < self.batch_size:
                        batch.append(queues[identity].pop())
            if len(batch) == self.batch_size or (batch and not self.drop_last):
                yield batch
