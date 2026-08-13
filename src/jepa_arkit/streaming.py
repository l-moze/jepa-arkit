from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
import torch

from jepa_arkit.contracts.streaming import StreamingProtocol
from jepa_arkit.errors import ContractError
from jepa_arkit.models.direct import DirectCausalModel


@dataclass(frozen=True)
class StreamOutput:
    """A contiguous block of model outputs with canonical frame timestamps."""

    start_frame: int
    predictions: np.ndarray
    timestamps: np.ndarray


class DirectStreamingSession:
    """Reference causal session for a direct model over 30 fps aligned features.

    The session accepts timestamp-indexed chunks. Overlapping input chunks are deduplicated,
    look-ahead delays emission, and `flush` emits all real frames without exposing padded frames.
    """

    def __init__(
        self,
        model: DirectCausalModel,
        protocol: StreamingProtocol,
        *,
        device: torch.device,
        max_frames: int,
    ) -> None:
        self.model = model
        self.protocol = protocol
        self.device = device
        self.fps = protocol.output_fps
        self.chunk_frames = max(1, round(protocol.chunk_ms * self.fps / 1000))
        self.overlap_frames = max(0, round(protocol.overlap_ms * self.fps / 1000))
        self.lookahead_frames = ceil(protocol.lookahead_ms * self.fps / 1000)
        self.history_frames = round(protocol.history_ms * self.fps / 1000)
        self.max_frames = max_frames
        if self.history_frames + self.chunk_frames > max_frames:
            raise ContractError("streaming history plus chunk exceeds model max_frames")
        self._features: np.ndarray | None = None
        self._next_emit = 0
        self._flushed = False
        self.received_chunks = 0
        self.deduplicated_frames = 0
        self.model_calls = 0

    @property
    def received_frames(self) -> int:
        return 0 if self._features is None else int(self._features.shape[0])

    @property
    def emitted_frames(self) -> int:
        return self._next_emit

    def push(self, features: np.ndarray, *, start_frame: int) -> tuple[StreamOutput, ...]:
        if self._flushed:
            raise ContractError("cannot push after flush")
        if features.ndim != 2 or features.shape[0] == 0:
            raise ContractError("stream features must have shape [T, D] with T > 0")
        if start_frame < 0:
            raise ContractError("start_frame cannot be negative")
        if self._features is None:
            if start_frame != 0:
                raise ContractError("first stream chunk must start at frame zero")
            self._features = features.astype(np.float32, copy=True)
        else:
            end_frame = self.received_frames
            if start_frame > end_frame:
                raise ContractError("stream chunk has a gap")
            overlap = max(0, end_frame - start_frame)
            if overlap:
                overlap = min(overlap, features.shape[0])
                existing = self._features[start_frame : start_frame + overlap]
                if not np.allclose(existing, features[:overlap], rtol=1e-4, atol=1e-5):
                    raise ContractError("overlapping stream chunks disagree")
                self.deduplicated_frames += overlap
            tail = features[overlap:]
            if len(tail):
                self._features = np.concatenate((self._features, tail.astype(np.float32)), axis=0)
        self.received_chunks += 1
        return self._emit(final=False)

    def flush(self) -> tuple[StreamOutput, ...]:
        if self._flushed:
            return ()
        self._flushed = True
        return self._emit(final=True)

    @torch.no_grad()
    def _emit(self, *, final: bool) -> tuple[StreamOutput, ...]:
        if self._features is None:
            return ()
        ready_end = self.received_frames if final else max(
            0, self.received_frames - self.lookahead_frames
        )
        outputs: list[StreamOutput] = []
        self.model.eval()
        while self._next_emit < ready_end:
            block_end = min(ready_end, self._next_emit + self.chunk_frames)
            context_start = max(0, self._next_emit - self.history_frames)
            context_start = max(context_start, block_end - self.max_frames)
            context = self._features[context_start:block_end]
            left_padding = self.max_frames - len(context)
            if left_padding > 0:
                context = np.pad(context, ((left_padding, 0), (0, 0)), mode="constant")
            tensor = torch.from_numpy(context).to(self.device).unsqueeze(0)
            prediction = self.model(tensor)[0].float().cpu().numpy()
            self.model_calls += 1
            offset = left_padding + self._next_emit - context_start
            block = prediction[offset : offset + block_end - self._next_emit]
            start = self._next_emit
            self._next_emit = block_end
            outputs.append(
                StreamOutput(
                    start_frame=start,
                    predictions=block,
                    timestamps=np.arange(start, block_end, dtype=np.float64) / self.fps,
                )
            )
        return tuple(outputs)


def protocol_chunk_starts(total_frames: int, protocol: StreamingProtocol) -> tuple[int, ...]:
    """Return deterministic overlapping input chunk starts for a reference trace."""
    if total_frames <= 0:
        return ()
    chunk = max(1, round(protocol.chunk_ms * protocol.output_fps / 1000))
    overlap = max(0, round(protocol.overlap_ms * protocol.output_fps / 1000))
    step = max(1, chunk - overlap)
    return tuple(range(0, total_frames, step))


def run_reference_trace(
    model: DirectCausalModel,
    protocol: StreamingProtocol,
    features: np.ndarray,
    *,
    device: torch.device,
    max_frames: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    """Feed a fixed aligned trace and return exactly one output per real frame."""
    if features.ndim != 2:
        raise ContractError("features must have shape [T, D]")
    session = DirectStreamingSession(
        model, protocol, device=device, max_frames=max_frames
    )
    starts = protocol_chunk_starts(len(features), protocol)
    output_blocks: list[StreamOutput] = []
    chunk = max(1, round(protocol.chunk_ms * protocol.output_fps / 1000))
    for start in starts:
        output_blocks.extend(
            session.push(features[start : min(len(features), start + chunk)], start_frame=start)
        )
    output_blocks.extend(session.flush())
    if not output_blocks:
        raise ContractError("stream trace emitted no frames")
    predictions = np.concatenate([block.predictions for block in output_blocks], axis=0)
    timestamps = np.concatenate([block.timestamps for block in output_blocks], axis=0)
    expected = np.arange(len(features), dtype=np.float64) / protocol.output_fps
    if len(predictions) != len(features) or not np.array_equal(timestamps, expected):
        raise ContractError("stream trace did not emit exactly one contiguous frame per input")
    deltas = (
        np.linalg.norm(np.diff(predictions, axis=0), axis=1)
        if len(predictions) > 1
        else np.zeros(0)
    )
    boundary_indices = [block.start_frame - 1 for block in output_blocks[1:]]
    boundary_deltas = np.asarray(
        [deltas[index] for index in boundary_indices if 0 <= index < len(deltas)],
        dtype=np.float64,
    )
    interior_mask = np.ones(len(deltas), dtype=bool)
    interior_mask[boundary_indices] = False
    interior_deltas = deltas[interior_mask]
    report: dict[str, int | float] = {
        "input_frames": len(features),
        "output_frames": len(predictions),
        "input_chunks": session.received_chunks,
        "deduplicated_frames": session.deduplicated_frames,
        "model_calls": session.model_calls,
        "lookahead_frames": session.lookahead_frames,
        "history_frames": session.history_frames,
        "chunk_frames": session.chunk_frames,
        "overlap_frames": session.overlap_frames,
        "max_context_frames": max_frames,
        "output_blocks": len(output_blocks),
        "boundary_count": len(boundary_deltas),
        "boundary_jump_max": float(boundary_deltas.max()) if len(boundary_deltas) else 0.0,
        "interior_step_median": float(np.median(interior_deltas))
        if len(interior_deltas)
        else 0.0,
        "boundary_to_interior_median_ratio": float(
            np.median(boundary_deltas) / max(float(np.median(interior_deltas)), 1e-8)
        )
        if len(boundary_deltas) and len(interior_deltas)
        else 0.0,
    }
    return predictions, timestamps, report
