from pathlib import Path

import numpy as np
import pytest
import torch

from jepa_arkit.contracts.streaming import StreamingProtocol
from jepa_arkit.errors import ContractError
from jepa_arkit.models.direct import DirectCausalModel
from jepa_arkit.streaming import DirectStreamingSession, protocol_chunk_starts, run_reference_trace

ROOT = Path(__file__).parents[1]


def _protocol() -> StreamingProtocol:
    return StreamingProtocol.from_file(ROOT / "configs/contracts/streaming_protocol_candidate.json")


def test_streaming_session_deduplicates_overlap_and_flushes() -> None:
    protocol = _protocol()
    model = DirectCausalModel(4, 3, model_dim=16, layers=1, heads=4, max_frames=120).eval()
    features = np.arange(30 * 4, dtype=np.float32).reshape(30, 4)
    session = DirectStreamingSession(model, protocol, device=torch.device("cpu"), max_frames=120)
    chunks = protocol_chunk_starts(len(features), protocol)
    outputs = []
    chunk = round(protocol.chunk_ms * protocol.output_fps / 1000)
    for start in chunks:
        outputs.extend(session.push(features[start : start + chunk], start_frame=start))
    outputs.extend(session.flush())
    timestamps = np.concatenate([item.timestamps for item in outputs])
    assert len(timestamps) == len(features)
    np.testing.assert_array_equal(timestamps, np.arange(len(features)) / 30)
    assert session.deduplicated_frames > 0
    assert session.emitted_frames == len(features)


def test_streaming_rejects_disagreeing_overlap() -> None:
    protocol = _protocol()
    model = DirectCausalModel(4, 3, model_dim=16, layers=1, heads=4, max_frames=120)
    session = DirectStreamingSession(model, protocol, device=torch.device("cpu"), max_frames=120)
    session.push(np.zeros((8, 4), dtype=np.float32), start_frame=0)
    with pytest.raises(ContractError, match="overlapping"):
        session.push(np.ones((4, 4), dtype=np.float32), start_frame=4)


def test_reference_trace_returns_one_frame_per_input() -> None:
    protocol = _protocol()
    model = DirectCausalModel(4, 3, model_dim=16, layers=1, heads=4, max_frames=120)
    features = np.random.default_rng(3).normal(size=(40, 4)).astype(np.float32)
    predictions, timestamps, report = run_reference_trace(
        model, protocol, features, device=torch.device("cpu"), max_frames=120
    )
    assert predictions.shape == (40, 3)
    assert report["output_frames"] == 40
    np.testing.assert_array_equal(timestamps, np.arange(40) / 30)
