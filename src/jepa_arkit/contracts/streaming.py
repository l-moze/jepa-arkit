from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jepa_arkit.errors import ContractError
from jepa_arkit.io import load_json, stable_hash


@dataclass(frozen=True)
class StreamingProtocol:
    protocol_id: str
    sample_rate: int
    output_fps: int
    chunk_ms: int
    lookahead_ms: int
    overlap_ms: int
    history_ms: int
    state_limit_frames: int
    timestamp_origin: str
    tail_flush: str
    interpolation_owner: str
    model_context_alignment: str = "right_align_valid_context_left_neutral_pad"

    @classmethod
    def from_file(cls, path: str | Path) -> StreamingProtocol:
        value = load_json(path)
        protocol = cls(**value)
        protocol.validate()
        return protocol

    def validate(self) -> None:
        if self.sample_rate != 16000 or self.output_fps != 30:
            raise ContractError("v1 streaming requires 16 kHz input and 30 fps model output")
        if self.chunk_ms <= 0 or self.lookahead_ms < 0 or self.overlap_ms < 0:
            raise ContractError("streaming durations must be non-negative")
        if self.overlap_ms >= self.chunk_ms:
            raise ContractError("overlap must be smaller than chunk")
        if self.interpolation_owner != "ue":
            raise ContractError("30 to 60 fps interpolation must be owned by UE")
        if self.model_context_alignment != "right_align_valid_context_left_neutral_pad":
            raise ContractError("unsupported model context alignment")

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.__dict__)

    def output_timestamp(self, frame_index: int) -> float:
        if frame_index < 0:
            raise ContractError("frame index cannot be negative")
        return frame_index / self.output_fps
