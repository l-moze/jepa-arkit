import io
import json
import zipfile
from pathlib import Path

import numpy as np

from jepa_arkit.data.unitalker import audit_unitalker_candidate


def test_unitalker_candidate_stays_quarantined(tmp_path: Path) -> None:
    archive_path = tmp_path / "candidate.zip"
    base = "unitalker_data_release_V1/D6_unitalker_Chinese_speech"
    buffer = io.BytesIO()
    np.save(buffer, np.zeros((3, 51), dtype=np.float32), allow_pickle=False)
    record = {
        "info": {"id_list": ["speaker0"], "total_duration": 0.1},
        "data": [
            {
                "annot_type": "inhouse_blendshape_weight",
                "annot_path": "train/example.npy",
                "audio_path": "train/example.wav",
                "fps": 30,
            }
        ],
    }
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(f"{base}/train.json", json.dumps(record))
        archive.writestr(f"{base}/train/example.npy", buffer.getvalue())
        archive.writestr(f"{base}/train/example.wav", b"RIFF")
    output = tmp_path / "report.json"
    report = audit_unitalker_candidate(archive_path, output)
    assert report["zip_crc_passed"] is True
    assert report["ready_for_training"] is False
    assert report["sources"]["D6_unitalker_Chinese_speech"]["records"] == 1
    assert report["sources"]["D6_unitalker_Chinese_speech"]["missing_audio"] == 0
