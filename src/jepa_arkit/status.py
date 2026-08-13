from __future__ import annotations

from pathlib import Path

from jepa_arkit.io import load_json


def project_status(root: str | Path = ".") -> dict[str, object]:
    root = Path(root).resolve()

    def read(path: str) -> dict[str, object] | None:
        target = root / path
        return load_json(target) if target.is_file() else None

    catalog = read("artifacts/dataset_catalog_audit.json")
    d0a_demo = read("artifacts/d0a_demo_audit.json")
    direct = read("runs/t0_direct_smoke/metrics.json")
    jepa = read("runs/t0_jepa_smoke/metrics.json")
    real_d0p = read("artifacts/d0p_ravdess_pilot_v1_audit.json")
    e01 = read("runs/e01_ravdess_v2/metrics.json")
    e01_eval = read("runs/e01_ravdess_v2/evaluation.json")
    e10 = read("runs/e10_motion_jepa_ravdess_v1/metrics.json")
    e11 = read("runs/e11_audio_motion_jepa_audio_only_pretrained/metrics.json")
    e03 = read("artifacts/e03_regional_three_seed_summary.json")
    e03_export = read("artifacts/e03_regional_export_invariants.json")
    unitalker = read("artifacts/unitalker_candidate_audit.json")
    inference = read("artifacts/inference/e01_ravdess_actor20_clip.inference.json")
    streaming = read("artifacts/inference/e01_ravdess_actor20_stream_real36s.streaming.json")
    interpolation = read("artifacts/inference/e01_ravdess_actor20_clip_60fps.interpolation.json")
    environment = read("artifacts/environment.json")
    ravdess = read("data/raw/ravdess/release_v1/release.json")
    real_d0a_ready = bool(catalog and catalog.get("d0a_ready"))
    return {
        "milestones": {
            "contracts": "implemented",
            "synthetic_d0a": "passed" if d0a_demo and d0a_demo.get("passed") else "missing",
            "real_d0a": "ready" if real_d0a_ready else "blocked",
            "ravdess_download": "passed"
            if ravdess and ravdess.get("clips_extracted") == 1440
            else "blocked",
            "d0b": "pilot_passed_research_only"
            if real_d0p and real_d0p.get("passed")
            else "blocked_until_e00",
            "t0_direct": "passed" if direct and direct.get("passed") else "blocked",
            "t0_jepa_infrastructure": "passed"
            if jepa and jepa.get("infrastructure_passed")
            else "blocked",
            "e10_representation": "passed"
            if jepa and jepa.get("representation_passed")
            else "blocked",
            "e01_real_direct": "pilot_passed_research_only" if e01 else "missing",
            "e10_real_motion_jepa": "pilot_passed_research_only" if e10 else "missing",
            "e11_real_audio_motion_jepa": "pilot_passed_research_only" if e11 else "missing",
            "e03_regional": e03.get("gate_decision") if e03 else "missing",
            "unitalker_download": "candidate_quarantined"
            if unitalker and unitalker.get("zip_crc_passed")
            else "missing",
            "inference_export": "passed"
            if inference and inference.get("quaternion_norm_max_error", 1) < 1e-4
            else "missing",
            "streaming_reference": "passed"
            if streaming
            and streaming.get("timestamps_contiguous")
            and streaming.get("stream", {}).get("boundary_to_interior_median_ratio", 999) < 2
            else "missing",
            "ue_interpolation_reference": "passed"
            if interpolation and interpolation.get("passed")
            else "missing",
            "regional_export": "passed"
            if e03_export
            and e03_export.get("same_seed_npz_sha256_equal")
            and e03_export.get("different_seed_mouth_max_abs") == 0
            else "missing",
            "a0_unreal": "blocked_missing_ue_project",
        },
        "recommended_models": {
            "offline_deterministic": "runs/e01_ravdess_v2/best_checkpoint.pt",
            "streaming": "runs/e01_ravdess_v2/best_checkpoint.pt",
            "offline_optional_style": e03.get("selected_checkpoint") if e03 else None,
            "audio_motion_jepa": "not_promoted_no_clear_pure_audio_win",
        },
        "environment": {
            "cuda_available": environment.get("cuda_available") if environment else None,
            "gpu": environment.get("gpu") if environment else None,
            "torch": environment.get("torch") if environment else None,
        },
        "external_requirements": [
            "Run the 200-clip E00 human audit and refine a 2-3 hour Gold anchor",
            "Accept VOCASET or MMHead access for a true 3D Gold motion anchor",
            "Provide two annotators and one adjudicator for E00",
            "Install/provide UE 5.6 commandlet project and four A0 character assets",
            "Provide separately consented commercial recordings for the product track",
        ],
        "catalog": catalog,
        "ravdess": ravdess,
        "evidence": {
            "real_d0p": real_d0p,
            "e01_metrics": e01,
            "e01_evaluation": e01_eval,
            "e10_metrics": e10,
            "e11_audio_only_metrics": e11,
            "e03_three_seed_summary": e03,
            "e03_export_invariants": e03_export,
            "unitalker_candidate": unitalker,
            "inference": inference,
            "streaming": streaming,
            "interpolation": interpolation,
        },
    }
