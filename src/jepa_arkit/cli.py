from __future__ import annotations

import argparse
import json

from jepa_arkit.data.audit import audit_release
from jepa_arkit.data.catalog import audit_dataset_catalog
from jepa_arkit.demo import create_demo_dataset
from jepa_arkit.io import dump_json


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True))


def main() -> int:
    parser = argparse.ArgumentParser(prog="jepa-arkit")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("init-demo")
    demo.add_argument("--output", required=True)
    demo.add_argument("--schema", default="configs/contracts/canonical_arkit_v1.json")
    audit = commands.add_parser("audit-release")
    audit.add_argument("--config", required=True)
    audit.add_argument("--output", default="artifacts/audit_report.json")
    catalog = commands.add_parser("audit-catalog")
    catalog.add_argument("--config", default="configs/data/dataset_catalog.yaml")
    catalog.add_argument("--output", default="artifacts/dataset_catalog_audit.json")
    ravdess = commands.add_parser("ingest-ravdess")
    ravdess.add_argument("--archives", required=True)
    ravdess.add_argument("--output", required=True)
    prepare = commands.add_parser("prepare-ravdess")
    prepare.add_argument("--release", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--model", default="data/models/face_landmarker.task")
    prepare.add_argument("--schema", default="configs/contracts/canonical_arkit_v1.json")
    prepare.add_argument(
        "--missing-curve-policy",
        default="configs/contracts/mediapipe_missing_curve_policy_v1.json",
    )
    prepare.add_argument("--workers", type=int, default=4)
    prepare.add_argument("--minimum-valid-fraction", type=float, default=0.8)
    prepare.add_argument("--limit", type=int)
    build_release = commands.add_parser("build-ravdess-release")
    build_release.add_argument("--raw-release", required=True)
    build_release.add_argument("--prepared", required=True)
    build_release.add_argument("--output", required=True)
    build_release.add_argument("--schema", default="configs/contracts/canonical_arkit_v1.json")
    audit_unitalker = commands.add_parser("audit-unitalker-candidate")
    audit_unitalker.add_argument("--archive", required=True)
    audit_unitalker.add_argument("--output", required=True)
    extract_features = commands.add_parser("extract-wavlm")
    extract_features.add_argument("--manifest", required=True)
    extract_features.add_argument("--output", required=True)
    extract_features.add_argument("--model-id", default="microsoft/wavlm-base")
    extract_features.add_argument("--revision", required=True)
    extract_features.add_argument("--source-release-id", required=True)
    extract_features.add_argument("--feature-release-id", default="wavlm_base_fp16_ravdess_v2")
    extract_features.add_argument("--batch-size", type=int, default=8)
    extract_features.add_argument("--device", default="auto")
    evaluate = commands.add_parser("evaluate-direct")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    infer = commands.add_parser("infer-direct")
    infer.add_argument("--config", required=True)
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--audio", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--character-profile", default="canonical_arkit_v1")
    infer.add_argument("--ue-engine", default="UE5.6.x")
    infer_regional_parser = commands.add_parser("infer-regional")
    infer_regional_parser.add_argument("--config", required=True)
    infer_regional_parser.add_argument("--checkpoint", required=True)
    infer_regional_parser.add_argument("--audio", required=True)
    infer_regional_parser.add_argument("--output", required=True)
    infer_regional_parser.add_argument("--sampling-seed", type=int, default=0)
    infer_regional_parser.add_argument("--regional-temperature", type=float, default=0.75)
    infer_regional_parser.add_argument("--head-temperature", type=float, default=0.5)
    infer_regional_parser.add_argument("--character-profile", default="canonical_arkit_v1")
    infer_regional_parser.add_argument("--ue-engine", default="UE5.6.x")
    infer_stream = commands.add_parser("infer-streaming-direct")
    infer_stream.add_argument("--config", required=True)
    infer_stream.add_argument("--checkpoint", required=True)
    infer_stream.add_argument("--audio", required=True)
    infer_stream.add_argument("--protocol", required=True)
    infer_stream.add_argument("--output", required=True)
    infer_stream.add_argument("--repeat-seconds", type=float)
    infer_stream.add_argument("--character-profile", default="canonical_arkit_v1")
    infer_stream.add_argument("--ue-engine", default="UE5.6.x")
    interpolate = commands.add_parser("interpolate-motion")
    interpolate.add_argument("--input", required=True)
    interpolate.add_argument("--output", required=True)
    interpolate.add_argument("--report", required=True)
    interpolate.add_argument("--schema", default="configs/contracts/canonical_arkit_v1.json")
    interpolate.add_argument("--target-fps", type=int, default=60)
    direct = commands.add_parser("smoke-direct")
    direct.add_argument("--config", required=True)
    jepa = commands.add_parser("smoke-jepa")
    jepa.add_argument("--config", required=True)
    train = commands.add_parser("train-direct")
    train.add_argument("--config", required=True)
    train_jepa = commands.add_parser("train-motion-jepa")
    train_jepa.add_argument("--config", required=True)
    train_audio_jepa = commands.add_parser("train-audio-motion-jepa")
    train_audio_jepa.add_argument("--config", required=True)
    train_regional = commands.add_parser("train-regional")
    train_regional.add_argument("--config", required=True)
    evaluate_regional_parser = commands.add_parser("evaluate-regional")
    evaluate_regional_parser.add_argument("--config", required=True)
    evaluate_regional_parser.add_argument("--checkpoint", required=True)
    evaluate_regional_parser.add_argument(
        "--split", default="validation", choices=("train", "validation", "test")
    )
    summarize_regional_parser = commands.add_parser("summarize-regional")
    summarize_regional_parser.add_argument("--baseline-evaluation", required=True)
    summarize_regional_parser.add_argument("--run", action="append", required=True)
    summarize_regional_parser.add_argument("--output", required=True)
    eval_audio_jepa = commands.add_parser("evaluate-audio-motion-jepa")
    eval_audio_jepa.add_argument("--config", required=True)
    eval_audio_jepa.add_argument("--checkpoint", required=True)
    eval_audio_jepa.add_argument(
        "--split", default="validation", choices=("train", "validation", "test")
    )
    solve = commands.add_parser("solve-video")
    solve.add_argument("--video", required=True)
    solve.add_argument("--model", default="data/models/face_landmarker.task")
    solve.add_argument("--schema", default="configs/contracts/canonical_arkit_v1.json")
    solve.add_argument("--missing-curve-policy")
    solve.add_argument("--output", required=True)
    environment = commands.add_parser("environment-report")
    environment.add_argument("--output", default="artifacts/environment.json")
    status = commands.add_parser("status")
    status.add_argument("--output", default="artifacts/project_status.json")
    arguments = parser.parse_args()
    if arguments.command == "init-demo":
        _print(create_demo_dataset(arguments.output, arguments.schema))
        return 0
    if arguments.command == "audit-release":
        report = audit_release(arguments.config)
        dump_json(arguments.output, report.to_dict())
        _print(report.to_dict())
        return 0 if report.passed else 2
    if arguments.command == "audit-catalog":
        report = audit_dataset_catalog(arguments.config)
        dump_json(arguments.output, report)
        _print(report)
        return 0 if report["d0a_ready"] else 2
    if arguments.command == "ingest-ravdess":
        from jepa_arkit.data.ravdess import ingest_ravdess

        checksums = {
            f"Video_Speech_Actor_{actor:02d}.zip": checksum
            for actor, checksum in enumerate(
                [
                    "3c8ececaf392b4a9b11b32271f4f6d01",
                    "a6f40d413b2e6ef25b3a595099e59abb",
                    "68fa240ddd8a3cc410c64efe5fb2b0e4",
                    "013eed832af1f7e97082aae398a00dbe",
                    "b00e17d61374ef6ba86fef35d50a20fb",
                    "3c42877921cc08cfb5c841a0f2cb94a7",
                    "bdd29bfe082f80361ec6b589845c283d",
                    "bed5f43d9c18e177e34d4bc1ed9b6d77",
                    "775da349c50bf915a1bbcb37379bd092",
                    "70199d1c6902f76e17df308d5c41fa01",
                    "824504aa41a8fd575e5459d5701dd378",
                    "2a1f0ddc0ca207beedee6ce2ea863ad7",
                    "c5f2fd4fd77941636947620368c612b5",
                    "466d31f4cce92a14b1f4ee884bbdf7d7",
                    "4b3b9bc8473e86f884630f2506684887",
                    "a63c54570ebbe6bd616d0dced5630f7e",
                    "8b7ae3e5a85d2874ac8e1195e722e047",
                    "fef2131ac361da182c0cb85f3d8e8a6c",
                    "abb97d97d8c4e88c866cee1cea9d06d6",
                    "5ddfb22770093bafebdefe8faa449f48",
                    "d810cf2ff6863ef91306fc7a16d74fbb",
                    "fee55f72c9871401f23d4b167de3217d",
                    "b31e8c8904e66102e7a951694db752d6",
                    "eef90e806c9179bc5b4b098d03647ae3",
                ],
                start=1,
            )
        }
        _print(ingest_ravdess(arguments.archives, arguments.output, checksums))
        return 0
    if arguments.command == "prepare-ravdess":
        from jepa_arkit.data.ravdess_pipeline import prepare_ravdess

        report = prepare_ravdess(
            arguments.release,
            arguments.output,
            arguments.model,
            arguments.schema,
            arguments.missing_curve_policy,
            workers=arguments.workers,
            minimum_valid_fraction=arguments.minimum_valid_fraction,
            limit=arguments.limit,
        )
        _print(report)
        return 0 if report["status"] == "passed" else 2
    if arguments.command == "build-ravdess-release":
        from jepa_arkit.data.ravdess_pipeline import build_ravdess_pilot_release

        _print(
            build_ravdess_pilot_release(
                arguments.raw_release,
                arguments.prepared,
                arguments.output,
                arguments.schema,
            )
        )
        return 0
    if arguments.command == "audit-unitalker-candidate":
        from jepa_arkit.data.unitalker import audit_unitalker_candidate

        report = audit_unitalker_candidate(arguments.archive, arguments.output)
        _print(report)
        return 0 if report["zip_crc_passed"] else 2
    if arguments.command == "extract-wavlm":
        from jepa_arkit.features.wavlm import extract_wavlm_features

        report = extract_wavlm_features(
            arguments.manifest,
            arguments.output,
            model_id=arguments.model_id,
            revision=arguments.revision,
            source_data_release_id=arguments.source_release_id,
            feature_release_id=arguments.feature_release_id,
            batch_size=arguments.batch_size,
            device_name=arguments.device,
        )
        _print(report)
        return 0 if report["status"] == "passed" else 2
    if arguments.command == "evaluate-direct":
        from jepa_arkit.evaluation import evaluate_direct

        _print(evaluate_direct(arguments.config, arguments.checkpoint, split=arguments.split))
        return 0
    if arguments.command == "infer-direct":
        from jepa_arkit.inference import infer_direct

        _print(
            infer_direct(
                arguments.config,
                arguments.checkpoint,
                arguments.audio,
                arguments.output,
                character_profile_id=arguments.character_profile,
                ue_engine=arguments.ue_engine,
            )
        )
        return 0
    if arguments.command == "infer-regional":
        from jepa_arkit.inference import infer_regional

        _print(
            infer_regional(
                arguments.config,
                arguments.checkpoint,
                arguments.audio,
                arguments.output,
                sampling_seed=arguments.sampling_seed,
                regional_temperature=arguments.regional_temperature,
                head_temperature=arguments.head_temperature,
                character_profile_id=arguments.character_profile,
                ue_engine=arguments.ue_engine,
            )
        )
        return 0
    if arguments.command == "infer-streaming-direct":
        from jepa_arkit.inference import infer_streaming_direct

        _print(
            infer_streaming_direct(
                arguments.config,
                arguments.checkpoint,
                arguments.audio,
                arguments.protocol,
                arguments.output,
                repeat_seconds=arguments.repeat_seconds,
                character_profile_id=arguments.character_profile,
                ue_engine=arguments.ue_engine,
            )
        )
        return 0
    if arguments.command == "interpolate-motion":
        from jepa_arkit.interpolation import resample_canonical_npz

        report = resample_canonical_npz(
            arguments.input,
            arguments.output,
            arguments.report,
            schema_path=arguments.schema,
            target_fps=arguments.target_fps,
        )
        _print(report)
        return 0 if report["status"] == "passed" else 2
    if arguments.command == "smoke-direct":
        from jepa_arkit.training.smoke import run_direct_smoke

        report = run_direct_smoke(arguments.config)
        _print(report)
        return 0 if report["passed"] else 2
    if arguments.command == "smoke-jepa":
        from jepa_arkit.training.smoke import run_jepa_smoke

        report = run_jepa_smoke(arguments.config)
        _print(report)
        return 0 if report["passed"] else 2
    if arguments.command == "train-direct":
        from jepa_arkit.training.formal import train_direct

        _print(train_direct(arguments.config))
        return 0
    if arguments.command == "train-motion-jepa":
        from jepa_arkit.training.jepa_formal import train_motion_jepa

        _print(train_motion_jepa(arguments.config))
        return 0
    if arguments.command == "train-audio-motion-jepa":
        from jepa_arkit.training.audio_jepa_formal import train_audio_motion_jepa

        _print(train_audio_motion_jepa(arguments.config))
        return 0
    if arguments.command == "train-regional":
        from jepa_arkit.training.regional_formal import train_regional

        _print(train_regional(arguments.config))
        return 0
    if arguments.command == "evaluate-regional":
        from jepa_arkit.training.regional_formal import evaluate_regional

        _print(evaluate_regional(arguments.config, arguments.checkpoint, split=arguments.split))
        return 0
    if arguments.command == "summarize-regional":
        from jepa_arkit.training.regional_formal import summarize_regional_runs

        _print(
            summarize_regional_runs(
                arguments.baseline_evaluation,
                arguments.run,
                arguments.output,
            )
        )
        return 0
    if arguments.command == "evaluate-audio-motion-jepa":
        from jepa_arkit.training.audio_jepa_formal import evaluate_audio_motion_jepa

        _print(
            evaluate_audio_motion_jepa(
                arguments.config,
                arguments.checkpoint,
                split=arguments.split,
            )
        )
        return 0
    if arguments.command == "solve-video":
        from jepa_arkit.contracts.canonical import CanonicalSchema
        from jepa_arkit.solver import MediaPipeFaceSolver

        schema = CanonicalSchema.from_file(arguments.schema)
        solver = MediaPipeFaceSolver(model_asset=arguments.model, schema=schema)
        motion = solver.solve_video(arguments.video)
        valid_fraction = float((motion.frame_confidence > 0).mean())
        report = {
            "frames": len(motion.timestamps),
            "valid_fraction": valid_fraction,
            "missing_curves": list(motion.missing_curves),
        }
        _print(report)
        if valid_fraction == 0:
            return 2
        solver.save(
            motion,
            arguments.output,
            missing_curve_policy=arguments.missing_curve_policy,
        )
        return 0
    if arguments.command == "environment-report":
        from jepa_arkit.training.environment import write_environment_report

        _print(write_environment_report(arguments.output))
        return 0
    if arguments.command == "status":
        from jepa_arkit.status import project_status

        report = project_status()
        dump_json(arguments.output, report)
        _print(report)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
