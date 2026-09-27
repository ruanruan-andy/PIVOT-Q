from __future__ import annotations

import dataclasses
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tqdm
import tyro
from libero.libero import benchmark, get_libero_path

from examples.Libero.eval.run_libero_eval import GR00TPolicy
from examples.Libero.eval.utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    save_rollout_video,
)
from examples.LiberoPlus.task_manifest import select_manifest_task_ids
from scripts.eval_run_metadata import write_eval_run_metadata


SUPPORTED_SUITES = ("libero_spatial", "libero_goal", "libero_object", "libero_10")
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


@dataclass
class LiberoPlusEvalConfig:
    task_suite_name: str = "libero_spatial"
    port: int = 5556
    headless: bool = False
    num_steps_wait: int = 10
    num_trials_per_task: int = 1
    task_ids: list[int] | None = None
    category: str | None = None
    difficulty: int | None = None
    max_tasks: int | None = None
    sample_seed: int = 0
    save_video: bool = True
    resume: bool = False
    model_variant: str = "groot-fp16"
    sample_manifest: str | None = None
    policy_seed: int = 0


def _default_output_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "output" / "libero-plus"


def _load_classification(suite_name: str) -> tuple[list[dict[str, Any]], Path]:
    classification_path = (
        Path(get_libero_path("benchmark_root")) / "benchmark" / "task_classification.json"
    )
    with classification_path.open("r", encoding="utf-8") as stream:
        all_classifications = json.load(stream)
    if suite_name not in all_classifications:
        raise KeyError(f"Suite {suite_name!r} is missing from {classification_path}")
    return all_classifications[suite_name], classification_path


def _select_tasks(
    cfg: LiberoPlusEvalConfig,
    task_names: list[str],
    metadata_by_name: dict[str, dict[str, Any]],
) -> list[int]:
    if cfg.task_ids is None:
        selected = list(range(len(task_names)))
    else:
        invalid = [task_id for task_id in cfg.task_ids if not 0 <= task_id < len(task_names)]
        if invalid:
            raise ValueError(f"Invalid zero-based task IDs: {invalid}; suite size is {len(task_names)}")
        selected = list(dict.fromkeys(cfg.task_ids))

    if cfg.sample_manifest is not None:
        manifest_path = Path(cfg.sample_manifest).expanduser().resolve()
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        selected = select_manifest_task_ids(
            manifest,
            suite_name=cfg.task_suite_name,
            task_names=task_names,
            metadata_by_name=metadata_by_name,
            candidate_task_ids=selected,
        )

    if cfg.category is not None:
        requested = cfg.category.casefold()
        available = sorted(
            {str(item["category"]) for item in metadata_by_name.values()}, key=str.casefold
        )
        if requested not in {name.casefold() for name in available}:
            raise ValueError(f"Unknown category {cfg.category!r}. Available categories: {available}")
        selected = [
            task_id
            for task_id in selected
            if metadata_by_name[task_names[task_id]]["category"].casefold() == requested
        ]

    if cfg.difficulty is not None:
        selected = [
            task_id
            for task_id in selected
            if metadata_by_name[task_names[task_id]].get("difficulty_level") is not None
            and int(metadata_by_name[task_names[task_id]]["difficulty_level"])
            == cfg.difficulty
        ]

    if cfg.max_tasks is not None:
        if cfg.max_tasks <= 0:
            raise ValueError("max_tasks must be positive")
        if len(selected) > cfg.max_tasks:
            selected = sorted(random.Random(cfg.sample_seed).sample(selected, cfg.max_tasks))

    if not selected:
        raise ValueError("No tasks matched the requested filters")
    return selected


def _group_summary(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        value = record[field]
        key = "Unspecified" if value is None else str(value)
        groups.setdefault(key, []).append(record)
    result: dict[str, Any] = {}
    for key, items in sorted(groups.items()):
        successes = sum(bool(item["success"]) for item in items)
        result[key] = {
            "episodes": len(items),
            "successes": successes,
            "success_rate": successes / len(items),
            "errors": sum(item["error"] is not None for item in items),
        }
    return result


def _write_summary(
    path: Path,
    cfg: LiberoPlusEvalConfig,
    selected_task_count: int,
    records: list[dict[str, Any]],
) -> None:
    successes = sum(bool(record["success"]) for record in records)
    summary = {
        "model_variant": cfg.model_variant,
        "suite": cfg.task_suite_name,
        "selection": dataclasses.asdict(cfg),
        "selected_tasks": selected_task_count,
        "completed_episodes": len(records),
        "successes": successes,
        "success_rate": successes / len(records) if records else 0.0,
        "errors": sum(record["error"] is not None for record in records),
        "by_category": _group_summary(records, "category") if records else {},
        "by_difficulty": _group_summary(records, "difficulty_level") if records else {},
    }
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary_path.replace(path)


def _load_episode_records(
    path: Path, suite_name: str, model_variant: str
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Ignoring incomplete JSONL line {line_number} in {path}")
                continue
            if record.get("suite") != suite_name:
                raise ValueError(
                    f"Resume record suite {record.get('suite')!r} does not match {suite_name!r}"
                )
            if record.get("model_variant", model_variant) != model_variant:
                raise ValueError(
                    f"Resume record model {record.get('model_variant')!r} does not match "
                    f"{model_variant!r}"
                )
            records.append(record)
    return records


def evaluate(cfg: LiberoPlusEvalConfig) -> None:
    if cfg.task_suite_name not in SUPPORTED_SUITES:
        raise ValueError(
            f"Unsupported suite {cfg.task_suite_name!r}; available suites: {SUPPORTED_SUITES}"
        )
    if cfg.num_trials_per_task != 1:
        raise ValueError("LIBERO-Plus evaluation requires num_trials_per_task=1")

    output_dir = Path(os.environ.get("LIBERO_EVAL_LOG_DIR", _default_output_dir()))
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(os.environ.get("LIBERO_EVAL_METRICS_DIR", output_dir / "metrics"))
    logs_dir = Path(os.environ.get("LIBERO_EVAL_LOGS_DIR", output_dir / "logs"))
    video_dir = Path(os.environ.get("LIBERO_EVAL_VIDEO_DIR", output_dir / "videos"))
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save_video:
        video_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_EVAL_VIDEO_DIR"] = str(video_dir)
    write_eval_run_metadata(output_dir, cfg)
    text_log_path = logs_dir / "evaluator.log"
    episodes_path = metrics_dir / "episodes.jsonl"
    summary_path = metrics_dir / "summary.json"

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    task_names = task_suite.get_task_names()
    classifications, classification_path = _load_classification(cfg.task_suite_name)
    metadata_by_name = {str(item["name"]): item for item in classifications}
    missing_metadata = [name for name in task_names if name not in metadata_by_name]
    if missing_metadata:
        raise RuntimeError(
            f"Classification metadata is missing for {len(missing_metadata)} tasks; "
            f"first missing task: {missing_metadata[0]}"
        )
    selected_task_ids = _select_tasks(cfg, task_names, metadata_by_name)

    existing_records = (
        _load_episode_records(episodes_path, cfg.task_suite_name, cfg.model_variant)
        if cfg.resume
        else []
    )
    selected_task_id_set = set(selected_task_ids)
    records_by_task = {
        int(record["task_index"]): record
        for record in existing_records
        if int(record["task_index"]) in selected_task_id_set
    }
    completed_task_ids = {
        task_id
        for task_id, record in records_by_task.items()
        if record.get("error") is None
    }
    policy = GR00TPolicy(host="localhost", port=cfg.port, headless=cfg.headless)

    file_mode = "a" if cfg.resume else "w"
    with text_log_path.open(file_mode, encoding="utf-8", buffering=1) as text_log, episodes_path.open(
        file_mode, encoding="utf-8", buffering=1
    ) as episodes_log:

        def report(message: str) -> None:
            print(message, flush=True)
            text_log.write(message + "\n")

        report(f"Suite: {cfg.task_suite_name}")
        report(f"Model: {cfg.model_variant}")
        report(f"LIBERO package: {Path(benchmark.__file__).resolve()}")
        report(f"Classification: {classification_path}")
        report(f"Selected tasks: {len(selected_task_ids)} / {len(task_names)}")
        report(f"Sample manifest: {cfg.sample_manifest}")
        report(f"Resume: {cfg.resume}; restored completed rollouts: {len(completed_task_ids)}")
        report(f"Output: {output_dir}")

        for task_id in tqdm.tqdm(selected_task_ids, desc=cfg.task_suite_name):
            if task_id in completed_task_ids:
                report(f"Skipping completed task {task_id}")
                continue
            task_name = task_names[task_id]
            metadata = metadata_by_name[task_name]
            task = task_suite.get_task(task_id)
            env = None
            started_at = time.time()
            error: str | None = None
            success = False
            steps = 0
            video_path: str | None = None
            top_view: list[Any] = []
            wrist_view: list[Any] = []
            task_description = task.language

            try:
                initial_states = task_suite.get_task_init_states(task_id)
                if len(initial_states) < 1:
                    raise RuntimeError("Task has no initial state")
                max_steps = MAX_STEPS[cfg.task_suite_name]
                env, task_description = get_libero_env(
                    task,
                    resolution=256,
                    horizon=max(1000, max_steps + cfg.num_steps_wait),
                )
                # LIBERO-Plus encodes camera / robot perturbation parameters in
                # synthetic task filenames. ControlEnv resolves those parameters
                # and exposes the clean (or intentionally rewritten) BDDL language.
                task_description = str(
                    getattr(env, "language_instruction", task_description)
                )
                obs = env.reset()
                obs = env.set_init_state(initial_states[0])
                done = False
                while steps < max_steps + cfg.num_steps_wait:
                    if steps < cfg.num_steps_wait:
                        obs, _, done, _ = env.step(get_libero_dummy_action())
                    else:
                        image, wrist_image = get_libero_image(obs)
                        top_view.append(image)
                        wrist_view.append(wrist_image)
                        step_seed = (
                            cfg.policy_seed
                            + task_id * 1_000_000
                            + (steps - cfg.num_steps_wait)
                        )
                        action = policy.get_action(
                            obs, task_description, policy_seed=step_seed
                        )
                        obs, _, done, _ = env.step(action.tolist())
                    steps += 1
                    if done:
                        success = True
                        break
            except Exception as exc:  # Keep a long benchmark running and record infrastructure failures.
                error = f"{type(exc).__name__}: {exc}"
            finally:
                if env is not None:
                    env.close()

            if cfg.save_video and top_view and wrist_view:
                try:
                    video_path = save_rollout_video(
                        top_view,
                        wrist_view,
                        len(records_by_task) + (task_id not in records_by_task),
                        success=success,
                        task_description=task_description,
                        log_file=text_log,
                    )
                except Exception as exc:
                    video_error = f"{type(exc).__name__}: {exc}"
                    error = f"{error}; video: {video_error}" if error else f"video: {video_error}"

            raw_difficulty = metadata.get("difficulty_level")
            difficulty_level = int(raw_difficulty) if raw_difficulty is not None else None
            record = {
                "model_variant": cfg.model_variant,
                "suite": cfg.task_suite_name,
                "task_index": task_id,
                "classification_id": int(metadata["id"]),
                "task_name": task_name,
                "task_description": task_description,
                "category": metadata["category"],
                "difficulty_level": difficulty_level,
                "success": success,
                "steps": steps,
                "duration_seconds": round(time.time() - started_at, 3),
                "error": error,
                "video_path": video_path,
            }
            episodes_log.write(json.dumps(record, ensure_ascii=False) + "\n")
            records_by_task[task_id] = record
            if error is None:
                completed_task_ids.add(task_id)
            effective_records = [
                records_by_task[selected_task_id]
                for selected_task_id in selected_task_ids
                if selected_task_id in records_by_task
            ]
            _write_summary(summary_path, cfg, len(selected_task_ids), effective_records)
            report(
                f"[{len(effective_records)}/{len(selected_task_ids)}] task={task_id} "
                f"category={record['category']} difficulty={record['difficulty_level']} "
                f"success={success} steps={steps} error={error}"
            )

        report(f"Completed {len(records_by_task)} episodes; summary: {summary_path}")


if __name__ == "__main__":
    evaluate(tyro.cli(LiberoPlusEvalConfig))
