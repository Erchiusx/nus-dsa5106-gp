import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stream_bench.agents import load_agent
from stream_bench.benchmarks import load_benchmark
from stream_bench.benchmarks.hotpotqa_distract import compute_exact_match, compute_f1
from stream_bench.pipelines.run_bench import inject_parse_context, limit_dataset


BENCH_ORDER = ["spider", "cosql", "bird", "ds_1000", "ddxplus", "hotpotqa"]
BENCH_CFGS = {
    "spider": Path("configs/bench/spider.yml"),
    "cosql": Path("configs/bench/cosql.yml"),
    "bird": Path("configs/bench/bird.yml"),
    "ds_1000": Path("configs/bench/ds_1000.yml"),
    "ddxplus": Path("configs/bench/ddxplus.yml"),
    "hotpotqa": Path("configs/bench/hotpotqa.yml"),
}
METRIC_KEYS = {
    "spider": ("EX",),
    "cosql": ("EX",),
    "bird": ("EX",),
    "ds_1000": ("pass@1",),
    "ddxplus": ("accuracy",),
    "hotpotqa": ("em", "f1"),
}


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent_cfg",
        type=Path,
        default=Path("configs/agent/zeroshot_gemini_3_tuned.yml"),
        help="Path to the tuned Gemini 3 agent config.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="gemini3_tuned_full",
        help="Resume handle. Reuse the same run_name to continue a paused run.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("resume_runs"),
        help="Directory for checkpoints and per-benchmark result files.",
    )
    parser.add_argument(
        "--bench",
        nargs="*",
        choices=BENCH_ORDER,
        help="Optional subset of benchmarks to run.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap for quick testing. Omit to run the full benchmark.",
    )
    return parser.parse_args()


def ensure_google_api_key() -> None:
    if os.getenv("GOOGLE_API_KEY"):
        return
    if os.name != "nt":
        return
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, "GOOGLE_API_KEY")
            if value:
                os.environ["GOOGLE_API_KEY"] = value
    except OSError:
        pass


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def to_jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return str(value)


def load_records(results_path: Path) -> list[dict]:
    if not results_path.exists():
        return []
    records = []
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda x: x["time_step"])
    deduped = {}
    for record in records:
        deduped[record["time_step"]] = record
    return [deduped[idx] for idx in sorted(deduped)]


def summarize_records(bench_name: str, records: list[dict]) -> dict:
    summary = {
        "processed": len(records),
        "next_index": len(records),
    }
    if bench_name == "hotpotqa":
        total = len(records)
        sum_em = sum(float(record["em"]) for record in records)
        sum_f1 = sum(float(record["f1"]) for record in records)
        summary["metrics"] = {
            "em": (sum_em / total) if total else 0.0,
            "f1": (sum_f1 / total) if total else 0.0,
        }
    else:
        total = len(records)
        n_correct = sum(int(record["correct"]) for record in records)
        metric_name = METRIC_KEYS[bench_name][0]
        summary["metrics"] = {
            metric_name: (n_correct / total) if total else 0.0,
        }
        summary["n_correct"] = n_correct
    return summary


def score_sample(bench_name: str, bench, prediction, label) -> dict:
    if bench_name == "hotpotqa":
        em = compute_exact_match(prediction, label)
        f1 = compute_f1(prediction, label)
        return {
            "correct": int(em),
            "em": float(em),
            "f1": float(f1),
        }
    details = bench.process_results(prediction, label, return_details=True)
    return {
        "correct": int(details["correct"]),
    }


def build_agent_and_bench(agent_cfg_path: Path, bench_cfg_path: Path, run_name: str):
    agent_cfg = yaml.safe_load(agent_cfg_path.read_text(encoding="utf-8"))
    bench_cfg = yaml.safe_load(bench_cfg_path.read_text(encoding="utf-8"))
    agent_cfg["bench_name"] = bench_cfg["bench_name"]
    agent_cfg["split"] = bench_cfg["split"]
    agent_cfg["exp_name"] = f"resume__{run_name}"
    agent = load_agent(agent_cfg["agent_name"])(agent_cfg)
    bench_cfg["agent"] = agent
    bench = load_benchmark(bench_cfg["bench_name"])(**bench_cfg)
    agent.bench = bench
    return agent, bench, bench_cfg


def build_manifest(args: argparse.Namespace, benches: list[str]) -> dict:
    return {
        "run_name": args.run_name,
        "agent_cfg": str(args.agent_cfg),
        "benches": benches,
        "max_samples": args.max_samples,
        "metric_keys": {bench: list(METRIC_KEYS[bench]) for bench in benches},
    }


def write_run_state(state_path: Path, manifest: dict, per_bench: dict, current_bench: str | None) -> None:
    payload = {
        "manifest": manifest,
        "current_bench": current_bench,
        "benchmarks": per_bench,
    }
    atomic_write_json(state_path, payload)


def main() -> int:
    args = setup_args()
    ensure_google_api_key()
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. On PowerShell, run: "
            "$env:GOOGLE_API_KEY = [Environment]::GetEnvironmentVariable('GOOGLE_API_KEY','User')"
        )

    benches = args.bench if args.bench else BENCH_ORDER
    run_dir = args.output_dir / args.run_name
    results_dir = run_dir / "results"
    state_path = run_dir / "state.json"
    summary_path = run_dir / "summary.json"
    manifest = build_manifest(args, benches)
    atomic_write_json(run_dir / "manifest.json", manifest)

    per_bench = {}

    try:
        for bench_name in benches:
            bench_cfg_path = BENCH_CFGS[bench_name]
            results_path = results_dir / f"{bench_name}.jsonl"
            existing_records = load_records(results_path)
            existing_summary = summarize_records(bench_name, existing_records)

            agent, bench, _ = build_agent_and_bench(args.agent_cfg, bench_cfg_path, args.run_name)
            dataset = limit_dataset(bench.get_dataset(), args.max_samples)
            total = len(dataset)

            bench_state = {
                "bench_cfg": str(bench_cfg_path),
                "processed": existing_summary["processed"],
                "next_index": min(existing_summary["next_index"], total),
                "total": total,
                "status": "completed" if existing_summary["processed"] >= total else "running",
                "metrics": existing_summary["metrics"],
            }
            if "n_correct" in existing_summary:
                bench_state["n_correct"] = existing_summary["n_correct"]
            per_bench[bench_name] = bench_state
            write_run_state(state_path, manifest, per_bench, current_bench=bench_name)

            if bench_state["processed"] >= total:
                print(f"[skip] {bench_name}: already completed ({total}/{total})")
                continue

            print(f"[run] {bench_name}: resuming from {bench_state['next_index']} / {total}")
            progress = tqdm(
                range(bench_state["next_index"], total),
                initial=bench_state["next_index"],
                total=total,
                dynamic_ncols=True,
                desc=bench_name,
            )

            results_path.parent.mkdir(parents=True, exist_ok=True)
            with results_path.open("a", encoding="utf-8") as handle:
                for time_step in progress:
                    row = dataset[time_step]
                    row["time_step"] = time_step
                    row_input = bench.get_input(row)
                    row_input["time_step"] = time_step
                    row_input = inject_parse_context(agent, bench, row_input)

                    model_output = agent(**row_input)
                    prediction = bench.postprocess_generation(model_output, time_step)
                    label = bench.get_output(row)
                    sample_metrics = score_sample(bench_name, bench, prediction, label)

                    record = {
                        "time_step": time_step,
                        "model_output": model_output,
                        "prediction": to_jsonable(prediction),
                        "label": to_jsonable(label),
                        **sample_metrics,
                        **{k: to_jsonable(v) for k, v in agent.log_info.items()},
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()

                    bench_state["processed"] = time_step + 1
                    bench_state["next_index"] = time_step + 1
                    if bench_name == "hotpotqa":
                        total_done = bench_state["processed"]
                        prev_em_sum = bench_state["metrics"]["em"] * (total_done - 1)
                        prev_f1_sum = bench_state["metrics"]["f1"] * (total_done - 1)
                        bench_state["metrics"]["em"] = (prev_em_sum + sample_metrics["em"]) / total_done
                        bench_state["metrics"]["f1"] = (prev_f1_sum + sample_metrics["f1"]) / total_done
                    else:
                        bench_state["n_correct"] = bench_state.get("n_correct", 0) + sample_metrics["correct"]
                        metric_name = METRIC_KEYS[bench_name][0]
                        bench_state["metrics"][metric_name] = bench_state["n_correct"] / bench_state["processed"]

                    progress.set_postfix(bench_state["metrics"])
                    write_run_state(state_path, manifest, per_bench, current_bench=bench_name)
                    atomic_write_json(summary_path, {"benchmarks": per_bench})

            bench_state["status"] = "completed"
            write_run_state(state_path, manifest, per_bench, current_bench=None)
            atomic_write_json(summary_path, {"benchmarks": per_bench})
            print(f"[done] {bench_name}: {bench_state['metrics']}")
    except KeyboardInterrupt:
        print("\nPaused. Re-run the same command to resume from the last completed sample.")
        write_run_state(state_path, manifest, per_bench, current_bench=None)
        atomic_write_json(summary_path, {"benchmarks": per_bench})
        return 130

    print("\nAll requested benchmarks completed.")
    print(json.dumps({"benchmarks": per_bench}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
