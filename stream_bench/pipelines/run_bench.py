import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import yaml
import wandb
from itertools import islice
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

from stream_bench.benchmarks import Bench, load_benchmark
from stream_bench.agents import load_agent
from .utils import merge_dicts

def setup_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--agent_cfg",
        type=Path,
        required=True,
        help="Path to the agent's config yaml file"
    )
    parser.add_argument(
        "--bench_cfg",
        type=Path,
        required=True,
        help="Path to the benchmark's config yaml file."
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Whether to use wandb for experiment tracking."
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Project name for wandb"
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="Team name for wandb"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="The name of the experiment for wandb."
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Run only the first N shuffled samples for a quick smoke test."
    )

    return parser.parse_args()


def limit_dataset(dataset, max_samples: int | None):
    if max_samples is None:
        return dataset
    if max_samples < 0:
        raise ValueError("--max_samples must be non-negative.")
    if hasattr(dataset, "select") and hasattr(dataset, "__len__"):
        return dataset.select(range(min(max_samples, len(dataset))))
    return list(islice(dataset, max_samples))


def inject_parse_context(agent, bench: Bench, row_input: dict) -> dict:
    if ("parse_template" in row_input) or ("label_set" in row_input):
        return row_input
    if "prompt_zeroshot" not in row_input:
        return row_input

    label_map = getattr(bench, "LABEL2TEXT", None)
    if not isinstance(label_map, dict) or len(label_map) == 0:
        return row_input

    options = [f"{idx}. {text}" for idx, text in label_map.items()]
    row_input["label_set"] = set(options)
    row_input["parse_template"] = (
        "Original task:\n"
        f"{row_input['prompt_zeroshot']}\n\n"
        "Model output: {model_output}\n\n"
        "If the model output is incomplete, malformed, or unrelated, solve the original task again and return the single best option.\n"
        "Convert the model output into one of the following options (one option per line):\n"
        f"{agent.get_options_text(set(options))}\n\n"
        "Answer (please only answer with a single option):"
    )
    return row_input


def main():
    args = setup_args()
    agent_cfg = yaml.safe_load(args.agent_cfg.read_text())
    bench_cfg = yaml.safe_load(args.bench_cfg.read_text())
    assert 'bench_name' in bench_cfg
    agent_cfg["bench_name"] = bench_cfg["bench_name"]
    agent_cfg["split"] = bench_cfg["split"]
    if args.name is not None:
        agent_cfg['exp_name'] = args.name
    print('init agent')
    agent = load_agent(agent_cfg['agent_name'])(agent_cfg)
    bench_cfg['agent'] = agent
    # bench_cfg['agent_callback'] = agent.retrieve_experience
    print('init bench environment')
    bench: Bench = load_benchmark(bench_cfg['bench_name'])(**bench_cfg)
    agent.bench = bench
    dataset = limit_dataset(bench.get_dataset(), args.max_samples)
    if args.max_samples is not None:
        print(f"running a limited smoke test on {len(dataset)} sample(s)")

    if args.use_wandb:
        wandb.init(
            project=args.project if (args.project is not None) else f"streambench-{bench_cfg['bench_name']}",
            entity=args.entity,
            name=args.name if (args.name is not None) else agent.get_name(),
            config=merge_dicts(dicts=[agent_cfg, bench_cfg])  # NOTE: agent configurations and benchmark configurations
        )

    for time_step, row in enumerate(tqdm(dataset, dynamic_ncols=True)):
        try:
            row['time_step'] = time_step
            x = bench.get_input(row)  # remove ground truth related information
            x['time_step'] = time_step
            x = inject_parse_context(agent, bench, x)
            model_output = agent(**x)
            prediction = bench.postprocess_generation(model_output, time_step)
            label = bench.get_output(row)
            pred_res = bench.process_results(
                prediction,
                label,
                return_details=True,
                time_step=time_step
            )

            has_feedback, feedback = bench.give_feedback(model_output, row, pred_res)
            feedback['time_step'] = time_step
            feedback = { **row, **feedback }
            agent.update(has_feedback, **feedback)

            if args.use_wandb:
                log_data = merge_dicts(dicts=[agent.get_wandb_log_info(), pred_res])
                wandb.log(data=log_data)
            # NOTE: agent.log() should be called after wandb
            if isinstance(label, int):
                label = bench.LABEL2TEXT[label]
            elif isinstance(label, dict):
                label = label.get("label", json.dumps(label))

            agent.log(label_text=label)
        except (KeyError, IndexError) as e:
            print(e)

    metrics = bench.get_metrics()
    print(metrics)
    if args.use_wandb:
        wandb.log(data={'final/'+k: v for k, v in metrics.items()})

if __name__ == "__main__":
    main()
