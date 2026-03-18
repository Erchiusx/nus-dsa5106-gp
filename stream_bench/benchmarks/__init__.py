from importlib import import_module

from .base import Bench

_TASK_SPECS = {
    "ddxplus": ("stream_bench.benchmarks.ddxplus", "create_ddxplus", True),
    "ds_1000": ("stream_bench.benchmarks.ds_1000", "DS1000", False),
    "hotpotqa_distract": ("stream_bench.benchmarks.hotpotqa_distract", "HotpotQADistract", False),
    "toolbench": ("stream_bench.benchmarks.toolbench", "ToolBench", False),
    "gsm8k": ("stream_bench.benchmarks.gsm8k", "GSM8KBench", False),
    "math": ("stream_bench.benchmarks.math", "MATHBench", False),
    "bird": ("stream_bench.benchmarks.text_to_sql", "create_bird", True),
    "cosql": ("stream_bench.benchmarks.text_to_sql", "create_cosql", True),
    "spider": ("stream_bench.benchmarks.text_to_sql", "create_spider", True),
}


def load_benchmark(benchmark_name: str) -> type[Bench]:
    if benchmark_name not in _TASK_SPECS:
        raise ValueError("Benchmark %s not found" % benchmark_name)

    module_name, attr_name, is_factory = _TASK_SPECS[benchmark_name]
    task = getattr(import_module(module_name), attr_name)
    return task() if is_factory else task
