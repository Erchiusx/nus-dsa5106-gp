from importlib import import_module

from .base import Agent

_TASK_SPECS = {
    "zeroshot": ("stream_bench.agents.zeroshot", "ZeroShotAgent"),
    "fewshot": ("stream_bench.agents.fewshot", "FewShotAgent"),
    "cot": ("stream_bench.agents.cot", "CoTAgent"),
    "self_refine": ("stream_bench.agents.iter_prompt", "IterPromptAgent"),
    "grow_prompt": ("stream_bench.agents.scratchpad", "ScratchPadAgent"),
    "mem_prompt": ("stream_bench.agents.fewshot_rag", "FewShotRAGAgent"),
    "self_stream_icl": ("stream_bench.agents.fewshot_rag", "FewShotRAGAgent"),
    "self_stream_icl_cot": ("stream_bench.agents.fewshot_rag", "FewShotRAGAgent"),
    "ma_rr": ("stream_bench.agents.multiagent_rag", "MultiAgent"),
    "ma_rr_cot": ("stream_bench.agents.multiagent_rag", "MultiAgent"),
    "gt": ("stream_bench.agents.gt", "GroundTruthAgent"),
}


def load_agent(agent_name: str) -> type[Agent]:
    if agent_name not in _TASK_SPECS:
        raise ValueError("Agent %s not found" % agent_name)

    module_name, attr_name = _TASK_SPECS[agent_name]
    return getattr(import_module(module_name), attr_name)
