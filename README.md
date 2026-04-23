# DSA5106 Final Project: StreamBench Reproduction and Extension

This repository contains our DSA5106 final project based on the paper *StreamBench: Towards Benchmarking Continuous Improvement of Language Agents*.

The `upstream/` directory stores the original StreamBench codebase from the paper. On top of that code, we added project-specific scripts, experiment logs, and a small extension study examining how different memory constructions behave for stronger modern models.

## Repository Structure

- `upstream/`: original StreamBench implementation, including benchmark pipelines, agent definitions, configs, and the upstream README.
- `scripts/`: helper scripts used during our project work.
- `logs/`: experiment outputs and logs from our reproduction and extension runs.
- `conda.yaml`: exported environment used in our project.
- `makefile`: minimal environment setup helper.

## What We Did

Our project has two parts:

1. Reproduction of the main StreamBench pipeline and selected benchmark settings from the original paper.
2. Extension of the framework to study whether stronger LLMs can benefit from different memory designs in streaming evaluation.

In the extension study, we compared three memory settings on a subset of the `DS-1000` benchmark:

- positive-only memory
- negative-only memory
- mixed memory

The goal was to test whether incorrect historical examples are always harmful, or whether stronger models can make partial use of them.

## Getting Started

Most of the runnable code comes from the upstream StreamBench project. To set up the environment, you can use either:

```bash
make setup
```

or create the environment manually with the upstream requirements:

```bash
conda create -n nus-dsa5106-gp python=3.10
conda activate nus-dsa5106-gp
python -m pip install -r upstream/requirements.txt
```

If you want the exact exported environment used during the project, refer to `conda.yaml`.

## Notes

- For original usage details, benchmark commands, and dataset preparation instructions, see `upstream/README.md`.
- This repository is primarily organized as a course project archive: the upstream research code is preserved, while our additions document how we reproduced and extended the original work.

## Reference

- Original paper: <https://arxiv.org/abs/2406.08747>
