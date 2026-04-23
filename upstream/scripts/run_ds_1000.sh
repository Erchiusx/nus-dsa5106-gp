python -m stream_bench.pipelines.run_bench \
    --bench_cfg "configs/bench/ds_1000.yml" \
    --agent_cfg "configs/agent/$1.yml"
    # --entity "photocopier" \
    # --use_wandb
