# Benchmark Summary

Values below are percentages.

| Baseline | Spider (EX) | CoSQL (EX) | BIRD (EX) | DS-1000 (Pass@1) | ToolBench | DDXPlus (Accuracy) | HotpotQA (EM) | HotpotQA (F1) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero-Shot | 81.79 | 59.29 | 42.05 | 81.70 | N/A | 77.15 | 68.00 | 82.92 |
| Few-Shot | 81.88 | 61.27 | 49.87 | 81.60 | N/A | 89.17 | 68.67 | 83.12 |
| CoT | 81.37 | 60.68 | 48.31 | 80.80 | N/A | 77.78 | 67.93 | 83.21 |
| Self-Refine | 81.42 | 59.29 | 43.81 | 82.60 | N/A | 77.10 | N/A | N/A |
| Self-StreamICL | 81.14 | 59.58 | 41.79 | 81.50 | N/A | 77.32 | 67.53 | 82.50 |
| MAM-StreamICL (Qwen+Gemini) | 79.46 | 57.89 | 39.63 | 73.60 | N/A | 72.96 | 65.73 | 80.90 |

## Sources

- Zero-Shot SQL results: `resume_runs/gemini3_sql_v2/summary.json`
- Zero-Shot non-SQL results: `resume_runs/gemini3_other_v2/summary.json`
- Few-Shot SQL results: `C:/Temp/streambench_compare_v2/gemini3_fewshot_sql/summary.json`
- Few-Shot non-SQL results: `C:/Temp/streambench_compare_v2/gemini3_fewshot_other/summary.json`
- CoT SQL results: `C:/Temp/streambench_compare_v2/gemini3_cot_sql/summary.json`
- CoT non-SQL results: `C:/Temp/streambench_compare_v2/gemini3_cot_other/summary.json`
- Self-Refine SQL results: `C:/Temp/streambench_compare_v2/gemini3_selfrefine_sql/summary.json`
- Self-Refine non-SQL results: `C:/Temp/streambench_compare_v2/gemini3_selfrefine_other/summary.json` (HotpotQA is not supported by the current Self-Refine benchmark implementation)
- Self-StreamICL SQL results: `C:/Temp/streambench_compare_v2/gemini3_selfstream_sql/summary.json`
- Self-StreamICL non-SQL results: `C:/Temp/streambench_compare_v2/gemini3_selfstream_other/summary.json`
- MAM-StreamICL (Qwen+Gemini) SQL results: `C:/Temp/streambench_compare_v2/mam_qwen_gemini_sql/summary.json`
- MAM-StreamICL (Qwen+Gemini) non-SQL results: `C:/Temp/streambench_compare_v2/mam_qwen_gemini_other/summary.json` / `state.json`
