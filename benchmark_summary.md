# Benchmark Summary

Values below are percentages.

| Baseline | Spider (EX) | CoSQL (EX) | BIRD (EX) | DS-1000 (Pass@1) | ToolBench | DDXPlus (Accuracy) | HotpotQA (EM) | HotpotQA (F1) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero-Shot | 81.79 | 59.29 | 42.05 | 81.70 | N/A | 77.15 | 68.00 | 82.92 |
| Few-Shot | 81.88 | 61.27 | 49.87 | 81.60 | N/A | 89.17 | 68.67 | 83.12 |
| CoT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Self-Refine | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Sources

- Zero-Shot SQL results: `resume_runs/gemini3_sql_v2/summary.json`
- Zero-Shot non-SQL results: `resume_runs/gemini3_other_v2/summary.json`
- Few-Shot SQL results: `C:/Temp/streambench_compare_v2/gemini3_fewshot_sql/summary.json`
- Few-Shot non-SQL results: `C:/Temp/streambench_compare_v2/gemini3_fewshot_other/summary.json`
