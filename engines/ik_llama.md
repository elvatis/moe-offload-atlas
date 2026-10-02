# ik_llama.cpp

[ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp): llama.cpp fork with faster CPU kernels
(`iqk_mul_mat`) and its own quant types.

- MiMo support: [#1723](https://github.com/ikawrakow/ik_llama.cpp/pull/1723) "Support Mimo-2.5" (merged), DFlash drafts
  for MiMo-V2.5-Pro ([#2048](https://github.com/ikawrakow/ik_llama.cpp/pull/2048)).
- No VRAM expert-cache work found (searched PRs and issues, 2026-10-02).
- [#2259](https://github.com/ikawrakow/ik_llama.cpp/pull/2259) (open): ATSInfer automatic tensor placement solver
  (DP knapsack + hardware profiler); the author reports 2.4x decode over manual `--n-cpu-moe` on an RTX 5090.
- [#2585](https://github.com/ikawrakow/ik_llama.cpp/pull/2585) (open): work-stealing chunking for MoE CPU kernels.

Not measured here yet.
