# Bench

`bench.py` measures any OpenAI-compatible server (Strata, llama-server, forks) the same way. Standard library only.

```sh
bench/sysinfo.sh > bench/results/<date>-<gpu>-<cpu>/MACHINE.md
BENCH_API_KEY=... python3 bench/bench.py --url http://127.0.0.1:8080/v1 --label <engine>-<model>-<quant> \
    --out bench/results/<date>-<gpu>-<cpu> [--extra '<json merged into every request>']
```

- **decode**: a short coding prompt, 256 tokens out, temperature 0; tokens/s after the first token.
- **prefill-4096 / prefill-32768**: deterministic synthetic Python code of about that many tokens, 1 token out;
  prompt tokens / time to first token.
- Every request starts with a unique nonce, so no server answers from its prompt cache. One warm-up request first;
  3 runs per test, the median is reported.
- Turn thinking off so decode measures answer tokens: Strata `{"reasoning_effort":"none"}`, llama-server
  `{"chat_template_kwargs":{"enable_thinking":false}}`.

## Adding a result

One folder per machine and date under `results/`, with `MACHINE.md`, one JSON per engine run, and a `README.md`
listing the exact server command lines. Name what differs from the defaults.

## Windows

[windows/bench-windows.ps1](windows/) runs the same measurements on a Windows PC with an AMD card: Strata's
ready-made AMD engine and llama.cpp (Vulkan release and HIP builds), including the workarounds those needed.
