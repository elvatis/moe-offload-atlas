# RTX 2080 Ti 11 GB + Threadripper 3960X, 128 GB DDR4-3200 (quad channel)

Machine details: [MACHINE.md](MACHINE.md). Thinking off in every run. Medians of 3 runs, `bench.py` defaults.

| Engine | Model / quant | Decode tok/s | Prefill 4K tok/s | Prefill 32K tok/s |
|---|---|---:|---:|---:|
| Strata 0.1.34 (main @ 1678de3) | Qwen3.8-Flash-Next IQ3_S | 56.0 | 839 | 783 |

## Command lines

**Strata** (`strata-iq3_s.json`, set up by Strata's installer): `serve/server.py --engine strata --config strata-iq3_s.json`
with `--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp <rt> --max-context 262144 --kv int8
--kv-resident 32768 --vision --vram-reserve-mib 700`. Request extra: `{"reasoning_effort":"none"}`.
