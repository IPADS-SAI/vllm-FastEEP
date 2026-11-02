# FastEEP examples

Two small scripts that reproduce the quickstart in the root README.

| File | Purpose |
| --- | --- |
| `serve.sh` | Start vLLM at the **maximum** EP size with the validated FastEEP flag set (sleep mode, EPLB, ElasticEP, NIXL EP, PyNCCL migration, Ray DP). |
| `demo.py` | Walk a list of EP sizes through `POST /flash_epscale` and print action, active size, sleeping ranks and per-call timing. |

## Run

```bash
# terminal 1
MODEL=/models/DeepSeek-V2-Lite DP=8 ./serve.sh

# terminal 2 — shrink to 4 (discard mode), then grow back to 8
python demo.py --url http://localhost:8005 --sizes 4,8 --level 2

# with an output sanity check after every transition
python demo.py --sizes 4,8 --model DeepSeek-V2-Lite --repeat 2
```

`serve.sh` honours `MODEL`, `DP`, `PORT`, `GPU_MEM_UTIL`, `MAX_MODEL_LEN`,
`REDUNDANT_EXPERTS`, `RAY_PORT`, `VLLM_FLASH_EPSCALE_SUSPEND_MAX_DP` and
`VLLM_FLASH_EPSCALE_SKIP_NIXL_TEARDOWN`.

Notes:

- `DP` is the ceiling. FastEEP cannot scale beyond the size the server started
  with, so pick the peak you may need rather than today's size.
- The first level-1 transition pays a one-off pinned-buffer allocation. Run one
  scale-down/scale-up cycle before timing anything.
- `demo.py` needs no third-party packages; `serve.sh` needs `vllm`, `ray` and
  an 8-GPU node.
