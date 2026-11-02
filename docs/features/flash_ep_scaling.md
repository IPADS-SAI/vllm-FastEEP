# Flash EP Scaling

Flash EP scaling (FastEEP) is a fast path for resizing the **active** Expert
Parallel (EP) group of a running vLLM serving instance. It targets MoE
deployments where the **maximum** DP/EP size is known at startup: the server
pays the cost of weight loading, kernel compilation, graph capture and NCCL
topology construction **once**, and later scaling operations only resize the
active subset of that prepared topology.

In the authors' 8-GPU single-node tests this brings end-to-end EP scale-up/down
from tens of seconds (with the existing
[Elastic EP](https://vllm.ai/blog/2026-05-14-elastic-expert-parallelism)
rebuild flow) down to **~2–4 seconds**. See the
[benchmark table](https://github.com/vllm-project/vllm/pull/44994) for the
measured transitions.

!!! note
    Flash EP scaling is exposed via a development endpoint and requires
    `VLLM_SERVER_DEV_MODE=1`. It is intended for operators, not for
    end-user requests.

## How it works

The server starts at the maximum DP/EP size and stays there for the
lifetime of the process. `/flash_epscale` then runs one of three paths:

```text
                   ┌─────────────────────────────────────────────────┐
                   │  /flash_epscale  ep_size=N                      │
                   └─────────────────────────────────────────────────┘
                                       │
                       ┌───────────────┼───────────────┐
                       ▼               ▼               ▼
                     noop          scale_down       scale_up
                       │               │               │
                       │               │               │
                       │  drain target ranks (active   │  pause
                       │  ranks keep serving)          │
                       │               │               │  wake offloaded
                       │               │  pause        │  ranks
                       │               │               │
                       │               │  wake / EPLB  │  resize NCCL
                       │               │  remap / NCCL │  group
                       │               │  split        │
                       │               │               │  resume
                       │               │  offload to   │
                       │               │  CPU          │  open routing
                       │               │               │  to new ranks
                       │               │  resume       │
                       └───────────────┴───────────────┘
```

- **Scale down** rearranges experts off the to-be-inactive EP ranks,
  offloads their weights and KV cache to CPU (sleep/wake-style), splits
  the NCCL communicator with `ncclCommSplit`, and masks those ranks in
  NIXL all2all. Active ranks keep serving while the soon-to-sleep ranks
  drain in flight. Routing shrinks and in-flight requests are re-issued on
  the active prefix before the drain, so the drain window is short.
- **Scale up** wakes the previously offloaded ranks, restores the
  logical expert mapping, resizes the NCCL group, and only then opens
  routing back to the larger active set, so requests never reach a rank
  whose communicator is still being resized.

### Execution window

Only steps that must not interleave with forward collectives run inside a
stop-the-world **pause window** (NCCL resize, EPLB remap, NIXL
connect/disconnect, L2 refill). Everything else — routing updates, in-flight
rerouting, draining, pinned-buffer pre-allocation, and the actual memory
release on sleeping ranks — runs outside it, so the active ranks keep
producing tokens for most of the transition.

Two invariants follow from this:

1. **Routing is monotone with readiness.** Scale-down shrinks routing before
   draining; scale-up grows routing only after the transition completed.
2. **Rollback is best-effort before the resize, forbidden after it.** A
   failure before the resize restores the previous sleeping set; once the
   resize is live, only the memory release may fail, and that failure is
   reported without rolling back (rolling back would corrupt live rank
   placement).

### Concurrency and idempotency

Scaling calls are serialized by a server-side lock: a second request waits for
the first transition to finish. Requesting the current active size returns
`action: "noop"` with `changed: false` and does not touch memory or routing.

## Memory modes

`level` selects what happens to the allocations on ranks that sleep or wake.

| Level | Behaviour | CPU cost | Notes |
| --- | --- | --- | --- |
| `1` (default) | Copy tagged allocations out to pinned host memory, unmap the GPU pages; copy back on wake. | Weights + KV cache of every sleeping rank. | Fastest wake; needs enough host RAM. The first call pays a one-off pinned-buffer allocation (tens of seconds at 30B scale). |
| `2` | Discard contents, keep the virtual address range; on wake, refill dense weights and the EPLB map from active peers over NIXL/PyNCCL. | Only the CUDA graph pool backup. | For CPU-constrained hosts. Requires `--eplb-config.communicator pynccl` with `--eplb-config.use_async false` in the validated configuration. |

CUDA graph capture allocations always go to a dedicated CuMem `graphs` pool and
are backed up even in level 2: the pool is remapped at the same virtual
addresses on wake, so captured graphs stay valid without re-capture. This path
requires `enable_cumem_allocator=True` and capture into that pool (it is not
exercised under `--enforce-eager`).

## Requirements

To use this endpoint, the server must be started with:

- `--enable-sleep-mode` — the offload path reuses sleep/wake
  infrastructure.
- `--enable-expert-parallel` and `--enable-eplb` — the EPLB state
  manages the logical-sleep mapping during resize.
- `--enable-elastic-ep` — required for routing requests away from
  to-be-inactive DP ranks.
- `--all2all-backend nixl_ep` — peer masking during scale_down uses the
  NIXL EP backend.
- `--data-parallel-backend ray` and a `--data-parallel-size` equal to
  the maximum EP world size.
- `VLLM_SERVER_DEV_MODE=1` in the environment to expose
  `/flash_epscale`.

Validated configuration: single node, `TP=1`, 8×H20-96G,
`--gpu-memory-utilization 0.45`, `--max-model-len 4096`,
`--eplb-config.num_redundant_experts 64`,
`--eplb-config.communicator pynccl`, `--eplb-config.use_async false`,
`nixl-cu13==1.0.1`.

## Environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `VLLM_SERVER_DEV_MODE` | `0` | Registers `/flash_epscale` and the sleep-by-tag helpers. Without it the endpoint is `404`. |
| `VLLM_FLASH_EPSCALE_SUSPEND_MAX_DP` | `1` | After a scale-down, call `ncclCommSuspend` on the retained max-DP communicator so its GPU buffers are released; `resume_max_dp_comm()` re-creates them on wake. Set to `0` to keep the buffers resident and shave recovery latency at the cost of sleeping-rank memory. |
| `VLLM_FLASH_EPSCALE_SKIP_NIXL_TEARDOWN` | `0` | Set to `1` to keep the sleeping ranks' NIXL buffers registered. Peers are still disconnected and masked; only the destroy step is skipped. Lower latency, more memory held. |

## HTTP endpoint

### `POST /flash_epscale`

Resize the active EP group to `ep_size`.

#### Request body

```json
{
  "ep_size": 4,
  "level": 2,
  "tags": ["shared_weights", "expert_weights", "kv_cache"],
  "drain_timeout": 300
}
```

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `ep_size` | int | yes | — | New active EP size; must be in `[1, ep_world_size]`. |
| `level` | int | no | `1` | Memory mode for ranks that sleep or wake: `1` = CPU backup, `2` = discard + peer refill. Any other value is rejected with `400`. |
| `tags` | list&nbsp;of&nbsp;str | no | `["shared_weights", "expert_weights", "kv_cache"]` | CuMem allocator tags to offload (scale-down) or wake (scale-up). `"weights"` expands to `shared_weights` + `expert_weights`. |
| `drain_timeout` | number&nbsp;(seconds) | no | `300` | Time to wait for in-flight requests on to-be-inactive ranks to finish. Only used by `scale_down`. |

#### Response body

```json
{
  "ok": true,
  "ep_world_size": 8,
  "active_ep_size": 4,
  "sleeping_ep_ranks": [4, 5, 6, 7],
  "changed": true,
  "action": "scale_down",
  "tags": ["shared_weights", "expert_weights", "kv_cache"],
  "timing_ms": {
    "query_state": 0.0,
    "route_shrink": 0.0,
    "reroute": 0.0,
    "drain": 0.0,
    "warm_backup": 0.0,
    "pause": 0.0,
    "resize": 0.0,
    "nixl_disconnect": 0.0,
    "resume": 0.0,
    "sleep": 0.0,
    "nixl_destroy": 0.0,
    "final_state": 0.0,
    "total": 0.0
  }
}
```

| Field | Description |
| --- | --- |
| `action` | One of `noop`, `scale_up`, `scale_down`. |
| `changed` | `true` if the active group was resized. |
| `sleeping_ep_ranks` | Suffix of EP ranks currently offloaded. |
| `timing_ms` | Per-step timing for diagnosing latency. Only steps that actually ran are present. |

`timing_ms` keys correspond to the orchestration steps:

| Key | Phase | Description |
| --- | --- | --- |
| `query_state` / `final_state` | pre / post | Pre- and post-transition EP-state queries across workers. |
| `route_shrink` / `route_grow` | outside pause | Front-end active-DP routing updates. |
| `reroute` | outside pause | Best-effort re-issue of in-flight requests off the shrinking ranks. `reroute_count` records how many were moved. |
| `drain` | outside pause | Wait for in-flight requests on the to-be-inactive DP ranks (scale-down only). |
| `pause` / `resume` | boundary | Engine stop-the-world enter/exit. |
| `wake` | pre-pause | `wake_up_ep_ranks` collective RPC. |
| `warm_backup` | pre-pause | `warm_sleep_backup`: pre-pin the host buffers the post-resume sleep will use (best effort, failures are logged and ignored). |
| `resize` | in pause | `resize_sleep_ep_ranks` collective RPC (EPLB remap + NIXL mask + `ncclCommSplit`). |
| `nixl_ensure` | in pause | Rebuild or reconnect NIXL peers before a refill / resize. |
| `l2_refill` | in pause | `finalize_l2_wake`: dense-weight and EPLB-map refill from active peers (level 2 only). |
| `nixl_disconnect` | in pause | Make the live ranks forget the new sleeping peers' endpoints. |
| `sleep` | post-resume | `sleep_ep_ranks_by_tags` collective RPC (CPU copy-out or discard). |
| `nixl_destroy` | post-resume | Destroy the sleeping ranks' RDMA buffers (skipped with `VLLM_FLASH_EPSCALE_SKIP_NIXL_TEARDOWN=1`). |
| `rollback` | on failure | Best-effort revert of a failed transition. |
| `total` | — | Wall time of the whole call. |

#### Error responses

- `400` — invalid `ep_size`, `level`, `tags`, or `drain_timeout`.
- `404` — `VLLM_SERVER_DEV_MODE` is not set, so the router is not registered.
- `500` — workers reported inconsistent EP state, or a step inside
  the pause window failed. The endpoint attempts to resume the engine
  before returning the error; resume failures are propagated as errors
  instead of being reported as success.

A `500` that happens *after* the resize (memory release on the sleeping ranks
failed) does not roll back: the new active set is already live and correct,
and the response reports the release failure. Retrying the same call is safe —
the transition becomes a `noop` plus a retried release.

## Helper endpoints

These are reused internally by `/flash_epscale` but can also be called
directly when scripting recovery flows. They do not update routing or EPLB
state, so prefer `/flash_epscale` unless you are implementing recovery.

### `POST /sleep_ep_ranks_tags`

Offload selected ranks for the given tags. Body:

```json
{
  "sleeping_ep_ranks": [4, 5, 6, 7],
  "tags": ["expert_weights", "kv_cache"],
  "level": 1
}
```

### `POST /wake_up_ep_ranks_tags`

Reverse of the above; restores the listed tags on the listed ranks.

## Example session

Start the server at the maximum size:

```bash
export VLLM_SERVER_DEV_MODE=1
export VLLM_FLASH_EPSCALE_SUSPEND_MAX_DP=1
export VLLM_FLASH_EPSCALE_SKIP_NIXL_TEARDOWN=0

vllm serve "$MODEL" \
    --trust-remote-code \
    --enable-sleep-mode \
    --enable-expert-parallel \
    --enable-eplb \
    --enable-elastic-ep \
    --all2all-backend nixl_ep \
    --eplb-config.num_redundant_experts 64 \
    --eplb-config.communicator pynccl \
    --eplb-config.use_async false \
    --data-parallel-backend ray \
    --distributed-executor-backend ray \
    --data-parallel-size 8 \
    --data-parallel-size-local 8 \
    --gpu-memory-utilization 0.45 \
    --port 8005 \
    --enforce-eager
```

Scale down from EP=8 to EP=4 while serving:

```bash
curl -X POST http://localhost:8005/flash_epscale \
  -H 'Content-Type: application/json' \
  -d '{"ep_size": 4, "level": 2}'
```

Scale back up:

```bash
curl -X POST http://localhost:8005/flash_epscale \
  -H 'Content-Type: application/json' \
  -d '{"ep_size": 8}'
```

## Limitations

- **Bounded by the startup size.** This path only resizes the active
  subset within the EP world that was constructed at startup. Going
  beyond the startup maximum requires the existing ElasticEP flow or
  NCCL `grow`/`shrink` primitives.
- **Wait policy during scaling.** In-flight requests on the
  to-be-inactive ranks are drained before the transition runs. Active
  ranks keep serving during the drain.
- **First scaling call is slower.** Sleep/wake (level 1) allocates a
  pinned CPU buffer the first time it is invoked (~20 s). Subsequent
  scaling calls run at ~1 s per offload/restore.
- **MoE only.** The endpoint only makes sense for EPLB-managed MoE
  deployments.
- **Single node, `TP=1`.** Multi-node, TP/PP and graph-replay paths are not
  validated yet.
- **Level 2 + NIXL weight migration** hit UCX failures across repeated wake
  cycles with NIXL 1.0.1; the validated configuration uses synchronous PyNCCL.

## Related

- [Sleep mode](sleep_mode.md) — the offload primitive flash EP scaling
  builds on.
- [Elastic EP](https://vllm.ai/blog/2026-05-14-elastic-expert-parallelism) —
  the existing path for changes that exceed the startup topology.
