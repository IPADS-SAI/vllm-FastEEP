# Upstream tracking

This repository is a fork of [vllm-project/vllm](https://github.com/vllm-project/vllm)
carrying the FastEEP implementation. This page records exactly where the code
came from, how the history was reconstructed, and how to keep the fork in sync.

## Provenance

| Item | Value |
| --- | --- |
| Upstream base commit | [`fcae5c9`](https://github.com/vllm-project/vllm/commit/fcae5c98870ff2c20659731dce7ba3a3ff0d48aa) — "[Security] Restrict default CORS origins (#61009)", 2026-10-11, post-`v0.31.0` |
| Feature source | [`vllm-project/vllm#44994`](https://github.com/vllm-project/vllm/pull/44994) — "[Feat] Fast EP scaling with NCCL communicator split and fine-grained runtime-state management" |
| PR branch | `nanookfyf/vllm:flash_epscale-cleanup-pause-fix` |
| PR head commit | [`527350d`](https://github.com/nanookfyf/vllm/commit/527350d8c089d75efef6ec7096b8c7be5c374ee4) — 22 commits |
| Delta | 51 files, +5142 / −348 |
| Paper | [FastEEP: Fast Intra-Instance Elastic Scaling for Expert-Parallel MoE Serving](https://arxiv.org/abs/2610.18451) |
| License | Apache-2.0 (inherited) |

Authoritative text of the change: [`docs/fasteep/fasteep.patch`](docs/fasteep/fasteep.patch)
(the PR diff against the base commit) and
[`docs/features/flash_ep_scaling.md`](docs/features/flash_ep_scaling.md) (the feature
specification).

## How this repository was reconstructed

The tree at `HEAD~1` is byte-identical to the PR head, and `HEAD~2` to the
upstream base commit:

```text
1  git clone --depth 1 -b flash_epscale-cleanup-pause-fix https://github.com/nanookfyf/vllm
2  git fetch --depth 1 origin fcae5c98870ff2c20659731dce7ba3a3ff0d48aa
3  # snapshot the PR tree, check out the base commit, re-init a fresh history
4  git archive 527350d | tar -x -C /tmp/fasteep_pr
5  git checkout -f fcae5c9
6  git init -b main && git add -A && git commit  # upstream snapshot
7  rsync -a --delete --exclude=/.git /tmp/fasteep_pr/ ./
8  git add -A && git commit                      # FastEEP
```

Tree hashes recorded during the import:

| Tree | SHA |
| --- | --- |
| upstream base `fcae5c9^{tree}` | `eeb8d78429378cf748a92f2f41bbbdaa33487da1` |
| PR head `527350d^{tree}` | `d76b73f5c6cc39006fe219fc012c551bd1fa246f` |

Both were reproduced exactly by the import commits, so the content is
verifiable rather than a re-typed approximation. (Five files are listed in
upstream `.gitignore` yet tracked upstream; they were force-added so the trees
match.)

## History layout

The code history is three commits; documentation and examples are added on
top of them.

```text
233eec0  Repo hygiene: drop stray npm files, move NCCL repro under tools/fasteep
51e912a  Add FastEEP: fast intra-instance elastic EP scaling      <- PR head tree
01cf7ed  Import vLLM upstream snapshot at fcae5c9                  <- upstream base tree
```

The split keeps the feature reviewable as a single diff (`git show 51e912a`)
while making `git diff 01cf7ed 51e912a` the canonical patch.

### Why only a `v0.31.0` tag

setuptools-scm derives the package version from git tags, and the base commit
sits six days after the `v0.31.0` release. A single tag is kept on the import
commit so an editable install reports a sane version
(`0.32.0.devN+g<sha>`); it does not claim the tree is exactly `v0.31.0`.
Override it with `SETUPTOOLS_SCM_PRETEND_VERSION` when you need something else.

## Differences from the PR head

Two hygiene changes, both in `233eec0`, neither touching runtime code:

- removed root-level `package.json` / `package-lock.json` (they pinned a local
  editor dependency and are unrelated to a Python/C++ project);
- moved `repro_nccl_buf.py` to `tools/fasteep/repro_nccl_buf.py`.

Everything else is the PR as submitted, including its tests and the 230-line
feature doc.

## Re-applying the patch on a newer vLLM

```bash
git remote add upstream https://github.com/vllm-project/vllm.git   # once
git fetch upstream
git switch -c fasteep-rebase upstream/main
git apply -3 docs/fasteep/fasteep.patch
```

`-3` (three-way) surfaces conflicts instead of silently skipping hunks. The
likely conflict areas, in order of friction:

1. `vllm/v1/worker/gpu_worker.py` and `vllm/v1/worker/gpu_model_runner.py` —
   the largest hunks and the fastest-moving files upstream;
2. `vllm/distributed/device_communicators/{pynccl,all2all}.py` — NCCL/NIXL
   API churn;
3. `vllm/entrypoints/serve/**` and `vllm/engine/protocol.py` — router and RPC
   registration;
4. `vllm/device_allocator/cumem.py` — allocator bookkeeping.

Regenerate the patch after any rebase so it stays authoritative:

```bash
git diff 01cf7ed 51e912a > docs/fasteep/fasteep.patch
```

## Syncing with upstream main

FastEEP is additive: it mostly extends existing modules rather than forking
them, so a normal merge works for small upstream windows.

```bash
git fetch upstream
git switch -c sync-$(date +%Y%m%d) main
git merge upstream/main          # or: git rebase upstream/main
.venv/bin/python -m pytest -q tests/entrypoints/serve/test_flash_epscale.py
```

Watch for upstream changes that break the fast path's assumptions:

- the `--enable-elastic-ep` / ElasticEP interface, which supplies DP-rank
  routing and weight-transfer helpers;
- `sleep()` / `wake_up()` tag semantics in `device_allocator/cumem.py`;
- EPLB policy and communicator backends;
- the NIXL EP prepare/finalize layer.

If a release upstream lands FastEEP itself, prefer that implementation and
drop this fork; the PR remains the reference for the design.

## Reporting

- For FastEEP behaviour, benchmarks or design: the
  [upstream PR](https://github.com/vllm-project/vllm/pull/44994) or the paper
  authors.
- For vLLM bugs unrelated to FastEEP: upstream issue tracker.
