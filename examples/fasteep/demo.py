#!/usr/bin/env python3
"""Drive FastEEP scale events on a running vLLM server.

    python demo.py --url http://localhost:8005 --sizes 4,8,4 --level 2

Prints one line per transition and, with --model, a completion sample after
each step so you can eyeball that the MoE output stayed sane.

Only the standard library is used, so this runs in any environment that can
reach the server.
"""

import argparse
import json
import time
import urllib.error
import urllib.request


def post(url: str, payload: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def complete(base: str, model: str, prompt: str, max_tokens: int) -> str:
    body = post(
        f"{base}/v1/completions",
        {"model": model, "prompt": prompt, "max_tokens": max_tokens},
        timeout=120.0,
    )
    text = body["choices"][0]["text"].strip().replace("\n", " ")
    return text[:100]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8005")
    parser.add_argument(
        "--sizes",
        default="4,8",
        help="comma-separated EP sizes to visit in order (default: 4,8)",
    )
    parser.add_argument("--level", type=int, choices=(1, 2), default=1)
    parser.add_argument("--drain-timeout", type=float, default=300.0)
    parser.add_argument(
        "--model",
        default=None,
        help="served model name; when set, sample a completion after each step",
    )
    parser.add_argument("--repeat", type=int, default=1, help="repeat the sequence")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    prompt = "Explain expert parallelism in one sentence."

    for round_idx in range(args.repeat):
        if args.repeat > 1:
            print(f"--- round {round_idx + 1}/{args.repeat} ---")
        for ep_size in sizes:
            start = time.perf_counter()
            try:
                body = post(
                    f"{args.url}/flash_epscale",
                    {
                        "ep_size": ep_size,
                        "level": args.level,
                        "drain_timeout": args.drain_timeout,
                    },
                )
            except urllib.error.HTTPError as err:
                detail = err.read().decode(errors="replace")
                print(f"ep_size={ep_size:<2} HTTP {err.code}: {detail}")
                return 1
            wall = (time.perf_counter() - start) * 1000
            timing = body.get("timing_ms", {})
            print(
                f"ep_size={ep_size:<2} action={body['action']:<10} "
                f"active={body['active_ep_size']:<2} "
                f"sleeping={str(body['sleeping_ep_ranks']):<18} "
                f"server={timing.get('total', 0.0):8.1f}ms wall={wall:8.1f}ms"
            )
            if args.model:
                print(f"  sample: {complete(args.url, args.model, prompt, 32)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
