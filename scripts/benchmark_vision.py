from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

from vision.lfm2_5_vl_engine import LFM25VLEngine, VisionEngineError


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark LFM2.5-VL with a local OpenAI-compatible vision server."
    )
    parser.add_argument("image", help="Path to the benchmark image.")
    parser.add_argument(
        "--prompt",
        default="What is this? Answer in one short sentence.",
        help="Question to ask the vision model.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of measured inference runs.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Request structured JSON output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image = Path(args.image)
    if not image.is_file():
        print(f"[VisionBenchmark] Image not found: {image}")
        return 2

    engine = LFM25VLEngine()
    if not engine.warmup():
        print(
            "[VisionBenchmark] Vision server is not ready. "
            "Set ASTA_VISION_BASE_URL and ASTA_VISION_MODEL as needed."
        )
        return 2

    ttfts: list[float] = []
    speeds: list[float] = []
    totals: list[float] = []

    for index in range(max(1, args.runs)):
        try:
            result = engine.inspect(
                image,
                args.prompt,
                json_mode=args.json,
            )
        except VisionEngineError as exc:
            print(f"[VisionBenchmark] run {index + 1} failed: {exc}")
            return 3

        ttft = result.get("ttft")
        speed = result.get("tokens_per_second")
        total = result.get("request_time")

        if isinstance(ttft, (int, float)):
            ttfts.append(float(ttft))
        if isinstance(speed, (int, float)) and speed > 0:
            speeds.append(float(speed))
        if isinstance(total, (int, float)):
            totals.append(float(total))

        print(
            f"run={index + 1} "
            f"TTFT={ttft if ttft is not None else 'n/a'}s "
            f"tok/s={speed if speed else 'n/a'} "
            f"total={total if total is not None else 'n/a'}s"
        )
        print(f"answer={result.get('text', '')}")

    def avg(values: list[float]):
        return statistics.mean(values) if values else None

    print("\n=== LFM2.5-VL benchmark summary ===")
    print(f"runs={max(1, args.runs)}")
    print(f"avg_ttft_s={avg(ttfts)}")
    print(f"avg_tok_s={avg(speeds)}")
    print(f"avg_total_s={avg(totals)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
