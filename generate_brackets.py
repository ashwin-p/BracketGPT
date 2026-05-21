#!/usr/bin/env python3
"""
Generate 160k UNIQUE bracket-opening strings (length 1..30) and their valid closing strings,
then split deterministically into:
- train: 100,000
- val:   10,000
- test:  50,000

Columns:
- opening
- closing

Dependencies:
    pip install tqdm
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Tuple, Set

from tqdm import tqdm

OPEN_TO_CLOSE = {
    "(": ")",
    "[": "]",
}
OPEN_SYMBOLS = ("(", "[")


def make_closing(opening: str) -> str:
    return "".join(OPEN_TO_CLOSE[ch] for ch in reversed(opening))


def sample_opening(rng: random.Random, min_len: int, max_len: int) -> str:
    L = rng.randint(min_len, max_len)
    return "".join(rng.choice(OPEN_SYMBOLS) for _ in range(L))


def open_writers(out_dir: Path) -> Tuple[csv.writer, csv.writer, csv.writer, object, object, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    f_train = (out_dir / "train.csv").open("w", newline="", encoding="utf-8")
    f_val = (out_dir / "val.csv").open("w", newline="", encoding="utf-8")
    f_test = (out_dir / "test_34.csv").open("w", newline="", encoding="utf-8")

    w_train = csv.writer(f_train)
    w_val = csv.writer(f_val)
    w_test = csv.writer(f_test)

    header = ["opening", "closing"]
    w_train.writerow(header)
    w_val.writerow(header)
    w_test.writerow(header)

    return w_train, w_val, w_test, f_train, f_val, f_test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-rows", type=int, default=65_536)
    parser.add_argument("--train-count", type=int, default=0)
    parser.add_argument("--val-count", type=int, default=0)
    parser.add_argument("--test-count", type=int, default=65_536)
    parser.add_argument("--min-len", type=int, default=34)
    parser.add_argument("--max-len", type=int, default=34)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    if args.train_count + args.val_count + args.test_count != args.total_rows:
        raise ValueError("Split counts must sum to total_rows.")

    rng = random.Random(args.seed)

    seen: Set[str] = set()

    w_train, w_val, w_test, f_train, f_val, f_test = open_writers(args.out_dir)

    try:
        with tqdm(total=args.total_rows, desc="Generating unique rows", unit="row") as pbar:
            i = 0
            while i < args.total_rows:
                opening = sample_opening(rng, args.min_len, args.max_len)

                if opening in seen:
                    continue

                seen.add(opening)
                closing = make_closing(opening)

                if i < args.train_count:
                    w_train.writerow([opening, closing])
                elif i < args.train_count + args.val_count:
                    w_val.writerow([opening, closing])
                else:
                    w_test.writerow([opening, closing])

                i += 1
                pbar.update(1)
    finally:
        f_train.close()
        f_val.close()
        f_test.close()

    print("Done.")
    print(f"Unique rows generated: {len(seen)}")
    print(f"train: {args.train_count} rows -> {args.out_dir / 'train.csv'}")
    print(f"val:   {args.val_count} rows -> {args.out_dir / 'val.csv'}")
    print(f"test:  {args.test_count} rows -> {args.out_dir / 'test_34.csv'}")


if __name__ == "__main__":
    main()

