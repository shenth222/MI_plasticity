# src/analysis/make_subset.py
import json, argparse, random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=999)  # Fixed subset seed
    ap.add_argument("--size", type=int, required=True)  # total eval size
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    idx = list(range(args.size))
    rnd.shuffle(idx)
    idx = idx[:args.n]
    with open(args.out, "w") as f:
        json.dump(idx, f)

if __name__ == "__main__":
    main()
