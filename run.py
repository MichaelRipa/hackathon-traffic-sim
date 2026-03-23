#!/usr/bin/env python3
import argparse
import json
import os
import nnsight
import dotenv
from nnterp import StandardizedTransformer

from src.dataset import load_dataset
from src.evaluation import run_evaluation
from src.probe import run_probe_training

dotenv.load_dotenv()

nnsight.CONFIG.set_default_api_key(os.getenv("NNSIGHT_API_KEY"))
nnsight.CONFIG.API.HOST = os.getenv("NNSIGHT_API_HOST", "https://api.ndif.us")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["static-game", "probe"], default="static-game")
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--dataset", default="data/liars-bench")
    parser.add_argument("--size", choices=["xsmall", "small", "medium", "large", "xlarge"], default="xsmall")
    parser.add_argument("--prompt-col", default="statement")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--batch-all", action="store_true", help="Batch entire dataset at once")
    parser.add_argument("--remote", action="store_true", default=True)
    parser.add_argument("--local", action="store_false", dest="remote")
    parser.add_argument("--layers", type=int, nargs="*", help="Layers to extract")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--all-tokens", action="store_true", help="Probe: use all tokens (default: last only)")
    parser.add_argument("--epochs", type=int, default=1, help="Probe: training epochs")
    parser.add_argument("--lr", type=float, default=0.1, help="Probe: learning rate")
    parser.add_argument("--local-grads", action="store_true", help="Probe: download acts, compute grads locally")
    args = parser.parse_args()

    model = StandardizedTransformer(args.model)
    games = load_dataset(args.dataset, args.size, args.prompt_col, args.max_tokens, args.layers)

    batch_size = None if args.batch_all else args.batch_size
    print(f"Mode: {args.mode}, {len(games)} games, batch_size={batch_size or 'all'}, remote={args.remote}")

    if args.mode == "static-game":
        results = run_evaluation(model, games, remote=args.remote, batch_size=batch_size)
    else:
        results = run_probe_training(model, games, remote=args.remote, batch_size=batch_size,
                                     all_tokens=args.all_tokens, layers=args.layers,
                                     epochs=args.epochs, lr=args.lr, local_grads=args.local_grads)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
