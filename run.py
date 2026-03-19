#!/usr/bin/env python3
import argparse
import json
import os
import nnsight
import dotenv

from src.dataset import load_csv
from src.evaluation import run_evaluation

dotenv.load_dotenv()

nnsight.CONFIG.set_default_api_key(os.getenv("NNSIGHT_API_KEY"))
nnsight.CONFIG.API.HOST = os.getenv("NNSIGHT_API_HOST", "https://api.ndif.us")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prompt-col", default="statement")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--remote", action="store_true", default=True)
    parser.add_argument("--layers", type=int, nargs="*", help="Layers to extract (whitebox)")
    parser.add_argument("--max-tokens", type=int, default=1)
    args = parser.parse_args()

    model = nnsight.LanguageModel(args.model)
    games = load_csv(args.dataset, args.prompt_col, args.max_tokens, args.layers)

    print(f"Running {len(games)} games, batch={args.batch}, remote={args.remote}")
    results = run_evaluation(model, games, remote=args.remote, batch=args.batch)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()