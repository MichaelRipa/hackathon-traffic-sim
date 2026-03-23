import csv
from pathlib import Path

SIZES = {
    "xsmall": 10,
    "small": 100,
    "medium": 1000,
    "large": 10000,
    "xlarge": None,  # all
}


def load_csv(path: str, prompt_col: str = "statement", max_tokens: int = 1, layers: list[int] | None = None) -> list[dict]:
    """Load games from CSV file."""
    games = []
    with open(path) as f:
        for row in csv.DictReader(f):
            games.append({"prompt": row[prompt_col], "max_tokens": max_tokens, "layers": layers})
    return games


def load_dir(path: str, prompt_col: str = "statement", max_tokens: int = 1, layers: list[int] | None = None) -> list[dict]:
    """Load games from all CSVs in directory."""
    games = []
    for csv_file in sorted(Path(path).glob("*.csv")):
        games.extend(load_csv(str(csv_file), prompt_col, max_tokens, layers))
    return games


def load_dataset(path: str, size: str = "xlarge", prompt_col: str = "statement",
                 max_tokens: int = 1, layers: list[int] | None = None) -> list[dict]:
    """Load dataset with size preset."""
    p = Path(path)
    if p.is_dir():
        games = load_dir(path, prompt_col, max_tokens, layers)
    else:
        games = load_csv(path, prompt_col, max_tokens, layers)

    limit = SIZES.get(size)
    if limit:
        games = games[:limit]
    return games
