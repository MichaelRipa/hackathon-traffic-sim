import csv


def load_csv(path: str, prompt_col: str = "statement", max_tokens: int = 1, layers: list[int] | None = None) -> list[dict]:
    """Load games from CSV file."""
    games = []
    with open(path) as f:
        for row in csv.DictReader(f):
            games.append({"prompt": row[prompt_col], "max_tokens": max_tokens, "layers": layers})
    return games
