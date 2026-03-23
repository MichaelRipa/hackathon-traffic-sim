import time
from nnterp import StandardizedTransformer


def run_evaluation(model: StandardizedTransformer, games: list[dict], remote: bool = True, batch_size: int | None = None):
    """Run games and return results with timing. batch_size=None means all at once."""
    results = []

    if batch_size is None:
        batch_size = len(games)

    for batch_start in range(0, len(games), batch_size):
        batch_games = games[batch_start:batch_start + batch_size]
        start = time.time()
        try:
            with model.session(remote=remote):
                for game in batch_games:
                    with model.generate(game["prompt"], max_new_tokens=game["max_tokens"]) as tracer:
                        out = model.generator.output.save()
                        if game["layers"]:
                            for step in tracer.iter[:]:
                                acts = {l: model.layers_output[l].save() for l in game["layers"]}
            elapsed = time.time() - start
            results.append({"batch_start": batch_start, "count": len(batch_games), "time": elapsed})
        except Exception as e:
            elapsed = time.time() - start
            print(f"[batch {batch_start}] Error: {e}")
            results.append({"batch_start": batch_start, "count": len(batch_games), "time": elapsed, "error": str(e)})

    return results
