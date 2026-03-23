import time
from nnterp import StandardizedTransformer


def run_evaluation(model: StandardizedTransformer, games: list[dict], remote: bool = True, batch: bool = False):
    """Run games and return results with timing."""
    results = []

    if batch:
        start = time.time()
        try:
            with model.session(remote=remote):
                for game in games:
                    with model.generate(game["prompt"], max_new_tokens=game["max_tokens"]) as tracer:
                        out = model.generator.output.save()
                        if game["layers"]:
                            for step in tracer.iter[:]:
                                acts = {l: model.layers_output[l].save() for l in game["layers"]}
            elapsed = time.time() - start
            results.append({"batch": True, "count": len(games), "time": elapsed})
        except Exception as e:
            elapsed = time.time() - start
            results.append({"batch": True, "count": len(games), "time": elapsed, "error": str(e)})
    else:
        for i, game in enumerate(games):
            start = time.time()
            try:
                with model.session(remote=remote):
                    with model.generate(game["prompt"], max_new_tokens=game["max_tokens"]) as tracer:
                        out = model.generator.output.save()
                        if game["layers"]:
                            for step in tracer.iter[:]:
                                acts = {l: model.layers_output[l].save() for l in game["layers"]}
                elapsed = time.time() - start
                results.append({"prompt": game["prompt"][:50], "time": elapsed})
            except Exception as e:
                elapsed = time.time() - start
                print(f"[{i}] Error: {e}")
                results.append({"prompt": game["prompt"][:50], "time": elapsed, "error": str(e)})

    return results
