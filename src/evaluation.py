import time
import nnsight


def run_evaluation(model: nnsight.LanguageModel, games: list[dict], remote: bool = True, batch: bool = False):
    """Run games and return results with timing."""
    results = []

    if batch:
        start = time.time()
        with model.session(remote=remote):
            for game in games:
                with model.generate(game["prompt"], max_new_tokens=game["max_tokens"]) as tracer:
                    out = model.generator.output.save()
                    if game["layers"]:
                        for step in tracer.iter[:]:
                            acts = {l: model.model.layers[l].output[0].save() for l in game["layers"]}
        elapsed = time.time() - start
        results.append({"batch": True, "count": len(games), "time": elapsed})
    else:
        for game in games:
            start = time.time()
            with model.session(remote=remote):
                with model.generate(game["prompt"], max_new_tokens=game["max_tokens"]) as tracer:
                    out = model.generator.output.save()
                    if game["layers"]:
                        for step in tracer.iter[:]:
                            acts = {l: model.model.layers[l].output[0].save() for l in game["layers"]}
            elapsed = time.time() - start
            results.append({"prompt": game["prompt"][:50], "time": elapsed})

    return results
