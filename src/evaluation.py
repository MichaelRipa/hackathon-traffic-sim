import time
import torch
from nnterp import StandardizedTransformer


def run_evaluation(model: StandardizedTransformer, games: list[dict], remote: bool = True,
                   batch_size: int | None = None, probe_path: str | None = None, probe_layer: int | None = None):
    """Run games with optional probe scoring. batch_size=None means all at once."""
    results = []

    if batch_size is None:
        batch_size = len(games)

    # Load or init probe
    probe = None
    if probe_layer is not None:
        if probe_path:
            checkpoint = torch.load(probe_path, weights_only=True)
            probe = {"weight": checkpoint["weight"].float(), "bias": checkpoint["bias"].float()}
        else:
            hidden_dim = model.config.hidden_size
            probe = {"weight": torch.randn(1, hidden_dim) * 0.01, "bias": torch.zeros(1)}

    for batch_start in range(0, len(games), batch_size):
        batch_games = games[batch_start:batch_start + batch_size]
        start = time.time()
        try:
            with model.session(remote=remote):
                scores = []
                for game in batch_games:
                    with model.generate(game["prompt"], max_new_tokens=game["max_tokens"]) as tracer:
                        if probe:
                            gen_scores = []
                            for step in tracer.iter[:]:
                                act = model.layers_output[probe_layer][:, -1, :].cpu()
                                logit = (act * probe["weight"]).sum(dim=-1) + probe["bias"]
                                score = torch.sigmoid(logit)
                                gen_scores.append(score.save())
                            scores.append(gen_scores)
                scores.save()

            elapsed = time.time() - start
            result = {"batch_start": batch_start, "count": len(batch_games), "time": elapsed}
            if probe:
                result["scores"] = [[s.item() for s in game_scores] for game_scores in scores]
            results.append(result)
        except Exception as e:
            elapsed = time.time() - start
            print(f"[batch {batch_start}] Error: {e}")
            results.append({"batch_start": batch_start, "count": len(batch_games), "time": elapsed, "error": str(e)})

    return results
