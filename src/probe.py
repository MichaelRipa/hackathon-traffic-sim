import time
import torch
from nnterp import StandardizedTransformer


def run_probe_training(model: StandardizedTransformer, games: list[dict], remote: bool = True,
                       batch: bool = False, all_tokens: bool = False, layers: list[int] | None = None,
                       epochs: int = 1, lr: float = 0.1):
    """Simulate probe training on remote server."""
    results = []

    if layers is None:
        layers = list(range(model.config.num_hidden_layers))

    hidden_dim = model.config.hidden_size

    # Initialize probes (one per layer)
    probes = {l: {"weight": torch.randn(1, hidden_dim) * 0.01, "bias": torch.zeros(1)} for l in layers}

    # Fake labels for simulation (alternate 0/1)
    labels = [float(i % 2) for i in range(len(games))]

    for epoch in range(epochs):
        start = time.time()
        errors = 0

        if batch:
            try:
                with model.session(remote=remote):
                    grads = {l: {"w": [], "b": []} for l in layers}
                    for game, label in zip(games, labels):
                        with model.trace(game["prompt"]):
                            for l in layers:
                                if all_tokens:
                                    act = model.layers_output[l].mean(dim=1).cpu()
                                else:
                                    act = model.layers_output[l][:, -1, :].cpu()

                                logit = (act * probes[l]["weight"]).sum(dim=-1) + probes[l]["bias"]
                                prob = torch.sigmoid(logit)
                                d_logit = prob - label
                                grad_w = (d_logit.unsqueeze(-1) * act)
                                grad_b = d_logit
                                grads[l]["w"].append(grad_w.save())
                                grads[l]["b"].append(grad_b.save())
                    grads.save()

                for l in layers:
                    mean_gw = torch.stack([g.detach().cpu() for g in grads[l]["w"]]).mean(dim=0)
                    mean_gb = torch.stack([g.detach().cpu() for g in grads[l]["b"]]).mean(dim=0)
                    probes[l]["weight"] -= lr * mean_gw
                    probes[l]["bias"] -= lr * mean_gb
            except Exception as e:
                errors += 1
                print(f"[epoch {epoch+1}] Batch error: {e}")

        else:
            for i, (game, label) in enumerate(zip(games, labels)):
                try:
                    with model.session(remote=remote):
                        with model.trace(game["prompt"]):
                            for l in layers:
                                if all_tokens:
                                    act = model.layers_output[l].mean(dim=1).cpu()
                                else:
                                    act = model.layers_output[l][:, -1, :].cpu()

                                logit = (act * probes[l]["weight"]).sum(dim=-1) + probes[l]["bias"]
                                prob = torch.sigmoid(logit)
                                d_logit = prob - label
                                grad_w = (d_logit.unsqueeze(-1) * act).save()
                                grad_b = d_logit.save()

                        probes[l]["weight"] -= lr * grad_w.detach().cpu()
                        probes[l]["bias"] -= lr * grad_b.detach().cpu()
                except Exception as e:
                    errors += 1
                    print(f"[epoch {epoch+1}, game {i}] Error: {e}")

        elapsed = time.time() - start
        results.append({"epoch": epoch + 1, "layers": len(layers), "games": len(games), "errors": errors, "time": elapsed})

    return results
