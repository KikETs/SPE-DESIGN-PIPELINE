from __future__ import annotations


def pair_match(pred: tuple[int, int], target: tuple[int, int]) -> bool:
    pa, pb = pred
    ta, tb = target
    if pa > pb:
        pa, pb = pb, pa
    if ta > tb:
        ta, tb = tb, ta
    return pa == ta and pb == tb


def ordered_match(pred: tuple[int, int], target: tuple[int, int]) -> bool:
    return int(pred[0]) == int(target[0]) and int(pred[1]) == int(target[1])


def endpoint_metrics(pred_pairs: list[tuple[int, int]], target_pairs: list[tuple[int, int]]) -> dict[str, float]:
    n = len(pred_pairs)
    if n == 0:
        return {
            "endpoint_exact_match_accuracy": 0.0,
            "endpoint_pair_accuracy": 0.0,
        }

    ordered = 0
    pair = 0
    for p, t in zip(pred_pairs, target_pairs):
        ordered += int(ordered_match(p, t))
        pair += int(pair_match(p, t))

    return {
        "endpoint_exact_match_accuracy": ordered / n,
        "endpoint_pair_accuracy": pair / n,
    }
