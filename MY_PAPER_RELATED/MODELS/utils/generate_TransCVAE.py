from utils.utils import *


@torch.no_grad()
def generate_batch_sequence(
    model,
    z,
    max_length: int,
    start_token: int,
    end_token: int,
    pad_token: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    device: str = "cuda",
):
    """
    Transformer-CVAE 배치 생성.
    """
    model.eval()

    B = z.size(0)
    z = z.to(device)

    generated: list[list[int]] = [[start_token] for _ in range(B)]
    finished = [False] * B

    for _ in range(max_length):
        cur_len = max(len(seq) for seq in generated)
        seq_tensor = torch.full((B, cur_len), pad_token, dtype=torch.long, device=device)
        seq_lens = []

        for i, seq in enumerate(generated):
            seq_lens.append(len(seq))
            seq_tensor[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

        logits = model.decoder(seq_tensor, z)
        logits = model.predict(logits) / temperature

        last_pos = torch.tensor([l - 1 for l in seq_lens], device=device)
        next_logits = logits[torch.arange(B, device=device), last_pos]

        if top_k is not None:
            topk_val, _ = torch.topk(next_logits, top_k)
            threshold = topk_val[:, -1, None]
            next_logits[next_logits < threshold] = -float("inf")

        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(next_logits, dim=-1, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative = probs.cumsum(dim=-1)
            mask = cumulative > top_p
            mask[..., 0] = False
            sorted_logits[mask] = -float("inf")
            next_logits.scatter_(1, sorted_idx, sorted_logits)

        probs = F.softmax(next_logits, dim=-1)

        all_done = True
        for i in range(B):
            if finished[i]:
                continue
            tok = torch.multinomial(probs[i], 1).item()
            generated[i].append(tok)
            if tok == end_token:
                finished[i] = True
            else:
                all_done = False

        if all_done:
            break

    return generated
