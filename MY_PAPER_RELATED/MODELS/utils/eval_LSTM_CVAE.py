from utils.utils import *
from utils.dataloader import dataset
from rdkit import DataStructs

PAD_IDX = dataset.vocab["[PAD]"]
vocab = dataset.vocab
index_to_token = {idx: token for token, idx in vocab.items()}


def _smiles_to_fp(sm: str):
    mol = Chem.MolFromSmiles(sm)
    return Chem.RDKFingerprint(mol) if mol is not None else None


def tanimoto(sm1: str, sm2: str) -> float:
    fp1, fp2 = _smiles_to_fp(sm1), _smiles_to_fp(sm2)
    if fp1 is None or fp2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def _beam_search_decode(
    model,
    z,
    sos_id: int,
    eos_id: int,
    *,
    beam_width: int = 20,
    max_len: int = 128,
    len_penalty: float = 1.0,
    vocab_size: int = 9999,
    temperature: float = 0.9,
    alpha: float = 0.7,
):
    device = z.device
    batch = z.size(0)
    out_tokens = []

    for b in tqdm(range(batch)):
        beams = [([sos_id], 0.0, 0.0)]
        finished = []

        for _ in range(max_len):
            new_beams = []
            for seq, raw_lp, _ in beams:
                if seq[-1] == eos_id:
                    finished.append((seq, raw_lp))
                    continue

                hidden = model.decoder(torch.tensor(seq, device=device).unsqueeze(0), z[b:b + 1])
                logits = model.predict(hidden)[:, -1] / temperature
                logp = F.log_softmax(logits, dim=-1).squeeze(0)

                topk_val, topk_idx = torch.topk(logp, k=min(beam_width * 2, logp.size(0)))
                candidates = [
                    (tid, tv) for tid, tv in zip(topk_idx.tolist(), topk_val.tolist()) if tid < vocab_size
                ]
                for tid, tv in candidates[:beam_width]:
                    new_seq = seq + [tid]
                    new_raw = raw_lp + tv
                    norm_lp = new_raw / (len(new_seq) ** alpha)
                    new_beams.append((new_seq, new_raw, norm_lp))

            if not new_beams:
                break
            new_beams.sort(key=lambda x: x[2], reverse=True)
            beams = new_beams[:beam_width]

        if finished:
            for seq, raw_lp in finished:
                beams.append((seq, raw_lp, raw_lp / (len(seq) ** alpha)))

        best_seq = max(beams, key=lambda x: x[1] / (len(x[0]) ** len_penalty))[0]
        best_seq = best_seq[1: best_seq.index(eos_id)] if eos_id in best_seq else best_seq[1:]
        out_tokens.append(best_seq)

    return out_tokens


def reconstruct_zmu(
    model,
    dataloader,
    vocab: dict,
    *,
    beam_width: int = 1,
    len_penalty: float = 1.0,
    max_len: int | None = None,
    temperature: float = 0.9,
    alpha: float = 0.7,
):
    model.eval()
    tanis, pairs = [], 0
    device = next(model.parameters()).device

    sos = vocab["[SOS]"]
    eos = vocab["[EOS]"]
    pad = vocab["[PAD]"]
    if max_len is None:
        max_len = dataset.max_len + 2

    for enc_in, _, _, props in dataloader:
        enc_in = enc_in.to(device)
        props = props.to(device)
        smi_mask = enc_in == pad

        with torch.no_grad():
            prop_e = model.input_embedding(props)
            prop_ctx = prop_e.squeeze(1) if prop_e.dim() == 3 else prop_e
            encoded_last = model.encoder(enc_in, smi_mask=smi_mask)
            z = model.to_means(encoded_last + prop_ctx)

            if beam_width == 1:
                dec_in = torch.full((enc_in.size(0), 1), sos, device=device)
                done = torch.zeros(enc_in.size(0), dtype=torch.bool, device=device)
                out_tok = [[] for _ in range(enc_in.size(0))]

                for _ in range(max_len):
                    hidden = model.decoder(dec_in, z)
                    logits = model.predict(hidden)[:, -1] / temperature
                    next_tok = logits.argmax(-1, keepdim=True)
                    dec_in = torch.cat([dec_in, next_tok], dim=1)

                    for i, tok in enumerate(next_tok.squeeze(1).tolist()):
                        if not done[i]:
                            if tok == eos:
                                done[i] = True
                            else:
                                out_tok[i].append(tok)
                    if done.all():
                        break
            else:
                out_tok = _beam_search_decode(
                    model,
                    z,
                    sos,
                    eos,
                    beam_width=beam_width,
                    max_len=max_len,
                    len_penalty=len_penalty,
                    vocab_size=len(vocab),
                    temperature=temperature,
                    alpha=alpha,
                )

        for r_tok, o_tok in zip(enc_in.cpu().tolist(), out_tok):
            ref_sm = tok_ids_to_smiles([t for t in r_tok if t not in (pad,)], index_to_token)
            out_sm = tok_ids_to_smiles([t for t in o_tok if t not in (pad,)], index_to_token)
            if ref_sm and out_sm:
                tanis.append(tanimoto(ref_sm, out_sm))
                pairs += 1

    return np.mean(tanis) if tanis else 0.0, pairs


def eval_iwae_bound(
    model,
    prior,
    sm_enc,
    sm_dec_in,
    sm_dec_out,
    prop,
    smi_mask,
    K=64,
    chunk=8,
):
    model.eval()
    B = sm_enc.size(0)

    prop_e = model.input_embedding(prop)
    prop_ctx = prop_e.squeeze(1) if prop_e.dim() == 3 else prop_e
    enc_last = model.encoder(sm_enc, smi_mask=smi_mask)
    post_ctx = enc_last + prop_ctx
    mu_q = model.to_means(post_ctx).unsqueeze(1)
    q_sigma = model.q_sigma_floor + model.q_sigma_range * torch.sigmoid(model.to_var(post_ctx) / model.q_sigma_temp)
    lv_q = (q_sigma * q_sigma).log().unsqueeze(1)
    q = Normal(mu_q, (0.5 * lv_q).exp())

    prior_input = prop.squeeze(-1) if prop.dim() == 3 else prop
    if prior_input.dim() == 1:
        prior_input = prior_input.unsqueeze(-1)
    mu_p, lv_p = prior(prior_input)
    if mu_p.dim() == 2:
        mu_p = mu_p.unsqueeze(1)
        lv_p = lv_p.unsqueeze(1)
    p = Normal(mu_p, (0.5 * lv_p).exp())

    mask4 = torch.ones((1, B, 1, 1), dtype=mu_q.dtype, device=mu_q.device)

    L_eff_B = (sm_dec_out != PAD_IDX).sum(-1).clamp_min(1).unsqueeze(0)

    log_ws = []
    log_ws_len = []
    log_ws_latent = []

    for k0 in range(0, K, chunk):
        k = min(chunk, K - k0)
        z = q.rsample((k,))
        z2 = z.squeeze(2).reshape(k * B, -1)

        logits = model.predict(model.decoder(sm_dec_in.repeat(k, 1), z2))
        logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        logp = torch.log_softmax(logits, dim=-1)
        tgt = sm_dec_out.repeat(k, 1)
        ll = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        ll.masked_fill_(tgt == PAD_IDX, 0.0)
        log_px = torch.nan_to_num(ll.sum(-1), nan=-1e6, posinf=1e6, neginf=-1e6).view(k, B)

        log_qz = torch.nan_to_num((q.log_prob(z) * mask4).sum((2, 3)), nan=-1e6, posinf=1e6, neginf=-1e6)
        log_pz = torch.nan_to_num((p.log_prob(z) * mask4).sum((2, 3)), nan=-1e6, posinf=1e6, neginf=-1e6)

        log_ws.append(log_px + log_pz - log_qz)
        log_px_norm = log_px / L_eff_B.to(log_px.dtype)
        log_ws_len.append(log_px_norm + log_pz - log_qz)
        log_ws_latent.append(log_pz - log_qz)

    log_ws = torch.nan_to_num(torch.cat(log_ws, 0), nan=-1e6, posinf=1e6, neginf=-1e6).double()
    log_ws_len = torch.nan_to_num(torch.cat(log_ws_len, 0), nan=-1e6, posinf=1e6, neginf=-1e6).double()
    log_ws_latent = torch.nan_to_num(torch.cat(log_ws_latent, 0), nan=-1e6, posinf=1e6, neginf=-1e6).double()

    m = log_ws.max(0, keepdim=True).values
    iwae = (m + (log_ws - m).exp().mean(0).log()).squeeze(0)

    w = (log_ws - m).exp()
    w = w / w.sum(0, keepdim=True).clamp_min(1e-300)
    ess = (w.sum(0) ** 2 / (w ** 2).sum(0)).mean().item()

    m_len = log_ws_len.max(0, keepdim=True).values
    w_len = (log_ws_len - m_len).exp()
    w_len = w_len / w_len.sum(0, keepdim=True).clamp_min(1e-300)
    ess_len = (w_len.sum(0) ** 2 / (w_len ** 2).sum(0)).mean().item()

    m_lat = log_ws_latent.max(0, keepdim=True).values
    w_lat = (log_ws_latent - m_lat).exp()
    w_lat = w_lat / w_lat.sum(0, keepdim=True).clamp_min(1e-300)
    ess_latent = (w_lat.sum(0) ** 2 / (w_lat ** 2).sum(0)).mean().item()

    return iwae.mean().item(), ess, ess_len, ess_latent
