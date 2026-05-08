from utils.utils import *
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent
sys.path.append(str(PROJECT_ROOT))


def _psmiles_core(ps: str) -> str:
    """{...}n / (... )n 형태의 PSMILES 래퍼 제거."""
    s = ps.strip()
    s = re.sub(r"^\{\s*(.+?)\s*\}\s*(\d+(?:-\d+)?)?$", r"\1", s)
    s = re.sub(r"^\(\s*(.+?)\s*\)\s*(\d+(?:-\d+)?)?$", r"\1", s)
    return s


class PI1MPretrainDataset(Dataset):
    """
    PI1M p-SMILES corpus -> SELFIES token LM pretraining dataset.
    Returns tuple shape:
      (enc, dec_in, dec_out, cond_scalar)
    cond_scalar is a constant zero tensor [1, 1] for pretraining compatibility.
    """

    CACHE_VERSION = 2

    def __init__(self, csv_path, cache_dir="cache", smiles_col="SMILES", max_rows=None):
        self.csv_path = Path(csv_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.smiles_col = str(smiles_col)
        self.max_rows = None if max_rows is None else int(max_rows)

        max_rows_tag = "all" if self.max_rows is None else f"n{self.max_rows}"
        cache_prefix = f"{self.csv_path.stem}_pi1m_pretrain_{max_rows_tag}"
        cache_pt = self.cache_dir / f"{cache_prefix}.pt"
        cache_meta = self.cache_dir / f"{cache_prefix}.pkl"

        if cache_pt.exists() and cache_meta.exists():
            if self._load_cache(cache_pt, cache_meta):
                return

        self._build_from_csv()
        self._save_cache(cache_pt, cache_meta)

    def _save_cache(self, data_path, meta_path):
        torch.save(
            {
                "SMILES_dec_input": self.SMILES_dec_input,
                "SMILES_dec_output": self.SMILES_dec_output,
                "properties": self.properties,
            },
            data_path,
        )
        with open(meta_path, "wb") as f:
            pickle.dump(
                {
                    "cache_version": self.CACHE_VERSION,
                    "vocab": self.vocab,
                    "max_len": self.max_len,
                    "num_rows_read": self.num_rows_read,
                    "num_rows_valid": self.num_rows_valid,
                    "smiles_col": self.smiles_col,
                    "cond_labels": self.cond_labels,
                    "cond_bin_edges_norm": self.cond_bin_edges_norm,
                },
                f,
            )

    def _load_cache(self, data_path, meta_path):
        try:
            blob = torch.load(data_path, map_location="cpu")
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            if meta.get("cache_version") != self.CACHE_VERSION:
                return False
            self.SMILES_dec_input = blob["SMILES_dec_input"].long()
            self.SMILES_dec_output = blob["SMILES_dec_output"].long()
            self.properties = blob["properties"].float()
            # Alias to avoid duplicating tensor memory; pretrain loop does not use enc.
            self.SMILES_enc = self.SMILES_dec_input

            self.vocab = meta["vocab"]
            self.max_len = int(meta["max_len"])
            self.num_rows_read = int(meta.get("num_rows_read", len(self.SMILES_dec_input)))
            self.num_rows_valid = int(meta.get("num_rows_valid", len(self.SMILES_dec_input)))
            self.smiles_col = str(meta.get("smiles_col", self.smiles_col))
            self.cond_labels = [str(x) for x in meta.get("cond_labels", ["SCALAR"])]
            self.cond_bin_edges_norm = [float(x) for x in meta.get("cond_bin_edges_norm", [])]
            self.vocab_size = len(self.vocab)
            self.num_data = self.SMILES_dec_input.shape[0]
            return True
        except Exception:
            return False

    def _build_from_csv(self):
        read_kwargs = {"usecols": [self.smiles_col]}
        if self.max_rows is not None:
            read_kwargs["nrows"] = self.max_rows
        df = pd.read_csv(self.csv_path, **read_kwargs)
        smiles_arr = df[self.smiles_col].astype(str).values
        self.num_rows_read = len(smiles_arr)

        sf_tokens = []
        dropped = 0
        for sm in tqdm(smiles_arr, desc="PI1M tokenize", leave=False):
            try:
                ps = PS(sm).canonicalize.psmiles
                psf = sfp.encoder_psmiles(_psmiles_core(ps), strict=False)
                toks = list(sfp.split_selfies(psf))
                if len(toks) == 0:
                    dropped += 1
                    continue
                sf_tokens.append(toks)
            except Exception:
                dropped += 1

        if not sf_tokens:
            raise RuntimeError("PI1M preprocessing produced zero valid sequences.")

        self.num_rows_valid = len(sf_tokens)
        self.num_rows_dropped = dropped
        self.max_len = max(len(t) for t in sf_tokens) + 1

        corpus = [tok for seq in sf_tokens for tok in seq] + ["[SOS]", "[EOS]", "[PAD]"]
        self.vocab = {tok: i for i, tok in enumerate(sorted(set(corpus)))}
        self.vocab_size = len(self.vocab)
        self.num_data = len(sf_tokens)

        pad_idx = self.vocab["[PAD]"]
        sos_idx = self.vocab["[SOS]"]
        eos_idx = self.vocab["[EOS]"]

        dec_in = torch.full((self.num_data, self.max_len), pad_idx, dtype=torch.long)
        dec_out = torch.full_like(dec_in, pad_idx)

        for i, seq in enumerate(tqdm(sf_tokens, desc="PI1M tensorize", leave=False)):
            seq_ids = [self.vocab[t] for t in seq]
            seq_len = len(seq_ids)
            dec_in[i, 0] = sos_idx
            dec_in[i, 1: seq_len + 1] = torch.tensor(seq_ids, dtype=torch.long)
            dec_out[i, :seq_len] = torch.tensor(seq_ids, dtype=torch.long)
            dec_out[i, seq_len] = eos_idx

        self.SMILES_dec_input = dec_in
        self.SMILES_dec_output = dec_out
        self.properties = torch.zeros((self.num_data, 1, 1), dtype=torch.float32)
        self.SMILES_enc = self.SMILES_dec_input  # alias
        self.cond_labels = ["SCALAR"]
        self.cond_bin_edges_norm = []

    def __getitem__(self, i):
        return (
            self.SMILES_enc[i],
            self.SMILES_dec_input[i],
            self.SMILES_dec_output[i],
            self.properties[i],
        )

    def __len__(self):
        return self.SMILES_dec_input.shape[0]


def build_pi1m_pretrain_loader(
    csv_path,
    batch_size=256,
    cache_dir=None,
    smiles_col="SMILES",
    max_rows=None,
    shuffle=True,
):
    if cache_dir is None:
        cache_dir = PROJECT_ROOT / "checkpoints" / "cache"
    ds = PI1MPretrainDataset(
        csv_path=csv_path,
        cache_dir=cache_dir,
        smiles_col=smiles_col,
        max_rows=max_rows,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return ds, loader
