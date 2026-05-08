from __future__ import annotations

import os
import sys
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from torch.utils.data import Subset

from utils.utils import *

_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent
sys.path.append(str(PROJECT_ROOT))

_ALIAS = {}


def _current_variant() -> str:
    raw = os.environ.get("MODELS_VARIANT", "LSTM_CVAE").strip() or "LSTM_CVAE"
    return _ALIAS.get(raw, raw)


def _variant_config(variant: str) -> dict:
    if variant == "Encoder_Only":
        return {
            "token_mode": "selfies",
            "property_mode": "log_minmax_zscore",
            "cache_version": 9,
        }
    if variant == "Encoder_Only_PSMILES":
        return {
            "token_mode": "psmiles",
            "property_mode": "log_minmax_zscore",
            "cache_version": 10,
        }
    if variant in {"LSTM_CVAE_PSMILES", "TransCVAE_PSMILES"}:
        return {
            "token_mode": "psmiles",
            "property_mode": "log_minmax_zscore",
            "cache_version": 5,
        }
    return {
        "token_mode": "selfies",
        "property_mode": "log_minmax_zscore",
        "cache_version": 4,
    }


def _psmiles_core(ps: str) -> str:
    """{...}n / (... )n 형태의 PSMILES 래퍼 제거."""
    s = ps.strip()
    s = re.sub(r"^\{\s*(.+?)\s*\}\s*(\d+(?:-\d+)?)?$", r"\1", s)
    s = re.sub(r"^\(\s*(.+?)\s*\)\s*(\d+(?:-\d+)?)?$", r"\1", s)
    return s


SYSTEM_GROUP_COLS = [
    "SMILES",
    "Molality",
    "Monomer Molecular Weight",
    "Degree of Polymerization",
    "Density",
]
TARGET_COL = "CONDUCTIVITY"
SPLIT_N_FOLDS = int(os.environ.get("MODELS_SPLIT_N_FOLDS", "5"))
SPLIT_N_BINS = int(os.environ.get("MODELS_SPLIT_N_BINS", "6"))
SPLIT_SEED = int(os.environ.get("MODELS_SPLIT_SEED", "42"))
TEST_FOLD = int(os.environ.get("MODELS_TEST_FOLD", "0"))
VAL_FOLD = int(os.environ.get("MODELS_VAL_FOLD", "1"))
BATCH_SIZE = int(os.environ.get("MODELS_BATCH_SIZE", "256"))
FULL_DATA_TRAINING = os.environ.get("MODELS_FULL_DATA_TRAINING", "0") == "1"


def _require_columns(df: pd.DataFrame, cols: list[str], context: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")


def _system_ids(df: pd.DataFrame) -> pd.Series:
    _require_columns(df, SYSTEM_GROUP_COLS, "system_id split")
    return df[SYSTEM_GROUP_COLS].astype(str).agg("|".join, axis=1)


def _conductivity_bins(df: pd.DataFrame, n_bins: int = SPLIT_N_BINS) -> np.ndarray:
    _require_columns(df, [TARGET_COL], "conductivity stratification")
    conductivity = pd.to_numeric(df[TARGET_COL], errors="raise").to_numpy(dtype=float)
    if np.any(~np.isfinite(conductivity)):
        raise ValueError(f"{TARGET_COL} contains non-finite values")
    if np.any(conductivity <= 0):
        raise ValueError(f"{TARGET_COL} must be positive for log10 stratification")

    y_log = np.log10(conductivity)
    # Quantile bins keep the high-conductivity tail represented without using row-random splits.
    bins = pd.qcut(y_log, q=n_bins, labels=False, duplicates="drop")
    bins = np.asarray(bins, dtype=int)
    if bins.size != len(df):
        raise RuntimeError("Failed to build conductivity stratification bins")
    return bins


def make_stratified_group_folds(
    df: pd.DataFrame,
    n_splits: int = SPLIT_N_FOLDS,
    n_bins: int = SPLIT_N_BINS,
    seed: int = SPLIT_SEED,
) -> tuple[np.ndarray, pd.Series, np.ndarray]:
    if n_splits < 3:
        raise ValueError("Need at least 3 folds to build train/val/test splits")

    groups = _system_ids(df)
    y_bin = _conductivity_bins(df, n_bins=n_bins)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold = np.full(len(df), -1, dtype=int)
    for fold_id, (_, holdout_idx) in enumerate(cv.split(df, y_bin, groups=groups)):
        fold[holdout_idx] = fold_id

    if np.any(fold < 0):
        raise RuntimeError("StratifiedGroupKFold did not assign every row to a fold")
    return fold, groups, y_bin


def _shared_group_count(groups: pd.Series, left_idx: np.ndarray, right_idx: np.ndarray) -> int:
    left = set(groups.iloc[left_idx].tolist())
    right = set(groups.iloc[right_idx].tolist())
    return len(left.intersection(right))


def make_train_val_test_split(
    df: pd.DataFrame,
    *,
    test_fold: int = TEST_FOLD,
    val_fold: int | None = VAL_FOLD,
    n_splits: int = SPLIT_N_FOLDS,
    n_bins: int = SPLIT_N_BINS,
    seed: int = SPLIT_SEED,
) -> dict:
    if val_fold is not None and test_fold == val_fold:
        raise ValueError("MODELS_TEST_FOLD and MODELS_VAL_FOLD must be different")
    if not (0 <= test_fold < n_splits):
        raise ValueError("MODELS_TEST_FOLD must be a valid fold id")
    if val_fold is not None and not (0 <= val_fold < n_splits):
        raise ValueError("MODELS_TEST_FOLD/MODELS_VAL_FOLD must be valid fold ids")

    fold, groups, y_bin = make_stratified_group_folds(df, n_splits=n_splits, n_bins=n_bins, seed=seed)
    test_idx = np.where(fold == test_fold)[0]
    val_idx = np.array([], dtype=int) if val_fold is None else np.where(fold == val_fold)[0]
    if val_fold is None:
        train_idx = np.where(fold != test_fold)[0]
    else:
        train_idx = np.where((fold != test_fold) & (fold != val_fold))[0]
    trainval_idx = np.where(fold != test_fold)[0]

    leak_counts = {
        "train_val": _shared_group_count(groups, train_idx, val_idx),
        "train_test": _shared_group_count(groups, train_idx, test_idx),
        "val_test": _shared_group_count(groups, val_idx, test_idx),
    }
    if any(v != 0 for v in leak_counts.values()):
        raise RuntimeError(f"system_id leakage detected across splits: {leak_counts}")

    split_label = np.full(len(df), "train", dtype=object)
    if len(val_idx) > 0:
        split_label[val_idx] = "val"
    split_label[test_idx] = "test"

    return {
        "fold": fold,
        "groups": groups,
        "y_bin": y_bin,
        "split_label": split_label,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "trainval_idx": trainval_idx,
        "leak_counts": leak_counts,
    }


def _split_summary(df: pd.DataFrame, split_info: dict) -> pd.DataFrame:
    groups = split_info["groups"]
    y_bin = split_info["y_bin"]
    rows = []
    for label, idx in [
        ("train", split_info["train_idx"]),
        ("val", split_info["val_idx"]),
        ("test", split_info["test_idx"]),
        ("trainval", split_info["trainval_idx"]),
        ("full", np.arange(len(df))),
    ]:
        bin_counts = np.bincount(y_bin[idx], minlength=int(y_bin.max()) + 1)
        row = {
            "split": label,
            "n_rows": int(len(idx)),
            "n_systems": int(groups.iloc[idx].nunique()),
            "cond_min": float(pd.to_numeric(df.iloc[idx][TARGET_COL]).min()) if len(idx) else np.nan,
            "cond_max": float(pd.to_numeric(df.iloc[idx][TARGET_COL]).max()) if len(idx) else np.nan,
        }
        row.update({f"bin_{i}": int(v) for i, v in enumerate(bin_counts)})
        rows.append(row)
    return pd.DataFrame(rows)


def _make_generator(offset: int = 0) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(SPLIT_SEED + int(offset))
    return generator


def _split_tag(test_fold: int | None = TEST_FOLD, val_fold: int | None = VAL_FOLD, suffix: str = "") -> str:
    val_part = "none" if val_fold is None else str(int(val_fold))
    tag = (
        f"sgkf_s{SPLIT_SEED}_k{SPLIT_N_FOLDS}_b{SPLIT_N_BINS}"
        f"_test{int(test_fold)}_val{val_part}_trainfit"
    )
    if suffix:
        safe_suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(suffix).strip("_"))
        tag = f"{tag}_{safe_suffix}"
    return tag


def build_split_context(
    *,
    csv_path: Path | str | None = None,
    variant_config: dict | None = None,
    cache_dir: Path | str | None = None,
    test_fold: int = TEST_FOLD,
    val_fold: int | None = VAL_FOLD,
    tag_suffix: str = "",
    batch_size: int = BATCH_SIZE,
    write_split_files: bool | None = None,
) -> dict:
    csv_path = Polymers if csv_path is None else Path(csv_path)
    cache_dir = CACHE_DIR if cache_dir is None else Path(cache_dir)
    variant_config = _cfg if variant_config is None else dict(variant_config)

    df = pd.read_csv(csv_path)
    local_split_info = make_train_val_test_split(df, test_fold=test_fold, val_fold=val_fold)
    local_train_idx = local_split_info["train_idx"]
    local_val_idx = local_split_info["val_idx"]
    local_test_idx = local_split_info["test_idx"]
    local_trainval_idx = local_split_info["trainval_idx"]

    tag = _split_tag(test_fold=test_fold, val_fold=val_fold, suffix=tag_suffix)
    local_assignment = df.copy()
    local_assignment["system_id"] = local_split_info["groups"].to_numpy()
    local_assignment["conductivity_bin"] = local_split_info["y_bin"]
    local_assignment["fold"] = local_split_info["fold"]
    local_assignment["split"] = local_split_info["split_label"]
    local_summary = _split_summary(df, local_split_info)

    if write_split_files is None:
        write_split_files = os.environ.get("MODELS_WRITE_SPLIT_FILES", "1") != "0"
    if write_split_files:
        SPLIT_DIR.mkdir(parents=True, exist_ok=True)
        local_assignment.to_csv(SPLIT_DIR / f"{tag}_assignment.csv", index=False)
        local_summary.to_csv(SPLIT_DIR / f"{tag}_summary.csv", index=False)

    local_dataset = load_data(
        csv_path,
        cache_dir=cache_dir,
        normalizer_fit_indices=local_train_idx,
        cache_suffix=tag,
        **variant_config,
    )

    local_train_dataset = Subset(local_dataset, local_train_idx.tolist())
    local_val_dataset = Subset(local_dataset, local_val_idx.tolist())
    local_test_dataset = Subset(local_dataset, local_test_idx.tolist())
    local_trainval_dataset = Subset(local_dataset, local_trainval_idx.tolist())

    return {
        "dataset": local_dataset,
        "train_dataset": local_train_dataset,
        "val_dataset": local_val_dataset,
        "test_dataset": local_test_dataset,
        "trainval_dataset": local_trainval_dataset,
        "train_dataloader": DataLoader(
            local_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=_make_generator(100 + (0 if val_fold is None else int(val_fold))),
        ),
        "val_dataloader": DataLoader(local_val_dataset, batch_size=batch_size, shuffle=False, drop_last=False),
        "test_dataloader": DataLoader(local_test_dataset, batch_size=batch_size, shuffle=False, drop_last=False),
        "trainval_dataloader": DataLoader(
            local_trainval_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=_make_generator(200 + (0 if val_fold is None else int(val_fold))),
        ),
        "eval_dataloader": DataLoader(local_dataset, batch_size=batch_size, shuffle=False, drop_last=False),
        "train_indices": local_train_idx,
        "val_indices": local_val_idx,
        "test_indices": local_test_idx,
        "trainval_indices": local_trainval_idx,
        "fold_assignment": local_assignment,
        "split_summary": local_summary,
        "split_info": local_split_info,
        "split_tag": tag,
    }


class load_data(Dataset):
    def __init__(
        self,
        csv_path,
        cache_dir="cache",
        token_mode: str = "selfies",
        property_mode: str = "log_minmax_zscore",
        cache_version: int = 1,
        normalizer_fit_indices=None,
        cache_suffix: str = "",
    ):
        self.token_mode = str(token_mode)
        self.property_mode = str(property_mode)
        self.CACHE_VERSION = int(cache_version)
        self.normalizer_fit_indices = (
            None
            if normalizer_fit_indices is None
            else np.asarray(normalizer_fit_indices, dtype=int)
        )

        csv_path = Path(csv_path)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_tag = f"{self.token_mode}_{self.property_mode}_v{self.CACHE_VERSION}"
        if cache_suffix:
            safe_suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(cache_suffix).strip("_"))
            cache_tag = f"{cache_tag}_{safe_suffix}"
        cache_pt = cache_dir / f"{csv_path.stem}_{cache_tag}_data.pt"
        cache_meta = cache_dir / f"{csv_path.stem}_{cache_tag}_meta.pkl"

        if cache_pt.exists() and cache_meta.exists():
            if self._load_cache(cache_pt, cache_meta):
                return

        self._build_from_csv(csv_path)
        self._save_cache(cache_pt, cache_meta)

    def _save_cache(self, data_path, meta_path):
        torch.save(
            {
                "SMILES_enc": self.SMILES_enc,
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
                    "token_mode": self.token_mode,
                    "property_mode": self.property_mode,
                    "vocab": self.vocab,
                    "max_len": self.max_len,
                    "mean_vec": self.mean_vec,
                    "std_vec": self.std_vec,
                    "cond_labels": self.cond_labels,
                    "cond_bin_edges_norm": self.cond_bin_edges_norm,
                    "normalizer_fit_rows": None
                    if self.normalizer_fit_indices is None
                    else int(len(self.normalizer_fit_indices)),
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
            if str(meta.get("token_mode")) != self.token_mode:
                return False
            if str(meta.get("property_mode")) != self.property_mode:
                return False

            self.SMILES_enc = blob["SMILES_enc"]
            self.SMILES_dec_input = blob["SMILES_dec_input"]
            self.SMILES_dec_output = blob["SMILES_dec_output"]
            self.properties = blob["properties"].float()

            self.vocab = meta["vocab"]
            self.max_len = int(meta["max_len"])
            self.mean_vec = np.asarray(meta.get("mean_vec", [0.0]))
            self.std_vec = np.asarray(meta.get("std_vec", [1.0]))
            self.cond_labels = [str(x) for x in meta.get("cond_labels", ["SCALAR"])]
            self.cond_bin_edges_norm = [float(x) for x in meta.get("cond_bin_edges_norm", [])]

            self.vocab_size = len(self.vocab)
            self.num_data = self.SMILES_enc.shape[0]
            return True
        except Exception:
            return False

    def _tokenize(self, smiles_arr):
        psmiles = [PS(smiles).canonicalize.psmiles for smiles in smiles_arr]
        if self.token_mode == "psmiles":
            return [split_psmiles_tokens(_psmiles_core(ps)) for ps in psmiles]
        if self.token_mode == "selfies":
            pselfies = [sfp.encoder_psmiles(_psmiles_core(ps), strict=False) for ps in psmiles]
            return [list(sfp.split_selfies(psf)) for psf in pselfies]
        raise ValueError(f"Unsupported token_mode: {self.token_mode}")

    def _normalize_property(self, conductivity):
        fit_idx = self.normalizer_fit_indices
        fit_values = conductivity if fit_idx is None else conductivity[fit_idx]

        if self.property_mode == "minmax":
            scaler = MinMaxScaler().fit(fit_values)
            cond = scaler.transform(conductivity)
            cond_fit = scaler.transform(fit_values)
            self.mean_vec = np.asarray(cond_fit.mean(axis=0))
            self.std_vec = np.asarray(cond_fit.std(axis=0) + 1e-8)
            self.cond_labels = ["SCALAR"]
            self.cond_bin_edges_norm = []
            return cond

        if self.property_mode == "log_minmax_zscore":
            pipeline = Pipeline(
                steps=[
                    ("log", FunctionTransformer(np.log1p, validate=True)),
                    ("minmax", MinMaxScaler()),
                ]
            ).fit(fit_values)
            cond = pipeline.transform(conductivity)
            cond_fit = pipeline.transform(fit_values)
            self.mean_vec = cond_fit.mean(axis=0)
            self.std_vec = cond_fit.std(axis=0) + 1e-8
            self.cond_labels = ["SCALAR"]
            self.cond_bin_edges_norm = []
            return (cond - self.mean_vec) / self.std_vec

        raise ValueError(f"Unsupported property_mode: {self.property_mode}")

    def _build_from_csv(self, path):
        self.raw = pd.read_csv(path)

        smiles_arr = self.raw.iloc[:, 1].astype(str).values
        conductivity = self.raw.iloc[:, 6].values.reshape(-1, 1)
        cond_norm = self._normalize_property(conductivity)
        tokens = self._tokenize(smiles_arr)

        self.max_len = max(len(t) for t in tokens) + 1

        corpus = [tok for seq in tokens for tok in seq] + ["[SOS]", "[EOS]", "[PAD]"]
        vocab = {tok: i for i, tok in enumerate(sorted(set(corpus)))}
        num_data = len(tokens)

        enc = torch.full((num_data, self.max_len), vocab["[PAD]"], dtype=torch.long)
        dec_in = torch.full_like(enc, vocab["[PAD]"])
        dec_out = torch.full_like(enc, vocab["[PAD]"])

        for i, seq in enumerate(tokens):
            for j, tok in enumerate(seq):
                enc[i, j] = vocab[tok]

            dec_in[i, 0] = vocab["[SOS]"]
            dec_in[i, 1 : len(seq) + 1] = torch.tensor([vocab[t] for t in seq], dtype=torch.long)

            dec_out[i, : len(seq)] = torch.tensor([vocab[t] for t in seq], dtype=torch.long)
            dec_out[i, len(seq)] = vocab["[EOS]"]

        self.SMILES_enc = enc
        self.SMILES_dec_input = dec_in
        self.SMILES_dec_output = dec_out
        self.properties = torch.tensor(cond_norm, dtype=torch.float32).unsqueeze(-1)  # [N,1,1]

        self.num_data = num_data
        self.vocab = vocab
        self.vocab_size = len(vocab)

    def __getitem__(self, i):
        return (
            self.SMILES_enc[i],
            self.SMILES_dec_input[i],
            self.SMILES_dec_output[i],
            self.properties[i],
        )

    def __len__(self):
        return self.SMILES_enc.shape[0]

    def vocab_len(self):
        return self.vocab_size


_variant = _current_variant()
_cfg = _variant_config(_variant)
Polymers = PROJECT_ROOT / "data/simulation-trajectory-aggregate_aligned.csv"
CACHE_DIR = PROJECT_ROOT / "checkpoints" / "cache"
split_df = pd.read_csv(Polymers)
split_info = make_train_val_test_split(split_df)
train_indices = split_info["train_idx"]
val_indices = split_info["val_idx"]
test_indices = split_info["test_idx"]
trainval_indices = split_info["trainval_idx"]
fold_assignment = split_df.copy()
fold_assignment["system_id"] = split_info["groups"].to_numpy()
fold_assignment["conductivity_bin"] = split_info["y_bin"]
fold_assignment["fold"] = split_info["fold"]
fold_assignment["split"] = split_info["split_label"]
split_summary = _split_summary(split_df, split_info)

SPLIT_TAG = (
    f"sgkf_s{SPLIT_SEED}_k{SPLIT_N_FOLDS}_b{SPLIT_N_BINS}"
    f"_test{TEST_FOLD}_val{VAL_FOLD}_trainfit"
)
SPLIT_DIR = PROJECT_ROOT / "checkpoints" / "splits"
if os.environ.get("MODELS_WRITE_SPLIT_FILES", "1") != "0":
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    fold_assignment.to_csv(SPLIT_DIR / f"{SPLIT_TAG}_assignment.csv", index=False)
    split_summary.to_csv(SPLIT_DIR / f"{SPLIT_TAG}_summary.csv", index=False)

dataset = load_data(
    Polymers,
    cache_dir=CACHE_DIR,
    normalizer_fit_indices=None if FULL_DATA_TRAINING else train_indices,
    cache_suffix=f"{SPLIT_TAG}_fullfit" if FULL_DATA_TRAINING else SPLIT_TAG,
    **_cfg,
)

if FULL_DATA_TRAINING:
    train_dataset = dataset
    val_dataset = dataset
    test_dataset = dataset
    trainval_dataset = dataset
else:
    train_dataset = Subset(dataset, train_indices.tolist())
    val_dataset = Subset(dataset, val_indices.tolist())
    test_dataset = Subset(dataset, test_indices.tolist())
    trainval_dataset = Subset(dataset, trainval_indices.tolist())

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False,
    generator=_make_generator(0),
)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
trainval_dataloader = DataLoader(
    trainval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False,
    generator=_make_generator(1),
)
eval_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

__all__ = [
    "PROJECT_ROOT",
    "load_data",
    "dataset",
    "train_dataset",
    "val_dataset",
    "test_dataset",
    "trainval_dataset",
    "train_dataloader",
    "val_dataloader",
    "test_dataloader",
    "trainval_dataloader",
    "eval_dataloader",
    "train_indices",
    "val_indices",
    "test_indices",
    "trainval_indices",
    "fold_assignment",
    "split_summary",
    "split_info",
    "build_split_context",
    "make_train_val_test_split",
    "make_stratified_group_folds",
    "SYSTEM_GROUP_COLS",
    "TARGET_COL",
    "SPLIT_TAG",
    "SPLIT_N_FOLDS",
    "SPLIT_N_BINS",
    "SPLIT_SEED",
    "TEST_FOLD",
    "VAL_FOLD",
    "FULL_DATA_TRAINING",
]
