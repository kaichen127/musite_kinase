import os, csv, json, yaml
import torch
import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from contextlib import nullcontext
from model.model import prepare_model
from types import SimpleNamespace

def load_configs(cfg_dict) -> SimpleNamespace:
    def _ns(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: _ns(v) for k, v in d.items()})
        return d
    return _ns(cfg_dict)

def load_checkpoint(path, model):
    ckpt = torch.load(path, map_location="cpu")
    key_options = [
        "model_state_dict", "state_dict", "model",
    ]
    state = None
    if isinstance(ckpt, dict):
        for k in key_options:
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break
    if state is None:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load_checkpoint] missing={len(missing)} unexpected={len(unexpected)}")
    return ckpt

def resolve_autocast_dtype(configs, device):
    """Return torch dtype for autocast or None if disabled/unsupported."""
    if device.type != "cuda":
        return None
    use_mp = bool(getattr(configs, "use_mixed_precision", False))
    if not use_mp:
        return None
    mp = str(getattr(configs, "mixed_precision_dtype", "")).lower()
    if mp in ("bf16", "bfloat16", "bfloat"):
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        print("Warning: bfloat16 not supported on this GPU; falling back to FP32.")
        return None
    if mp in ("fp32", "float32", "single"):
        return torch.float32
    return None

# -------------------------
# Dataset
# -------------------------
class KinaseInferenceDataset(Dataset):
    """
    Expects a dataframe with two columns (configurable):
      - substrate_col (default: 'substrate_sequence')
      - kinase_col    (default: 'kinase_sequence')
    """
    def __init__(self, df, tokenizer, max_length, substrate_col="substrate_sequence", kinase_col="kinase_sequence"):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_length = int(max_length)
        self.sub_col = substrate_col
        self.kin_col = kinase_col

        for col in [self.sub_col, self.kin_col]:
            if col not in self.df.columns:
                raise ValueError(f"Input CSV must include '{col}'. Found: {list(self.df.columns)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sub = str(row[self.sub_col])
        kin = str(row[self.kin_col])

        sub_enc = self.tok(
            sub,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        kin_enc = self.tok(
            kin,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        return {
            "substrate_input_ids": sub_enc["input_ids"].squeeze(0),
            "substrate_attention_mask": sub_enc["attention_mask"].squeeze(0),
            "kinase_input_ids": kin_enc["input_ids"].squeeze(0),
            "kinase_attention_mask": kin_enc["attention_mask"].squeeze(0),
            "substrate_len": len(sub),
        }


# -------------------------
# Inference
# -------------------------
@torch.no_grad()
def run_inference(model, tokenizer, device, df, configs):
    seq_len        = int(getattr(configs, "sequence_length", getattr(configs, "max_sequence_length", 1280)))
    batch_size     = int(getattr(configs, "batch_size", 8))
    pred_threshold = float(getattr(configs, "prediction_threshold", 0.5))
    substrate_col  = getattr(configs, "substrate_col", "substrate_sequence")
    kinase_col     = getattr(configs, "kinase_col", "kinase_sequence")
    out_probs      = bool(getattr(configs, "output_binding_probs", True))
    index_base     = int(getattr(configs, "positive_index_starts_at", 1))

    ac_dtype = resolve_autocast_dtype(configs, device)
    if ac_dtype is not None:
        print(f"[Precision] Mixed precision enabled: {ac_dtype}")
    else:
        print(f"[Precision] Running in full precision: torch.float32")

    dataset = KinaseInferenceDataset(
        df=df,
        tokenizer=tokenizer,
        max_length=seq_len,
        substrate_col=substrate_col,
        kinase_col=kinase_col,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    n = len(df)
    all_pred_json   = [""] * n
    all_pos_indices = [""] * n
    all_probs_json  = [""] * n if out_probs else None

    model.eval()
    offset = 0
    for batch in tqdm.tqdm(loader, desc="Inference"):
        s_ids   = batch["substrate_input_ids"].to(device)
        s_mask  = batch["substrate_attention_mask"].to(device)
        k_ids   = batch["kinase_input_ids"].to(device)
        k_mask  = batch["kinase_attention_mask"].to(device)
        s_lens  = batch["substrate_len"]

        ac_ctx = (
            autocast(device_type="cuda", dtype=ac_dtype)
            if (device.type == "cuda" and ac_dtype is not None)
            else nullcontext()
        )
        with ac_ctx:
            logits = model(
                substrate_input_ids=s_ids,
                substrate_attention_mask=s_mask,
                kinase_input_ids=k_ids,
                kinase_attention_mask=k_mask,
            )
        probs = torch.sigmoid(logits).squeeze(-1).to(torch.float32)
        preds = (probs > pred_threshold).to(torch.int32)

        probs_np = probs.cpu().numpy()
        preds_np = preds.cpu().numpy()

        B = s_ids.size(0)
        for b in range(B):
            L_true = int(s_lens[b])
            pred_list = preds_np[b][:L_true].tolist()
            all_pred_json[offset + b] = json.dumps(pred_list)

            pos_idx = [i + index_base for i, val in enumerate(pred_list) if val == 1]
            all_pos_indices[offset + b] = json.dumps(pos_idx)

            if out_probs:
                probs_list = [float(f"{p:.4f}") for p in probs_np[b][:L_true]]
                all_probs_json[offset + b] = json.dumps(probs_list)

        offset += B

    out_df = df.copy()
    out_df["predictions"] = all_pred_json
    out_df["positive_indices"] = all_pos_indices
    if out_probs:
        out_df["binding_probabilities"] = all_probs_json
    return out_df


# -------------------------
# Main
# -------------------------
def main():
    with open("./configs/config.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)
    configs = load_configs(cfg_dict)

    dev_str = getattr(configs, "device_type", "cuda")
    device = torch.device(dev_str if torch.cuda.is_available() and "cuda" in dev_str else "cpu")
    print("Using device:", device)

    tokenizer, model = prepare_model(configs)

    ckpt = getattr(configs, "checkpoint_path", None)
    if ckpt:
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        print(f"Loading checkpoint: {ckpt}")
        _ = load_checkpoint(ckpt, model)

    model.to(device)

    input_csv = getattr(configs, "input_csv_path", None)
    output_csv = getattr(configs, "output_csv_path", None)
    if not input_csv or not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not output_csv:
        raise ValueError("Config must define 'output_csv_path'")

    df = pd.read_csv(input_csv)

    out_df = run_inference(model, tokenizer, device, df, configs)

    out_df.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL)
    print(f"Saved predictions to: {output_csv}")


if __name__ == "__main__":
    main()
