#!/usr/bin/env python3
import os, argparse
from types import SimpleNamespace
from typing import Dict, Any, List, Tuple
from contextlib import nullcontext

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from esm.models.esmc import ESMC

# ============================================================
# 0) SETTINGS
# ============================================================
CONFIG = SimpleNamespace(
    noise_std=0.00,
    model=SimpleNamespace(
        backbone_type="ESMC",
        model_name="esmc_600m",
        hidden_size=1152,
        dtype=torch.bfloat16,
        structure_aware=False,
        freeze_backbone=True,
        freeze_embeddings=True,
        num_unfrozen_layers=6,
        backbone_dropout_rate=0.3,
        classifier_dropout_rate=0.5,
        last_state_dropout_rate=0.0,
        decoder_head=SimpleNamespace(
            num_heads=8,
            num_layers=6,
            dim_feedforward=640,
            dropout=0.3,
        ),
    ),
    dataset=SimpleNamespace(
        max_sequence_length=1280
    ),
)

# ============================================================
# 1) MODEL
# ============================================================
class KinaseModel(nn.Module):
    def __init__(self, configs, num_labels=1):
        super().__init__()

        base_model_name = configs.model.model_name
        hidden_size = configs.model.hidden_size
        dtype = configs.model.dtype
        freeze_backbone = configs.model.freeze_backbone
        freeze_embeddings = configs.model.freeze_embeddings
        num_unfrozen_layers = configs.model.num_unfrozen_layers
        classifier_dropout_rate = configs.model.classifier_dropout_rate
        backbone_dropout_rate = configs.model.backbone_dropout_rate
        esm_to_decoder_dropout_rate = configs.model.last_state_dropout_rate
        self.noise_std = configs.noise_std

        self.backbone_type = configs.model.backbone_type.upper()
        cache_dir = getattr(configs.model, "cache_dir", None)

        if self.backbone_type == "ESMC":
            self.base_model = ESMC.from_pretrained(base_model_name)
        else:
            config = AutoConfig.from_pretrained(base_model_name)
            config.torch_dtype = dtype
            if cache_dir and os.path.isdir(cache_dir):
                self.base_model = AutoModel.from_pretrained(base_model_name, config=config, cache_dir=cache_dir)
            else:
                self.base_model = AutoModel.from_pretrained(base_model_name, config=config)

        if freeze_backbone:
            if hasattr(self.base_model, "encoder") and hasattr(self.base_model.encoder, "layer"):
                layers = self.base_model.encoder.layer  # ESM2
            elif hasattr(self.base_model, "transformer") and hasattr(self.base_model.transformer, "blocks"):
                layers = self.base_model.transformer.blocks  # ESMC
            else:
                layers = []
            num_total_layers = len(layers)
            if freeze_embeddings:
                for p in self.base_model.parameters(): p.requires_grad = False
                for i, layer in enumerate(layers):
                    if i >= num_total_layers - num_unfrozen_layers:
                        for p in layer.parameters(): p.requires_grad = True
                        if hasattr(layer, "attention"):
                            layer.attention.self.dropout = nn.Dropout(backbone_dropout_rate)
                            layer.attention.output.dropout = nn.Dropout(backbone_dropout_rate)
                            layer.intermediate.dropout = nn.Dropout(backbone_dropout_rate)
                            layer.output.dropout = nn.Dropout(backbone_dropout_rate)
            else:
                for i, layer in enumerate(layers):
                    if i < num_total_layers - num_unfrozen_layers:
                        for p in layer.parameters(): p.requires_grad = False
                    else:
                        if hasattr(layer, "attention"):
                            layer.attention.self.dropout = nn.Dropout(backbone_dropout_rate)
                            layer.attention.output.dropout = nn.Dropout(backbone_dropout_rate)
                            layer.intermediate.dropout = nn.Dropout(backbone_dropout_rate)
                            layer.output.dropout = nn.Dropout(backbone_dropout_rate)

        self.encoder_to_decoder_dropout = nn.Dropout(esm_to_decoder_dropout_rate)

        num_heads = configs.model.decoder_head.num_heads
        num_layers = configs.model.decoder_head.num_layers
        dim_feedforward = configs.model.decoder_head.dim_feedforward
        dropout = configs.model.decoder_head.dropout
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(classifier_dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, substrate_input_ids, substrate_attention_mask, kinase_input_ids, kinase_attention_mask):
        if self.backbone_type != "ESMC":
            s_out = self.base_model(input_ids=substrate_input_ids, attention_mask=substrate_attention_mask)
            s_repr = self.encoder_to_decoder_dropout(s_out.last_hidden_state)
            k_out = self.base_model(input_ids=kinase_input_ids, attention_mask=kinase_attention_mask)
            k_repr = self.encoder_to_decoder_dropout(k_out.last_hidden_state)
        else:
            s_out = self.base_model(sequence_tokens=substrate_input_ids); s_repr = self.encoder_to_decoder_dropout(s_out.embeddings)
            k_out = self.base_model(sequence_tokens=kinase_input_ids);   k_repr = self.encoder_to_decoder_dropout(k_out.embeddings)

        memory_key_padding_mask = (kinase_attention_mask == 0)
        dec = self.transformer_decoder(tgt=s_repr, memory=k_repr, memory_key_padding_mask=memory_key_padding_mask)
        logits = self.classifier(self.dropout(self.norm(dec)))
        return logits

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def prepare_model(configs: SimpleNamespace):
    model_name = configs.model.model_name
    model_type = configs.model.backbone_type.upper()
    if model_type == "ESMC":
        tokenizer = ESMC.from_pretrained(model_name).tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = KinaseModel(configs)
    return tokenizer, model

# ============================================================
# 2) FASTA / TOKENIZATION HELPERS
# ============================================================
def read_fasta(path: str) -> List[Tuple[str, str]]:
    """
    Minimal FASTA reader. Returns list of (id, sequence).
    Header '>' line is used as id (first token up to whitespace).
    """
    entries: List[Tuple[str, str]] = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"FASTA not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cur_id, cur_seq = None, []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    entries.append((cur_id, "".join(cur_seq)))
                cur_id = line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
        if cur_id is not None:
            entries.append((cur_id, "".join(cur_seq)))
    return entries

def tokenize_pair_esm2(tokenizer, substrate_seq: str, kinase_seq: str, device: str, max_length: int = None):
    kwargs = dict(return_tensors="pt", padding=True, truncation=True)
    if max_length is not None:
        kwargs["max_length"] = max_length
    s = tokenizer([substrate_seq], **kwargs)
    k = tokenizer([kinase_seq], **kwargs)
    return (
        s["input_ids"].to(device), s["attention_mask"].to(device),
        k["input_ids"].to(device), k["attention_mask"].to(device),
    )

def tokenize_pair_esmc(tokenizer, substrate_seq: str, kinase_seq: str, device: str):
    s_ids = torch.tensor([tokenizer.encode(substrate_seq)], device=device)
    k_ids = torch.tensor([tokenizer.encode(kinase_seq)], device=device)
    s_mask = torch.ones_like(s_ids, device=device)
    k_mask = torch.ones_like(k_ids, device=device)
    return s_ids, s_mask, k_ids, k_mask

# ============================================================
# 3) INFERENCE
# ============================================================
def run_one(model: nn.Module, tokenizer, cfg: SimpleNamespace, substrate: str, kinase: str,
            device: str, threshold: float) -> Dict[str, Any]:
    if cfg.model.backbone_type.upper() == "ESMC":
        s_ids, s_mask, k_ids, k_mask = tokenize_pair_esmc(tokenizer, substrate, kinase, device)
    else:
        s_ids, s_mask, k_ids, k_mask = tokenize_pair_esm2(
            tokenizer, substrate, kinase, device, max_length=cfg.dataset.max_sequence_length
        )

    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    with torch.no_grad():
        with amp_ctx:
            logits = model(s_ids, s_mask, k_ids, k_mask) 
        probs = torch.sigmoid(logits.float()).squeeze(-1).squeeze(0)  

    bin_list = ["1" if p >= threshold else "0" for p in probs.tolist()]
    binary_str = "".join(bin_list)
    pos_idx = (probs >= threshold).nonzero(as_tuple=False).squeeze(-1)
    pos_idx_1based = (pos_idx + 1).tolist()

    return {
        "binary": binary_str,
        "positives_1based": pos_idx_1based,
        "length": len(binary_str),
    }

def run_batch(model: nn.Module, tokenizer, cfg: SimpleNamespace,
              substrates: list[str], kinases: list[str],
              device: str, threshold: float,
              max_length: int | None = None) -> list[Dict[str, Any]]:
    """
    Batched inference across multiple 1:1 substrate–kinase pairs.
    Returns a list of dicts in the same order as inputs.
    """
    assert len(substrates) == len(kinases), "run_batch: length mismatch"

    if cfg.model.backbone_type.upper() == "ESMC":
        s_ids_list = [torch.tensor([tokenizer.encode(s)]) for s in substrates]
        k_ids_list = [torch.tensor([tokenizer.encode(k)]) for k in kinases]
        s_ids = torch.nn.utils.rnn.pad_sequence([x.squeeze(0) for x in s_ids_list],
                                                batch_first=True, padding_value=0)
        k_ids = torch.nn.utils.rnn.pad_sequence([x.squeeze(0) for x in k_ids_list],
                                                batch_first=True, padding_value=0)
        s_mask = (s_ids != 0).to(dtype=torch.int64)
        k_mask = (k_ids != 0).to(dtype=torch.int64)
    else:
        tk_kwargs = dict(return_tensors="pt", padding=True, truncation=True)
        if max_length is not None:
            tk_kwargs["max_length"] = max_length
        s_tok = tokenizer(substrates, **tk_kwargs)
        k_tok = tokenizer(kinases, **tk_kwargs)
        s_ids, s_mask = s_tok["input_ids"], s_tok["attention_mask"]
        k_ids, k_mask = k_tok["input_ids"], k_tok["attention_mask"]

    s_ids, s_mask = s_ids.to(device), s_mask.to(device)
    k_ids, k_mask = k_ids.to(device), k_mask.to(device)

    results: list[Dict[str, Any]] = []
    with torch.no_grad():
        logits = model(s_ids, s_mask, k_ids, k_mask) 
        probs  = torch.sigmoid(logits.float()).squeeze(-1) 

    for p in probs:  # p: [Ls]
        bin_list = ["1" if float(x) >= threshold else "0" for x in p.tolist()]
        binary   = "".join(bin_list)
        pos_idx  = (p >= threshold).nonzero(as_tuple=False).squeeze(-1)
        results.append({
            "binary": binary,
            "positives_1based": (pos_idx + 1).tolist(),
            "length": len(binary),
        })
    return results

# ============================================================
# 4) TXT writer
# ============================================================
def write_txt_header(fh):
    fh.write("\t".join([
        "substrate_id",
        "kinase_id",
        "threshold",
        "length",
        "positives_1based_csv",
        "binary_mask"
    ]) + "\n")

def write_txt_line(fh, sub_id: str, kin_id: str, threshold: float, length: int,
                   positives_1based: List[int], binary_mask: str):
    pos_csv = ",".join(str(x) for x in positives_1based)
    fh.write("\t".join([
        sub_id, kin_id, f"{threshold:.3f}", str(length), pos_csv, binary_mask
    ]) + "\n")

# ============================================================
# 5) CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Kinase-Substrate inference (FASTA → TXT; 1:1 pairing)")
    ap.add_argument("--weights", default=os.getenv("MODEL_PATH", "model_state.pth"),
                    help="Path to weights (state_dict). Baked-in default: /app/model_state.pth")
    ap.add_argument("--substrate_fasta", required=True, help="FASTA file of substrate sequences")
    ap.add_argument("--kinase_fasta", required=True, help="FASTA file of kinase sequences")
    ap.add_argument("--out_txt", default="preds.txt", help="Output TXT (tab-separated)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--threshold", type=float, default=0.5)
    # NEW: batching controls
    ap.add_argument("--batched", action="store_true",
                    help="Enable batched inference across multiple 1:1 pairs")
    ap.add_argument("--batch_size", type=int, default=4,
                    help="Batch size when --batched is set (tune based on RAM/CPU cores)")
    args = ap.parse_args()

    # Decide device
    device = "cuda" if (args.device in ["auto", "cuda"] and torch.cuda.is_available()) else "cpu"

    # Build tokenizer + model
    tokenizer, model = prepare_model(CONFIG)

    # Load weights (handle {'model_state_dict': ...} checkpoints and DataParallel prefixes)
    sd = torch.load(args.weights, map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    if len(sd) and isinstance(next(iter(sd)), str) and next(iter(sd)).startswith("module."):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)

    # Device + dtype
    model.to(device)
    model.to(dtype=(torch.bfloat16 if device == "cuda" else torch.float32))
    model.eval()

    # Read FASTAs (one-to-one pairing)
    substrates = read_fasta(args.substrate_fasta)  # List[(id, seq)]
    kinases   = read_fasta(args.kinase_fasta)
    if not substrates:
        raise ValueError(f"No substrate sequences found in {args.substrate_fasta}")
    if not kinases:
        raise ValueError(f"No kinase sequences found in {args.kinase_fasta}")
    if len(substrates) != len(kinases):
        raise ValueError(f"FASTA length mismatch: {len(substrates)} substrates vs {len(kinases)} kinases "
                         f"(one-to-one pairing requires equal counts)")

    n_pairs = len(substrates)
    with open(args.out_txt, "w", encoding="utf-8") as fh:
        write_txt_header(fh)

        if not args.batched:
            # Default: unbatched, pair-by-pair using run_one
            for (sub_id, sub_seq), (kin_id, kin_seq) in zip(substrates, kinases):
                out = run_one(model, tokenizer, CONFIG, sub_seq, kin_seq, device, args.threshold)
                write_txt_line(
                    fh,
                    sub_id=sub_id,
                    kin_id=kin_id,
                    threshold=args.threshold,
                    length=out["length"],
                    positives_1based=out["positives_1based"],
                    binary_mask=out["binary"],
                )
        else:
            # Batched: process in chunks of --batch_size using run_batch
            B = max(1, int(args.batch_size))
            for start in range(0, n_pairs, B):
                chunk_sub = substrates[start:start+B]
                chunk_kin = kinases[start:start+B]
                sub_ids   = [sid for sid, _ in chunk_sub]
                sub_seqs  = [seq for _,  seq in chunk_sub]
                kin_ids   = [kid for kid, _ in chunk_kin]
                kin_seqs  = [seq for _,  seq in chunk_kin]

                outs = run_batch(
                    model, tokenizer, CONFIG,
                    substrates=sub_seqs, kinases=kin_seqs,
                    device=device, threshold=args.threshold,
                    max_length=CONFIG.dataset.max_sequence_length
                )
                for (sid, kid), out in zip(zip(sub_ids, kin_ids), outs):
                    write_txt_line(
                        fh,
                        sub_id=sid,
                        kin_id=kid,
                        threshold=args.threshold,
                        length=out["length"],
                        positives_1based=out["positives_1based"],
                        binary_mask=out["binary"],
                    )

    print(f"Wrote TXT: {args.out_txt} (pairs={n_pairs}) on device={device} "
          f"{'(batched, B=' + str(args.batch_size) + ')' if args.batched else '(unbatched)'}")


if __name__ == "__main__":
    main()
