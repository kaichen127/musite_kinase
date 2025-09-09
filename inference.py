#!/usr/bin/env python3
import os, json, argparse
from types import SimpleNamespace
from typing import Dict, Any
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
        # cache_dir="/home/dc57y/data/transformers_cache/",
        structure_aware=False,

        # Freezing & dropouts
        freeze_backbone=True,
        freeze_embeddings=True,
        num_unfrozen_layers=6,
        backbone_dropout_rate=0.3,
        classifier_dropout_rate=0.5,
        last_state_dropout_rate=0.0,

        # Decoder head
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
                        if hasattr(layer, "attention"):  # ESM2
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

        # Transformer Decoder Head
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
        logits = self.classifier(self.dropout(self.norm(dec)))  # [B, Ls, num_labels]
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
# 2) TOKENIZATION HELPERS
# ============================================================
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
            logits = model(s_ids, s_mask, k_ids, k_mask)               # [1, Ls, 1]
        probs = torch.sigmoid(logits.float()).squeeze(-1).squeeze(0)    # [Ls], fp32 for postproc

    bin_list = ["1" if p >= threshold else "0" for p in probs.tolist()]
    binary_str = "".join(bin_list)

    # 1-based positive indices
    pos_idx = (probs >= threshold).nonzero(as_tuple=False).squeeze(-1)
    pos_idx_1based = (pos_idx + 1).tolist()

    return {
        "substrate": substrate,
        "kinase": kinase,
        "binary": binary_str,
        "probs": [float(p) for p in probs.detach().cpu()],
        "one_indexed_positives": pos_idx_1based,
        "threshold": threshold,
    }

# ============================================================
# 4) CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Kinase-Substrate inference (config-free CLI)")
    ap.add_argument("--weights", default=os.getenv("MODEL_PATH", "model_state.pth"))
    ap.add_argument("--substrate", help="Substrate sequence (single inference)")
    ap.add_argument("--kinase", help="Kinase sequence (single inference)")
    ap.add_argument("--in_jsonl", help="Batch JSONL: lines with {\"substrate\":..., \"kinase\":...}")
    ap.add_argument("--out", default="preds.jsonl", help="Output JSONL path")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    device = "cuda" if (args.device in ["auto", "cuda"] and torch.cuda.is_available()) else "cpu"

    tokenizer, model = prepare_model(CONFIG)

    # Load weights (state_dict)
    sd = torch.load(args.weights, map_location="cpu")
    if len(sd) and isinstance(next(iter(sd)), str) and next(iter(sd)).startswith("module."):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)

    # Device + dtype policy
    model.to(device)
    if device == "cuda":
        model.to(dtype=torch.bfloat16)     # bf16 on GPU
    else:
        model.to(dtype=torch.float32)      # fp32 on CPU

    model.eval()

    # Single vs batch
    results = []
    if args.in_jsonl:
        with open(args.in_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                sub = item["substrate"]
                kin = item["kinase"]
                out = run_one(model, tokenizer, CONFIG, sub, kin, device, args.threshold)
                results.append(out)
    else:
        assert args.substrate and args.kinase, "Provide --substrate and --kinase or --in_jsonl"
        out = run_one(model, tokenizer, CONFIG, args.substrate, args.kinase, device, args.threshold)
        results.append(out)

    # Write JSONL
    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"wrote {args.out} ({len(results)} samples) on device={device}")

if __name__ == "__main__":
    main()
