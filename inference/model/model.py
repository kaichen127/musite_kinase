import torch
from torch import nn
from esm.models.esmc import ESMC


# ============================================================
# 1) MODEL DEFINITION (ESMC-only, inference-oriented)
# ============================================================
class KinaseModel(nn.Module):
    def __init__(self, configs, num_labels=1):
        super().__init__()

        # -----------------------------
        # 1. Core configuration
        # -----------------------------
        base_model_name = configs.model.model_name
        hidden_size = configs.model.hidden_size
        mp_dtype = str(getattr(configs, "mixed_precision_dtype", "fp32")).lower()
        torch_dtype = torch.bfloat16 if "bf16" in mp_dtype else torch.float32

        classifier_dropout_rate = getattr(configs.model, "classifier_dropout_rate", 0.1)
        esm_to_decoder_dropout_rate = getattr(configs.model, "last_state_dropout_rate", 0.1)
        self.noise_std = getattr(configs, "noise_std", 0.0)

        # -----------------------------
        # 2. Load ESMC backbone
        # -----------------------------
        if ESMC is None:
            raise ImportError("ESMC backbone requested but not installed.")
        self.base_model = ESMC.from_pretrained(base_model_name)
        self.encoder_to_decoder_dropout = nn.Dropout(esm_to_decoder_dropout_rate)
        # -----------------------------
        # Decoder Head
        # -----------------------------
        dec_cfg = configs.model.decoder_head
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=dec_cfg.num_heads,
            dim_feedforward=dec_cfg.dim_feedforward,
            dropout=dec_cfg.dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_cfg.num_layers)

        # -----------------------------
        # Classifier Head
        # -----------------------------
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(classifier_dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)

    # -----------------------------
    # Forward Pass
    # -----------------------------
    def forward(self, substrate_input_ids, substrate_attention_mask,
                kinase_input_ids, kinase_attention_mask):
        s_out = self.base_model(sequence_tokens=substrate_input_ids)
        s_repr = self.encoder_to_decoder_dropout(s_out.embeddings)

        k_out = self.base_model(sequence_tokens=kinase_input_ids)
        k_repr = self.encoder_to_decoder_dropout(k_out.embeddings)

        memory_key_padding_mask = (kinase_attention_mask == 0)
        dec = self.transformer_decoder(
            tgt=s_repr,
            memory=k_repr,
            memory_key_padding_mask=memory_key_padding_mask
        )

        logits = self.classifier(self.dropout(self.norm(dec)))
        return logits

# ============================================================
# MODEL FACTORY
# ============================================================
def prepare_model(configs):
    """
    Prepares the ESMC model and tokenizer for inference.
    """
    model_name = configs.model.model_name
    if ESMC is None:
        raise ImportError("ESMC is required but not installed.")
    tokenizer = ESMC.from_pretrained(model_name).tokenizer

    model = KinaseModel(configs)
    print(f"Loaded model")

    return tokenizer, model
