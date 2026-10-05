"""
Extract conditioning embeddings from Stable Diffusion's text encoder.

Uses the CLIP text encoder (the τ_θ in the formalism) to map prompts
to the embedding space where we build the safe set.
"""

import torch
from transformers import CLIPTokenizer, CLIPTextModel


def load_text_encoder(model_id: str = "openai/clip-vit-large-patch14", device: str = "mps"):
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    text_encoder = CLIPTextModel.from_pretrained(model_id).to(device).eval()
    return tokenizer, text_encoder


@torch.no_grad()
def encode_prompts(
    prompts: list[str],
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    device: str = "mps",
    pooling: str = "eos",
) -> torch.Tensor:
    """Encode prompts to conditioning embeddings.

    Args:
        pooling: "eos" uses the [EOS] token embedding (CLIP's sentence vector),
                 "mean" averages over non-padding tokens,
                 "full" returns the full (L, d) sequence (no reduction).

    Returns:
        (N, d) for eos/mean pooling, (N, L, d) for full.
    """
    all_embeddings = []
    batch_size = 16

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        tokens = tokenizer(
            batch, padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).to(device)

        outputs = text_encoder(**tokens)
        hidden = outputs.last_hidden_state  # (B, L, d)

        if pooling == "eos":
            eos_idx = tokens.input_ids.argmax(dim=-1)
            emb = hidden[torch.arange(len(batch), device=device), eos_idx]
        elif pooling == "mean":
            mask = tokens.attention_mask.unsqueeze(-1).float()
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        elif pooling == "full":
            emb = hidden
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings, dim=0)


@torch.no_grad()
def encode_prompts_with_tasks(
    prompts: list[str],
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    device: str = "mps",
) -> tuple[torch.Tensor, list]:
    """Encode prompts into the (task, weight) pairs the adaptive scores need.

    Returns:
        theta  : (N, d) pooled [EOS] embeddings — the "weights" the filter constrains.
        clouds : list of (L_i, d) arrays of CONTENT-token hidden states — the
                 "task descriptors" tau_i, used only for task-space proximity.

    BOS and [EOS] are dropped from the cloud so that tau is not a trivial copy
    of the theta it is meant to predict: theta is CLIP's causal-attention
    summary at the [EOS] position, tau is the bag of content-token states.
    Prompts with no content tokens (empty or whitespace-only) fall back to the
    [EOS] state alone so the cloud is never empty.
    """
    all_theta, all_clouds = [], []
    batch_size = 16

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        tokens = tokenizer(
            batch, padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).to(device)

        hidden = text_encoder(**tokens).last_hidden_state          # (B, L, d)
        eos_idx = tokens.input_ids.argmax(dim=-1)                  # [EOS] has the top id

        theta = hidden[torch.arange(len(batch), device=device), eos_idx]
        all_theta.append(theta.cpu())

        for b in range(len(batch)):
            e = int(eos_idx[b])
            cloud = hidden[b, 1:e] if e > 1 else hidden[b, e:e + 1]
            all_clouds.append(cloud.cpu().numpy().astype("float64"))

    return torch.cat(all_theta, dim=0), all_clouds
