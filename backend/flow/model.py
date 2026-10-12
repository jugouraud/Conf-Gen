"""Sana inference adapter used by probes and feature extraction."""
import hashlib
import importlib
import importlib.util
import json

import torch
from huggingface_hub import model_info

SanaPipeline = (importlib.import_module("diffusers").SanaPipeline
                if importlib.util.find_spec("diffusers") is not None else None)

class SanaAdapter:
    """Expose model-native effective inference velocity with no image decoding."""

    # Hierarchy 3: atomic actions.

    def encode(self, prompt: str):
        with self.torch.inference_mode():
            embeds, mask, _, _ = self.pipe.encode_prompt(
                prompt=prompt, do_classifier_free_guidance=False,
                device=self.device, clean_caption=False,
                complex_human_instruction=None,
            )
        return embeds.to(self.pipe.transformer.dtype), mask

    def _forward(self, x, time, pair):
        embeddings, mask = pair
        batch = x.shape[0]
        t = time.reshape(1).expand(batch)
        t = t * self.pipe.transformer.config.timestep_scale
        pred = self.pipe.transformer(
            hidden_states=x.to(dtype=self.pipe.transformer.dtype),
            encoder_hidden_states=embeddings.expand(batch, -1, -1),
            encoder_attention_mask=mask.expand(batch, -1),
            timestep=t, return_dict=False,
        )[0].float()
        if pred.shape[1] == 2 * x.shape[1]:
            pred = pred.chunk(2, dim=1)[0]
        if pred.shape != x.shape:
            raise RuntimeError(f"Unexpected Sana field shape: {pred.shape}")
        return pred

    # Hierarchy 2: subworkflows.

    def velocity(self, x, time, pair):
        with self.torch.inference_mode():
            conditioned = self._forward(x, time, pair)
            if self.guidance == 1:
                return conditioned
            unconditioned = self._forward(x, time, self.unconditional)
            return unconditioned + self.guidance * (conditioned - unconditioned)

    # Hierarchy 1: workflow orchestration.

    def __init__(self, cfg: dict):
        if SanaPipeline is None:
            raise RuntimeError("Sana inference requires the diffusers package")

        self.torch = torch
        model = cfg["model"]
        self.device = str(model["device"])
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device is unavailable")
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}[model["dtype"]]
        resolved_revision = model_info(model["id"], revision=model["revision"]).sha
        kwargs = {"revision": resolved_revision, "torch_dtype": dtype}
        if model.get("variant"):
            kwargs["variant"] = model["variant"]
        self.pipe = SanaPipeline.from_pretrained(model["id"], **kwargs).to(self.device)
        self.pipe.transformer.eval()
        self.pipe.text_encoder.eval()
        if self.device.startswith("cuda") and torch.cuda.is_bf16_supported():
            self.pipe.text_encoder.to(torch.bfloat16)
        self.guidance = float(cfg["experiment"]["guidance_scale"])
        if self.guidance <= 0:
            raise ValueError("Positive guidance_scale required")
        self.steps = int(cfg["probes"]["inference_steps"])
        self.pipe.scheduler.set_timesteps(self.steps, device=self.device)
        self.shape = (int(self.pipe.transformer.config.in_channels),
                      int(model["height"]) // self.pipe.vae_scale_factor,
                      int(model["width"]) // self.pipe.vae_scale_factor)
        if min(self.shape) <= 0:
            raise ValueError("Invalid native latent dimensions")
        self.metadata = {
            "model_id": model["id"], "revision_sha": resolved_revision,
            "text_encoder": type(self.pipe.text_encoder).__name__,
            "dtype": str(dtype), "text_encoder_dtype": str(self.pipe.text_encoder.dtype),
            "latent_shape": list(self.shape), "guidance_scale": self.guidance,
            "steps": self.steps, "scheduler": dict(self.pipe.scheduler.config),
            "timestep_scale": float(self.pipe.transformer.config.timestep_scale),
            "time_orientation": "descending scheduler noise-to-image",
            "clean_caption": False, "complex_human_instruction": None,
        }
        self.signature = hashlib.sha256(json.dumps(self.metadata, sort_keys=True,
                                                   default=str).encode()).hexdigest()
        self.unconditional = self.encode("")
