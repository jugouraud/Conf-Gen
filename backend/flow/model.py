"""Stable Diffusion 3 flow velocity and native classifier-free guidance."""

import hashlib
import importlib
import importlib.util
import json
import math

import torch
from huggingface_hub import model_info


def _pipeline_class():
    if importlib.util.find_spec("diffusers") is None:
        raise RuntimeError("Flow inference requires Diffusers with StableDiffusion3Pipeline")
    pipeline = getattr(importlib.import_module("diffusers"), "StableDiffusion3Pipeline", None)
    if pipeline is None:
        raise RuntimeError("Installed Diffusers does not provide StableDiffusion3Pipeline")
    return pipeline


def model_signature(metadata: dict) -> str:
    """Hash model settings without Diffusers' unordered default-value list."""
    stable = dict(metadata)
    scheduler = dict(stable["scheduler"])
    if "_use_default_values" in scheduler:
        scheduler["_use_default_values"] = sorted(scheduler["_use_default_values"])
    stable["scheduler"] = scheduler
    return hashlib.sha256(json.dumps(stable, sort_keys=True,
                                     default=str).encode()).hexdigest()


class FlowAdapter:
    """Expose V_p, V_null and their CFG prompt effect in t=0 noise to t=1 image time."""

    def encode(self, prompt: str):
        with torch.inference_mode():
            embedding, _, pooled, _ = self.pipe.encode_prompt(
                prompt=prompt, prompt_2=None, prompt_3=None, device=self.device,
                num_images_per_prompt=1, do_classifier_free_guidance=False,
                max_sequence_length=self.max_sequence_length,
            )
        return embedding, pooled

    def _native_null(self):
        # The checkpoint pipeline constructs its own CFG negative branch.
        with torch.inference_mode():
            _, negative, _, negative_pooled = self.pipe.encode_prompt(
                prompt="", prompt_2=None, prompt_3=None, device=self.device,
                num_images_per_prompt=1, do_classifier_free_guidance=True,
                negative_prompt=None, max_sequence_length=self.max_sequence_length,
            )
        return negative, negative_pooled

    def velocity_pair(self, x, t, prompt_embeddings):
        """Return (V_null, V_prompt) at the same latent and increasing flow time."""
        if not 0 <= float(t) < 1:
            raise ValueError("Flow field time must be in [0, 1)")
        batch = x.shape[0]
        null, null_pooled = self.null_embeddings
        positive, positive_pooled = prompt_embeddings
        conditions = torch.cat((null.expand(batch, -1, -1),
                                positive.expand(batch, -1, -1)))
        pooled = torch.cat((null_pooled.expand(batch, -1),
                            positive_pooled.expand(batch, -1)))
        timestep = torch.full((2 * batch,), (1 - float(t)) * self.num_train_timesteps,
                              device=self.device, dtype=torch.float32)
        with torch.inference_mode():
            prediction = self.pipe.transformer(
                hidden_states=x.to(device=self.device, dtype=self.pipe.transformer.dtype).repeat(2, 1, 1, 1),
                timestep=timestep, encoder_hidden_states=conditions,
                pooled_projections=pooled, return_dict=False,
            )[0]
        if prediction.shape != (2 * batch, *self.shape):
            raise RuntimeError(f"Unexpected flow output shape: {prediction.shape}")
        # The scheduler advances x by (sigma_next-sigma)*output. With
        # t=1-sigma increasing toward the image, the velocity is -output.
        return tuple(-part.float() for part in prediction.chunk(2))

    def prompt_effect(self, x, t, prompt_embeddings):
        null, positive = self.velocity_pair(x, t, prompt_embeddings)
        return self.guidance_scale * (positive - null)

    def guided_velocity(self, x, t, prompt_embeddings):
        null, positive = self.velocity_pair(x, t, prompt_embeddings)
        return null + self.guidance_scale * (positive - null)

    def __init__(self, cfg: dict):
        model = cfg["model"]
        self.device = str(model["device"])
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device is unavailable")
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}[model["dtype"]]
        self.guidance_scale = float(cfg["experiment"]["guidance_scale"])
        if not math.isfinite(self.guidance_scale) or self.guidance_scale <= 0:
            raise ValueError("guidance_scale must be finite and positive")
        self.max_sequence_length = int(model.get("max_sequence_length", 256))
        if self.max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")
        revision = model_info(model["id"], revision=model["revision"]).sha
        kwargs = {"revision": revision, "torch_dtype": dtype}
        if model.get("variant"):
            kwargs["variant"] = model["variant"]
        self.pipe = _pipeline_class().from_pretrained(model["id"], **kwargs)
        if self.pipe.scheduler.__class__.__name__ != "FlowMatchEulerDiscreteScheduler":
            raise ValueError("Checkpoint must use FlowMatchEulerDiscreteScheduler")
        if model.get("cpu_offload", False):
            if not self.device.startswith("cuda"):
                raise ValueError("cpu_offload requires CUDA")
            self.pipe.enable_model_cpu_offload(device=self.device)
        else:
            self.pipe.to(self.device)
        for component in ("transformer", "text_encoder", "text_encoder_2", "text_encoder_3"):
            module = getattr(self.pipe, component, None)
            if module is not None:
                module.eval()
        scale = self.pipe.vae_scale_factor
        height, width = int(model["height"]), int(model["width"])
        if min(height, width) <= 0 or height % (scale * self.pipe.patch_size) or width % (scale * self.pipe.patch_size):
            raise ValueError("Image dimensions must be positive multiples of the VAE scale and patch size")
        self.shape = (int(self.pipe.transformer.config.in_channels), height // scale, width // scale)
        self.num_train_timesteps = int(self.pipe.scheduler.config.num_train_timesteps)
        self.null_embeddings = self._native_null()
        self.metadata = {
            "model_id": model["id"], "revision_sha": revision,
            "pipeline": "StableDiffusion3Pipeline", "dtype": str(dtype),
            "latent_shape": list(self.shape), "height": height, "width": width,
            "guidance_scale": self.guidance_scale,
            "max_sequence_length": self.max_sequence_length,
            "null_condition": "pipeline CFG negative_prompt=None",
            "scheduler": dict(self.pipe.scheduler.config),
            "field": "guided prompt effect = w*(V_prompt-V_null)",
            "time_orientation": "t=0 Gaussian noise to t=1 image; t=1-sigma",
            "model_output_conversion": "V=-transformer output",
        }
        self.signature = model_signature(self.metadata)
