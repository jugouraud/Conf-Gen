"""Native SD3 embedding stages and resident, null-cached frozen-probe inference."""

import hashlib
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as functional


def save_atomic_tensor(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(record, temporary)
    os.replace(temporary, path)


class StagedPromptEncoder:
    """Keep one native text encoder resident for a complete pass over missing prompts."""

    components = ("clip0", "clip1", "t5")
    modules = ("text_encoder", "text_encoder_2", "text_encoder_3")

    def __init__(self, adapter, root):
        self.adapter = adapter
        self.root = Path(root)

    def resolve_component_path(self, component, row):
        return self.root / component / f"{row['id']}.pt"

    def read_component(self, component, row):
        record = torch.load(
            self.resolve_component_path(component, row),
            map_location="cpu",
            weights_only=True,
        )
        if (
            record["adapter_signature"] != self.adapter.signature
            or record["prompt_sha256"]
            != hashlib.sha256(row["prompt"].encode()).hexdigest()
        ):
            raise ValueError("Stale prompt embedding cache")
        return record

    def prepare_component(self, component):
        pipe = self.adapter.pipe
        pipe.remove_all_hooks()
        for name in (*self.modules, "transformer", "vae"):
            module = getattr(pipe, name, None)
            if module is not None:
                module.to("cpu")
        if str(self.adapter.device).startswith("cuda"):
            torch.cuda.empty_cache()
        module = getattr(pipe, self.modules[self.components.index(component)])
        module.to(self.adapter.device)

    def encode_component(self, component, row):
        path = self.resolve_component_path(component, row)
        if path.exists():
            self.read_component(component, row)
            return
        pipe = self.adapter.pipe
        with torch.inference_mode():
            if component == "t5":
                tokens = pipe._get_t5_prompt_embeds(
                    prompt=row["prompt"],
                    device=self.adapter.device,
                    max_sequence_length=self.adapter.max_sequence_length,
                )
                pooled = None
            else:
                tokens, pooled = pipe._get_clip_prompt_embeds(
                    prompt=row["prompt"],
                    device=self.adapter.device,
                    clip_model_index=int(component[-1]),
                )
        save_atomic_tensor(
            path,
            {
                "adapter_signature": self.adapter.signature,
                "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
                "tokens": tokens.cpu(),
                "pooled": None if pooled is None else pooled.cpu(),
            },
        )

    def read_embeddings(self, row):
        first, second, t5 = [
            self.read_component(component, row) for component in self.components
        ]
        clip = torch.cat((first["tokens"], second["tokens"]), dim=-1)
        clip = functional.pad(clip, (0, t5["tokens"].shape[-1] - clip.shape[-1]))
        return (
            torch.cat((clip, t5["tokens"]), dim=-2),
            torch.cat((first["pooled"], second["pooled"]), dim=-1),
        )

    def prepare_transformer(self):
        pipe = self.adapter.pipe
        pipe.remove_all_hooks()
        for name in (*self.modules, "vae"):
            module = getattr(pipe, name, None)
            if module is not None:
                module.to("cpu")
        if str(self.adapter.device).startswith("cuda"):
            torch.cuda.empty_cache()
        pipe.transformer.to(self.adapter.device)


class ResidentProbeEvaluator:
    """Evaluate positive conditions only; the frozen null field is computed once."""

    def __init__(
        self,
        adapter,
        bank,
        bank_hash,
        cache_path,
        batch_size=16,
        *,
        pad_final_batch=False,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.adapter = adapter
        self.batch_size = batch_size
        self.pad_final_batch = pad_final_batch
        self.bank_hash = bank_hash
        self.cache_path = Path(cache_path)
        self.states = bank["states"].to(
            device=adapter.device, dtype=adapter.pipe.transformer.dtype
        )
        self.times = bank["times"].to(adapter.device)
        self.null_values = None

    def evaluate_velocity(self, embeddings):
        positive, pooled = [value.to(self.adapter.device) for value in embeddings]
        values = []
        with torch.inference_mode():
            for offset in range(0, len(self.states), self.batch_size):
                states = self.states[offset : offset + self.batch_size]
                count = len(states)
                timestep = (
                    1 - self.times[offset : offset + count]
                ) * self.adapter.num_train_timesteps
                if self.pad_final_batch and count < self.batch_size:
                    # Match the native CFG batch shape for an eight-probe source bank.
                    padding = self.batch_size - count
                    states = torch.cat((states, states[:1].expand(padding, -1, -1, -1)))
                    timestep = torch.cat((timestep, timestep[:1].expand(padding)))
                evaluated_count = len(states)
                prediction = self.adapter.pipe.transformer(
                    hidden_states=states,
                    timestep=timestep,
                    encoder_hidden_states=positive.expand(
                        evaluated_count, -1, -1
                    ).contiguous(),
                    pooled_projections=pooled.expand(evaluated_count, -1).contiguous(),
                    return_dict=False,
                )[0]
                values.append(-prediction[:count].float())
        return torch.cat(values)

    def prepare_null_cache(self):
        metadata = {
            "adapter_signature": self.adapter.signature,
            "bank_sha256": self.bank_hash,
            "batch_size": self.batch_size,
            "kind": "native_null_velocity_v1",
        }
        if self.pad_final_batch:
            metadata["pad_final_batch"] = True
        if self.cache_path.exists():
            record = torch.load(self.cache_path, map_location="cpu", weights_only=True)
            if any(record.get(key) != value for key, value in metadata.items()):
                raise ValueError("Stale null field cache")
            self.null_values = record["values"].to(self.adapter.device)
        else:
            self.null_values = self.evaluate_velocity(self.adapter.null_embeddings)
            save_atomic_tensor(
                self.cache_path, {**metadata, "values": self.null_values.cpu()}
            )
        if (
            self.null_values.shape != self.states.shape
            or not torch.isfinite(self.null_values).all()
        ):
            raise ValueError("Invalid null field cache")

    def evaluate_field(self, embeddings, *, output_device="cpu"):
        if self.null_values is None:
            raise ValueError("Prepare null field cache before evaluating prompts")
        return (
            self.adapter.guidance_scale
            * (self.evaluate_velocity(embeddings) - self.null_values)
        ).to(output_device)


def compare_signature_values(reference, candidate):
    difference = candidate.double() - reference.double()
    norm = torch.linalg.vector_norm(reference.double())
    return {
        "bitwise_equal": bool(torch.equal(reference, candidate)),
        "max_absolute_error": float(difference.abs().max()),
        "relative_l2_error": float(
            torch.linalg.vector_norm(difference) / norm.clamp_min(1e-12)
        ),
    }


def synchronize_model_device(adapter):
    if str(adapter.device).startswith("cuda"):
        torch.cuda.synchronize()
    return time.perf_counter()
