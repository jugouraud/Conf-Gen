"""Measured GPU batching, cached SD3 conditions, and exact GPU repair geometry."""

import copy
import gc
import math
import statistics
import time
from types import SimpleNamespace

import numpy as np
import torch

from backend.src.common.storage import calculate_file_sha256, write_json
from backend.src.fields.resident import RebuiltSignatureStore
from backend.src.fields.signatures import (
    calculate_model_signature,
    evaluate_probe_field,
    load_signature_space_config,
)
from backend.src.inference.encoding import (
    ResidentProbeEvaluator,
    StagedPromptEncoder,
    compare_signature_values,
)
from backend.src.inference.flow import FlowModelAdapter, decode_latents_to_image
from backend.src.repair.steering import TemporalSteeringGeometry


def read_native_gpu_null(path, signature):
    record = torch.load(path, map_location="cpu", weights_only=True)
    if record.get("device") != "cuda" or record["adapter_signature"] != signature:
        raise ValueError(
            "GPU repair requires the original CUDA-encoded negative condition"
        )
    return record


def ensure_native_gpu_null(source, bank):
    """CPU FP16 text encoding is not interchangeable with the detector's CUDA encoding."""
    path = source / "optimization/native_gpu_null_embeddings.pt"
    if not path.exists():
        config = load_signature_space_config(source, "trajectory")
        config["model"].update(device="cuda", cpu_offload=True)
        original = FlowModelAdapter(config)
        if original.signature != bank["adapter_signature"]:
            raise ValueError("Original CUDA encoder model identity changed")
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "device": "cuda",
                "adapter_signature": original.signature,
                "embeddings": [v.cpu() for v in original.null_embeddings],
                "origin": "original FlowModelAdapter native CUDA negative encoding",
            },
            temporary,
        )
        temporary.replace(path)
        del original
        gc.collect()
        torch.cuda.empty_cache()
    read_native_gpu_null(path, bank["adapter_signature"])
    return path


def null_cache_path(source, batch, digest):
    return source / "optimization" / f"optimized-null-{batch}-{digest[:16]}.pt"


class GPURepairAdapter(FlowModelAdapter):
    def __init__(self, source, bank, groups):
        from diffusers import SD3Transformer2DModel

        native_null_path = ensure_native_gpu_null(source, bank)
        self.null_condition_digest = calculate_file_sha256(native_null_path)
        self.device = "cuda"
        self.metadata = copy.deepcopy(bank["metadata"])
        self.signature = calculate_model_signature(self.metadata)
        self.shape = tuple(self.metadata["latent_shape"])
        self.guidance_scale = self.metadata["guidance_scale"]
        self.max_sequence_length = self.metadata["max_sequence_length"]
        self.num_train_timesteps = self.metadata["scheduler"]["num_train_timesteps"]
        model = (
            SD3Transformer2DModel.from_pretrained(
                self.metadata["model_id"],
                revision=self.metadata["revision_sha"],
                subfolder="transformer",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            .eval()
            .requires_grad_(False)
            .to(self.device)
        )
        self.pipe = SimpleNamespace(transformer=model, vae=None)
        null = read_native_gpu_null(native_null_path, self.signature)
        if (
            null["adapter_signature"] != self.signature
            or bank["adapter_signature"] != self.signature
        ):
            raise ValueError("Recovered model and null-condition identities differ")
        self.null_embeddings = tuple(
            value.to(self.device) for value in null["embeddings"]
        )
        self.rows = {row["prompt"]: row for rows in groups.values() for row in rows}
        self.encoder = StagedPromptEncoder(self, source / "optimization/embeddings")
        self.embedding_cache = {}
        self.resident = None
        self._diagnostic = None

    def encode_prompt(self, prompt):
        if prompt not in self.embedding_cache:
            if len(self.embedding_cache) >= 64:
                self.embedding_cache.clear()
            self.embedding_cache[prompt] = tuple(
                value.to(self.device)
                for value in self.encoder.read_embeddings(self.rows[prompt])
            )
        return self.embedding_cache[prompt]

    def load_decoder(self):
        from diffusers import AutoencoderKL
        from diffusers.image_processor import VaeImageProcessor

        self.pipe.vae = (
            AutoencoderKL.from_pretrained(
                self.metadata["model_id"],
                revision=self.metadata["revision_sha"],
                subfolder="vae",
                torch_dtype=torch.float16,
            )
            .eval()
            .requires_grad_(False)
            .to(self.device)
        )
        self.pipe.image_processor = VaeImageProcessor(vae_scale_factor=8)

    def decode_latents_to_image(self, latents):
        return decode_latents_to_image(self, latents)

    def evaluate_frozen_probes(self, config, bank, embeddings):
        # Fresh positive model evaluations, using the same native null velocity
        # computed once on this GPU. No cached field mixture is substituted.
        return self.resident.evaluate_field(embeddings)

    def evaluate_repaired_probes(self, config, field, bank):
        first = self.resident.evaluate_field(field.prompt_embeddings)
        if field.anchor_embeddings is None:
            return first
        second = self.resident.evaluate_field(field.anchor_embeddings)
        strength = torch.tensor(
            [field.schedule(float(t)) for t in bank["times"]], dtype=first.dtype
        ).reshape((-1,) + (1,) * (first.ndim - 1))
        field.model_evaluations += 2 * math.ceil(
            len(bank["states"]) / self.resident.batch_size
        )
        return (1 - strength) * first + strength * second

    def evaluate_repaired_guided_velocity(
        self, states, flow_time, prompt, approved, strength
    ):
        self.last_guided_forward_calls = 1
        if not strength:
            return super().evaluate_guided_velocity(states, flow_time, prompt)
        if approved is None:
            raise ValueError("Nonzero steering requires an approved safe condition")
        if not getattr(self, "packed_sampling_enabled", True):
            original = super().evaluate_guided_velocity(states, flow_time, prompt)
            anchor = super().evaluate_guided_velocity(states, flow_time, approved)
            self.last_guided_forward_calls = 2
            self._diagnostic = (
                states,
                states._version,
                float(flow_time),
                prompt,
                original,
            )
            return (1 - strength) * original + strength * anchor
        count = len(states)
        conditions = (self.null_embeddings, prompt, approved)
        tokens = torch.cat([value[0].expand(count, -1, -1) for value in conditions])
        pooled = torch.cat([value[1].expand(count, -1) for value in conditions])
        timestep = torch.full(
            (3 * count,),
            (1 - float(flow_time)) * self.num_train_timesteps,
            device=self.device,
            dtype=torch.float32,
        )
        with torch.inference_mode():
            prediction = self.pipe.transformer(
                hidden_states=states.to(
                    self.device, dtype=self.pipe.transformer.dtype
                ).repeat(3, 1, 1, 1),
                timestep=timestep,
                encoder_hidden_states=tokens,
                pooled_projections=pooled,
                return_dict=False,
            )[0]
        null, original, anchor = (-part.float() for part in prediction.chunk(3))
        original = null + self.guidance_scale * (original - null)
        anchor = null + self.guidance_scale * (anchor - null)
        self._diagnostic = (states, states._version, float(flow_time), prompt, original)
        return (1 - strength) * original + strength * anchor

    def cached_diagnostic_velocity(self, states, flow_time, prompt):
        cached, self._diagnostic = self._diagnostic, None
        if (
            cached is not None
            and cached[0] is states
            and cached[1] == states._version
            and cached[2] == float(flow_time)
            and cached[3] is prompt
        ):
            return cached[4]
        return None

    def check_packed_sampling(self, output, bank, prompts):
        states = bank["states"][:1].to(self.device)
        flow_time = float(bank["times"][0])
        first, second = [self.encode_prompt(prompt) for prompt in prompts]
        original = super().evaluate_guided_velocity(states, flow_time, first)
        approved = super().evaluate_guided_velocity(states, flow_time, second)
        checks = []
        for strength in (0.25, 0.75, 1.0):
            actual = self.evaluate_repaired_guided_velocity(
                states, flow_time, first, second, strength
            )
            expected = (1 - strength) * original + strength * approved
            checks.append(
                {
                    "strength": strength,
                    **compare_signature_values(expected, actual),
                    "passed": bool(
                        torch.allclose(actual, expected, rtol=1e-3, atol=1e-3)
                    ),
                }
            )
        self.packed_sampling_enabled = all(record["passed"] for record in checks)
        self._diagnostic = None
        write_json(
            output / "gpu_sampling_compatibility.json",
            {
                "packed_sampling_enabled": self.packed_sampling_enabled,
                "checks": checks,
                "fallback": "separate native condition evaluations with diagnostic reuse",
            },
        )


def benchmark_probe_batches(
    output, source, adapter, bank, bank_hash, rows, config, execution
):
    embeddings = [adapter.encode_prompt(row["prompt"]) for row in rows]
    native_adapter = SimpleNamespace(
        device=adapter.device, evaluate_prompt_effect=adapter.evaluate_prompt_effect
    )
    native = [
        evaluate_probe_field(config, native_adapter, bank, values)
        for values in embeddings
    ]
    reports, best = [], None
    for batch in execution["probe_batch_candidates"]:
        try:
            evaluator = ResidentProbeEvaluator(
                adapter,
                bank,
                bank_hash,
                null_cache_path(source, batch, adapter.null_condition_digest),
                batch,
            )
            evaluator.prepare_null_cache()
            samples, checks = [], []
            for expected, condition in zip(native, embeddings, strict=True):
                actual = evaluator.evaluate_field(condition)
                checks.append(
                    {
                        **compare_signature_values(expected, actual),
                        "passed": bool(
                            torch.allclose(expected, actual, rtol=1e-3, atol=1e-3)
                        ),
                    }
                )
                for _ in range(execution["benchmark_repeats"]):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    evaluator.evaluate_field(condition)
                    torch.cuda.synchronize()
                    samples.append(time.perf_counter() - started)
            report = {
                "batch_size": batch,
                "median_field_seconds": statistics.median(samples),
                "samples_seconds": samples,
                "checks": checks,
                "passed": all(check["passed"] for check in checks),
                "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            }
            reports.append(report)
            if report["passed"] and (
                best is None or report["median_field_seconds"] < best[0]
            ):
                best = (report["median_field_seconds"], batch)
            del evaluator
        except torch.OutOfMemoryError:
            reports.append(
                {"batch_size": batch, "passed": False, "reason": "out_of_memory"}
            )
        gc.collect()
        torch.cuda.empty_cache()
    write_json(
        output / "gpu_batch_benchmark.json",
        {
            "gpu": torch.cuda.get_device_name(),
            "records": reports,
            "selected_batch_size": best[1] if best else None,
            "selection": "lowest measured median field time among numerically compatible candidates",
        },
    )
    if best is None:
        raise RuntimeError("No GPU batch size passed native compatibility checks")
    evaluator = ResidentProbeEvaluator(
        adapter,
        bank,
        bank_hash,
        null_cache_path(source, best[1], adapter.null_condition_digest),
        best[1],
    )
    evaluator.prepare_null_cache()
    return evaluator


class GPUReferenceStore(RebuiltSignatureStore):
    """Load each reference once; keep it on GPU when VRAM allows, otherwise host RAM."""

    def __init__(
        self,
        source,
        bank,
        bank_hash,
        signature,
        groups,
        workspace_gib=5,
        batch_size=8,
        device="cuda",
    ):
        super().__init__(source, bank, bank_hash, signature)
        self.device, self.batch_size = device, batch_size
        self.rows = [
            (split, row)
            for split in ("safe_reference", "unsafe_reference")
            for row in groups[split]
        ]
        self.lookup = {
            (split, row["id"]): index for index, (split, row) in enumerate(self.rows)
        }
        size = len(self.rows) * bank["states"].numel() * 4
        free = torch.cuda.mem_get_info()[0] if device == "cuda" else 0
        self.location = (
            device
            if device == "cuda" and size + workspace_gib * 1024**3 < free
            else "cpu"
        )
        if self.location == "cpu":
            import psutil

            if psutil.virtual_memory().available < size + 2 * 1024**3:
                raise RuntimeError(
                    "High-RAM reference caching needs more available host memory"
                )
        self.values = torch.empty(
            (len(self.rows), *bank["states"].shape),
            dtype=torch.float32,
            device=self.location,
        )
        for index, (split, row) in enumerate(self.rows):
            self.values[index].copy_(super().load_signature(split, row))
        self.query_cache = {}

    def load_signature(self, split, row):
        key = (split, row["id"])
        if key in self.lookup:
            return self.values[self.lookup[key]].cpu()
        if key not in self.query_cache:
            self.query_cache[key] = super().load_signature(split, row)
        return self.query_cache[key]

    def build_temporal_geometry(self, query, anchor, experiment):
        started = time.perf_counter()
        bank = self.bank
        k, m = bank["time_points"], bank["trajectories"]
        q = query.to(self.device).double().reshape(k, m, -1)
        direction = anchor.to(self.device).double().reshape(k, m, -1) - q
        weights = bank["weights"].to(self.device).double().reshape(k, m, 1)
        quadratic = (direction.square() * weights).sum(dim=(1, 2)).cpu().numpy()
        coefficients = {}
        for label in ("safe", "unsafe"):
            rows = experiment.groups[f"{label}_reference"]
            constant, linear = [], []
            for start in range(0, len(rows), self.batch_size):
                indexes = [
                    self.lookup[(f"{label}_reference", row["id"])]
                    for row in rows[start : start + self.batch_size]
                ]
                refs = (
                    self.values[indexes]
                    .to(self.device)
                    .double()
                    .reshape(len(indexes), k, m, -1)
                )
                difference = q.unsqueeze(0) - refs
                constant.append(
                    (difference.square() * weights).sum(dim=(2, 3)).cpu().numpy()
                )
                linear.append(
                    (2 * difference * direction * weights).sum(dim=(2, 3)).cpu().numpy()
                )
            coefficients[label] = (
                np.concatenate(constant),
                np.concatenate(linear),
                quadratic,
            )
        result = TemporalSteeringGeometry(
            coefficients, bank["midpoints"], experiment.calibration, query.numel()
        )
        result.geometry_seconds = time.perf_counter() - started
        return result
