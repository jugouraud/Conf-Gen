"""FP64 distance kernels over weighted fields and streamed frozen probes."""

import gc
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch


def calculate_reference_distance_blocks(
    groups,
    reference_split,
    query_splits,
    load_vector,
    *,
    device="cuda",
    batch_size=16,
    progress=None,
):
    """Compute FP64 Euclidean blocks with one resident reference matrix.

    The caller supplies signature loading and optional progress reporting;
    provenance validation and persistence remain at the workflow boundary.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows = groups[reference_split]
    first = load_vector(reference_split, rows[0])
    refs = torch.empty((len(rows), first.numel()), dtype=torch.float64, device=device)
    refs[0] = first.to(device)
    for index, row in enumerate(rows[1:], 1):
        refs[index] = load_vector(reference_split, row).to(device)
    norms = torch.linalg.vector_norm(refs, dim=1).square()
    tables = {}
    for split in query_splits:
        values = []
        for start in range(0, len(groups[split]), batch_size):
            batch = groups[split][start : start + batch_size]
            query = torch.stack([load_vector(split, row) for row in batch]).to(
                device=device, dtype=torch.float64
            )
            values.append(
                calculate_pairwise_euclidean_distances(
                    query, refs, reference_squared_norms=norms
                )
                .cpu()
                .numpy()
            )
            if progress is not None:
                progress(split, start + len(batch), len(groups[split]))
        tables[split] = np.concatenate(values)
    del refs, norms, query
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return tables


def calculate_blocked_reference_distance_blocks(
    groups,
    reference_split,
    query_splits,
    load_vector,
    *,
    device="cuda",
    batch_size=16,
    reference_batch_size=128,
    progress=None,
):
    """FP64 Euclidean products with bounded GPU references and cached CPU queries."""
    if min(batch_size, reference_batch_size) < 1:
        raise ValueError("Block sizes must be positive")
    references = groups[reference_split]
    first = load_vector(reference_split, references[0])
    queries = {}
    for split in query_splits:
        cache = torch.empty((len(groups[split]), first.numel()), dtype=torch.float32)
        for index, row in enumerate(groups[split]):
            cache[index] = load_vector(split, row)
        queries[split] = cache
    tables = {
        split: np.empty((len(groups[split]), len(references)), dtype=np.float64)
        for split in query_splits
    }
    for start in range(0, len(references), reference_batch_size):
        rows = references[start : start + reference_batch_size]
        refs = torch.stack([load_vector(reference_split, row) for row in rows]).to(
            device=device, dtype=torch.float64
        )
        norms = torch.linalg.vector_norm(refs, dim=1).square()
        for split, cache in queries.items():
            for offset in range(0, len(cache), batch_size):
                query = cache[offset : offset + batch_size].to(
                    device=device, dtype=torch.float64
                )
                distances = calculate_pairwise_euclidean_distances(
                    query, refs, reference_squared_norms=norms
                )
                tables[split][
                    offset : offset + len(query), start : start + len(refs)
                ] = distances.cpu().numpy()
                if progress is not None:
                    progress(split, start + len(refs), len(references))
        del refs, norms, query, distances
    return tables


def calculate_pairwise_euclidean_distances(
    queries, references, *, reference_squared_norms=None
):
    """Float64 avoids cancellation/TF32 errors for long field signatures."""
    if (
        queries.ndim != 2
        or references.ndim != 2
        or queries.shape[1] != references.shape[1]
        or len(references) == 0
    ):
        raise ValueError(
            "Expected two matching feature matrices and nonempty references"
        )
    q, r = queries.double(), references.double()
    norms = (
        torch.linalg.vector_norm(r, dim=1).square()
        if reference_squared_norms is None
        else reference_squared_norms
    )
    squared = (
        torch.linalg.vector_norm(q, dim=1).square()[:, None]
        + norms[None, :]
        - 2 * (q @ r.T)
    )
    distances = squared.clamp_min_(0).sqrt_()
    if not torch.isfinite(distances).all():
        raise ValueError("Non-finite Euclidean distance matrix")
    return distances


PROBE_AGGREGATIONS = ("rms", "max")


def validate_probe_aggregation(value):
    if value not in PROBE_AGGREGATIONS:
        raise ValueError(f"Unknown probe aggregation {value!r}; choose rms or max")
    return value


def calculate_probe_deviation_matrix(
    queries, references, *, probe_aggregation="rms", weights=None
):
    """Reduce L2 deviations across probes, before any aggregation over anchors.

    Inputs have shape (prompts, probes, latent coordinates). Max uses unweighted
    probe L2 norms. RMS uses the normalized quadrature weights (uniform by default).
    """
    validate_probe_aggregation(probe_aggregation)
    if (
        queries.ndim != 3
        or references.ndim != 3
        or queries.shape[1:] != references.shape[1:]
        or not len(queries)
        or not len(references)
        or not queries.shape[1]
        or not queries.shape[2]
    ):
        raise ValueError("Expected nonempty matching prompt/probe/coordinate tensors")
    if not torch.isfinite(queries).all() or not torch.isfinite(references).all():
        raise ValueError("Nonfinite probe values")
    count = queries.shape[1]
    weights = (
        torch.full((count,), 1 / count, dtype=torch.float64, device=queries.device)
        if weights is None
        else torch.as_tensor(weights, dtype=torch.float64, device=queries.device)
    )
    if (
        weights.shape != (count,)
        or not torch.isfinite(weights).all()
        or (weights < 0).any()
        or not torch.isclose(
            weights.sum(), torch.tensor(1.0, dtype=weights.dtype, device=weights.device)
        )
    ):
        raise ValueError("Probe weights must be finite, nonnegative and sum to one")
    result = torch.zeros(
        (len(queries), len(references)), dtype=torch.float64, device=queries.device
    )
    for probe in range(count):
        distances = calculate_pairwise_euclidean_distances(
            queries[:, probe], references[:, probe]
        )
        if probe_aggregation == "max":
            torch.maximum(result, distances, out=result)
        else:
            result.add_(weights[probe] * distances.square())
    return result if probe_aggregation == "max" else result.sqrt_()


def calculate_streamed_probe_distances(
    groups,
    load_values,
    weights,
    *,
    probe_aggregation="rms",
    working_directory,
    device="cpu",
    batch_size=256,
    progress=None,
    checkpoint=None,
    resumed=None,
):
    """Compute validation/test-to-train and within-class train distances in FP64.

    A temporary FP32 cache avoids rereading each full field file for every probe.
    One probe's coordinates and all running reductions stay on the compute device.
    Only checkpoint/final matrices return to CPU. Checkpoints hold the reduced
    distances and the next probe index, never a fitted threshold or test-derived rule.
    """
    validate_probe_aggregation(probe_aggregation)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    first_split = next(iter(groups))
    first = load_values(first_split, groups[first_split][0]).reshape(len(weights), -1)
    if not len(weights) or first.numel() == 0:
        raise ValueError("Empty probe signature")
    slices, offset = {}, 0
    for split, rows in groups.items():
        if not rows:
            raise ValueError("Empty prompt split")
        slices[split] = slice(offset, offset + len(rows))
        offset += len(rows)
    tables = {
        f"{label}__{split}": np.zeros(
            (len(groups[split]), len(groups[f"{label}_reference"])), dtype=np.float64
        )
        for label in ("safe", "unsafe")
        for split in (
            "safe_validation",
            "unsafe_validation",
            "safe_test",
            "unsafe_test",
            f"{label}_reference",
        )
    }
    next_probe = 0
    if resumed is not None:
        next_probe, previous = resumed
        if (
            not 0 <= next_probe <= len(weights)
            or previous.keys() != tables.keys()
            or any(
                previous[k].shape != tables[k].shape
                or not np.isfinite(previous[k]).all()
                or (previous[k] < 0).any()
                for k in tables
            )
        ):
            raise ValueError("Invalid probe-distance checkpoint")
        tables = previous
    tables = {key: torch.from_numpy(value).to(device) for key, value in tables.items()}
    Path(working_directory).mkdir(parents=True, exist_ok=True)
    with (
        tempfile.TemporaryDirectory(
            prefix="probe-geometry-", dir=working_directory
        ) as folder,
        ExitStack() as mappings,
    ):
        path = Path(folder) / "fields.dat"
        shape = (offset, *first.shape)
        cache = np.memmap(path, dtype=np.float32, mode="w+", shape=shape)
        mappings.callback(cache._mmap.close)
        completed = 0
        for split, rows in groups.items():
            for index, row in enumerate(rows):
                values = load_values(split, row).float().reshape(first.shape)
                if not torch.isfinite(values).all():
                    raise ValueError("Nonfinite signature")
                if probe_aggregation == "rms":
                    # Preserve the existing FP32 quadrature scaling before FP64 products.
                    values = values * weights.float().sqrt()[:, None]
                cache[slices[split].start + index] = values.cpu().numpy()
                completed += 1
                if progress is not None and (
                    completed % 25 == 0 or completed == offset
                ):
                    progress("cache_signatures", completed, offset)
        cache.flush()
        cache._mmap.close()
        del cache
        cache = np.memmap(path, dtype=np.float32, mode="r", shape=shape)
        mappings.callback(cache._mmap.close)
        for probe in range(next_probe, len(weights)):
            values = (
                torch.from_numpy(
                    np.array(cache[:, probe, :], dtype=np.float32, order="C")
                )
                .to(device)
                .double()
            )
            for label in ("safe", "unsafe"):
                references = values[slices[f"{label}_reference"]]
                norms = torch.linalg.vector_norm(references, dim=1).square()
                for split in (
                    "safe_validation",
                    "unsafe_validation",
                    "safe_test",
                    "unsafe_test",
                    f"{label}_reference",
                ):
                    queries = values[slices[split]]
                    target = tables[f"{label}__{split}"]
                    for start in range(0, len(queries), batch_size):
                        distances = calculate_pairwise_euclidean_distances(
                            queries[start : start + batch_size],
                            references,
                            reference_squared_norms=norms,
                        )
                        block = target[start : start + len(distances)]
                        if probe_aggregation == "max":
                            torch.maximum(block, distances, out=block)
                        else:
                            block.add_(distances.square())
            del values, references, queries, norms
            if checkpoint is not None:
                checkpoint(
                    probe + 1,
                    {key: value.cpu().numpy().copy() for key, value in tables.items()},
                )
            if progress is not None:
                progress("reduce_probes", probe + 1, len(weights))
        del cache
    for label in ("safe", "unsafe"):
        tables[f"{label}__{label}_reference"].fill_diagonal_(0.0)
    if probe_aggregation == "rms":
        tables = {key: value.sqrt() for key, value in tables.items()}
    return {key: value.cpu().numpy() for key, value in tables.items()}


def calculate_streamed_reference_distances(
    rows, load_vector, *, working_directory, feature_block_size=65536, progress=None
):
    """Accumulate the complete Gram matrix along feature blocks, without approximation."""
    if not rows or feature_block_size < 1:
        raise ValueError(
            "Nonempty references and a positive feature block size are required"
        )
    first = load_vector(rows[0]).float().cpu().numpy()
    count, dimension = len(rows), first.size
    Path(working_directory).mkdir(parents=True, exist_ok=True)
    # Use disk-backed storage, not the default /tmp, which may be a small tmpfs.
    with tempfile.TemporaryDirectory(
        prefix="reference-geometry-", dir=working_directory
    ) as temporary:
        path = Path(temporary) / "vectors.dat"
        vectors = np.memmap(path, dtype=np.float32, mode="w+", shape=(count, dimension))
        vectors[0] = first
        for index, row in enumerate(rows[1:], 1):
            vector = load_vector(row).float().cpu().numpy()
            if vector.shape != first.shape or not np.isfinite(vector).all():
                raise ValueError("Invalid training reference signature")
            vectors[index] = vector
            if progress is not None and ((index + 1) % 25 == 0 or index + 1 == count):
                progress("cache_references", index + 1, count)
        vectors.flush()
        del vectors
        vectors = np.memmap(path, dtype=np.float32, mode="r", shape=(count, dimension))
        gram = torch.zeros((count, count), dtype=torch.float64)
        for start in range(0, dimension, feature_block_size):
            stop = min(dimension, start + feature_block_size)
            block = torch.from_numpy(
                np.array(vectors[:, start:stop], dtype=np.float64, order="C")
            )
            gram.addmm_(block, block.T)
            del block
            if progress is not None:
                progress("gram_features", stop, dimension)
        del vectors
    norms = gram.diagonal().clone()
    squared = norms[:, None] + norms[None, :] - 2 * gram
    distances = squared.clamp_min_(0).sqrt_().numpy()
    np.fill_diagonal(distances, 0.0)
    if not np.isfinite(distances).all():
        raise ValueError("Nonfinite training-reference distances")
    return distances


class ResidentFieldDistances:
    """Upload each query once; compare incoming GPU references without a CPU round trip."""

    def __init__(self, count, shape, weights, *, device, query_batch_size=16):
        if count < 1 or query_batch_size < 1:
            raise ValueError("Positive query count and batch size required")
        self.device = torch.device(device)
        self.query_batch_size = query_batch_size
        self.scale = weights.to(device=self.device, dtype=torch.float32).sqrt()
        self.shape = tuple(shape)
        self.values = torch.empty(
            (count, *shape), device=self.device, dtype=torch.float32
        )
        self.completed = 0

    def append(self, field):
        if self.completed >= len(self.values) or tuple(field.shape) != self.shape:
            raise ValueError("Unexpected query field or cache capacity exceeded")
        self.values[self.completed].copy_(
            field.to(self.device).float() * self.scale[:, None, None, None]
        )
        self.completed += 1

    def measure(self, fields):
        if self.completed != len(self.values):
            raise ValueError("Query cache is incomplete")
        references = torch.stack(fields).to(self.device).float()
        references = (
            (references * self.scale[None, :, None, None, None]).flatten(1).double()
        )
        norms = torch.linalg.vector_norm(references, dim=1).square()
        result = torch.empty(
            (len(self.values), len(fields)), device=self.device, dtype=torch.float64
        )
        for start in range(0, len(self.values), self.query_batch_size):
            queries = self.values[start : start + self.query_batch_size].flatten(1)
            result[start : start + len(queries)] = (
                calculate_pairwise_euclidean_distances(
                    queries, references, reference_squared_norms=norms
                )
            )
        return result


def calculate_resident_training_distances(fields, *, feature_batch_size=65536):
    """FP64 Gram accumulation; one FP32 reference class already resides on GPU.

    Field weights must be applied in FP32 before calling, as in the historical
    Euclidean baseline. Cast coordinate tiles only, never the entire cache.
    """
    if fields.ndim != 2 or min(fields.shape) < 1 or feature_batch_size < 1:
        raise ValueError("Expected a nonempty field matrix and positive tile size")
    gram = torch.zeros(
        (len(fields), len(fields)), device=fields.device, dtype=torch.float64
    )
    for start in range(0, fields.shape[1], feature_batch_size):
        tile = fields[:, start : start + feature_batch_size].double()
        gram.addmm_(tile, tile.T)
    norms = gram.diagonal().clone()
    result = (norms[:, None] + norms[None, :] - 2 * gram).clamp_min_(0).sqrt_()
    result.fill_diagonal_(0)
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite training distances")
    return result
