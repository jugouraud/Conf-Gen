"""Field storage identity, signature spaces and distance batch limits."""

ANCHOR_BATCH_SIZE = 16

NEAREST_BATCH_SIZE = ANCHOR_BATCH_SIZE

VARIANTS = {
    "trajectory": ("euclidean", "anchor_static", "anchor_adaptive"),
    "source_slice": ("anchor_static", "anchor_adaptive"),
}

FEATURE_KIND = "guided_prompt_velocity_difference"
