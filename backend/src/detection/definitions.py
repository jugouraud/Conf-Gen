"""Detection methods, score identities and admission modes."""

SCORE_KIND = "uniform_mean_anchor_distance_v1"

ADAPTIVE_SCORE_KIND = "gaussian_weighted_anchor_distance_v1"

EUCLIDEAN_SCORE_KIND = "nearest_euclidean_anchor_distance_v1"

DISTANCE_METHODS = ("euclidean", "anchor_static", "anchor_adaptive")

METHOD_SCORE_KINDS = dict(
    zip(DISTANCE_METHODS, (EUCLIDEAN_SCORE_KIND, SCORE_KIND, ADAPTIVE_SCORE_KIND))
)

METHOD_ALIASES = {
    "euclidian": "euclidean",
    "anchor_:static": "anchor_static",
    "anchor:static": "anchor_static",
    "anchor_adaptative": "anchor_adaptive",
}

MODES = ("safe_attraction", "safe_attraction_unsafe_avoidance")
