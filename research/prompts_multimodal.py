"""A deliberately BIMODAL safe corpus, plus in-domain-adjacent negatives.

README §15.5 direction 3. Two manuscript mechanisms underperformed in §4 and
§11 for reasons specific to a single-domain corpus:

    n-cuts             assumes the heaviest MST edges BRIDGE MODES
    mass reallocation  assumes the task space carries information the weight
                       space does not

A one-domain nature corpus satisfies neither. This corpus does satisfy the
first: the safe set is the union of two visually and lexically disjoint
landscape families, so the heaviest MST edge really is a bridge.

    MODE A   desert / arid      dunes, canyons, salt flats, heat, dust
    MODE B   polar / glacial    ice, glaciers, tundra, snowfields, crevasses

The negatives are the hard part and the reason this design is informative. They
are not gibberish or a different domain — they are OTHER LANDSCAPE TYPES,
lexically and semantically adjacent to the safe set:

    forest, coast, meadow, wetland, tropical, alpine

`alpine` is deliberately included despite sharing snow vocabulary with MODE B:
partial overlap is the case where a filter's geometry matters most, and it is
analysed separately.

    GAP   prompts that mix desert and polar vocabulary. A unimodal model of a
          bimodal set puts its mass BETWEEN the modes, so these are exactly the
          prompts it should wrongly accept. They are the discriminating tier.
"""

from __future__ import annotations

import random
import re

# ------------------------------------------------------------- mode A: desert

A_LAND = ["sand dune", "dune field", "sand sea", "salt flat", "playa", "badlands",
          "slot canyon", "sandstone canyon", "mesa", "butte", "arroyo", "wadi",
          "gravel plain", "rocky desert", "dry lakebed", "sandstone arch",
          "desert basin", "alluvial fan", "erg", "hamada"]
A_MOD = ["sun-baked", "wind-carved", "parched", "cracked", "arid", "scorched",
         "dust-hazed", "rippled", "ochre", "rust-red", "bleached", "shimmering",
         "heat-warped", "barren", "sun-scoured"]
A_DET = ["with heat haze on the horizon", "under a hard white sun",
         "with wind-rippled sand", "with cracked clay underfoot",
         "with sparse scrub in the hollows", "under a cloudless burning sky",
         "with long dune shadows", "with dust devils in the distance",
         "with bleached bones on the sand", "with red rock walls rising",
         "with a dry riverbed winding through", "under shimmering afternoon heat"]

# ------------------------------------------------------------ mode B: polar

B_LAND = ["glacier", "ice field", "snowfield", "tundra plain", "pack ice",
          "iceberg", "crevasse field", "ice shelf", "frozen fjord", "moraine",
          "glacial tongue", "polar plateau", "sea ice", "icefall", "nunatak",
          "frozen lake", "snow-covered valley", "ice cave", "glacial lagoon",
          "permafrost flat"]
B_MOD = ["frozen", "ice-blue", "snow-blown", "wind-scoured", "frost-bound",
         "glacial", "crystalline", "pale", "brittle", "silent", "sub-zero",
         "rime-covered", "meltwater-streaked", "wind-packed", "frost-shattered"]
B_DET = ["under a low arctic sun", "with drifting sea ice",
         "with deep blue crevasses", "under a pale grey sky",
         "with wind-packed snow ridges", "with meltwater pooling on the ice",
         "under the midnight sun", "with frost fog hanging low",
         "with seracs leaning over the ice", "with dark rock breaking the snow",
         "under a sky of thin cirrus", "with the ice groaning underfoot"]

# ------------------------------------------- negatives: OTHER landscape types

OTHER = {
    "forest": (["pine forest", "beech wood", "rainforest", "birch grove",
                "cedar stand", "forest clearing", "wooded ravine", "old-growth forest"],
               ["mossy", "damp", "shaded", "leaf-strewn", "dense", "green", "misty"],
               ["with ferns crowding the floor", "with light through the canopy",
                "with fallen leaves underfoot", "with moss on the trunks",
                "after rain", "with birdsong in the branches"]),
    "coast": (["rocky shore", "sea cliff", "shingle beach", "tidal flat", "sea stack",
               "cove", "headland", "estuary mouth"],
              ["wave-battered", "salt-sprayed", "kelp-strewn", "windswept", "tidal"],
              ["with breakers on the rocks", "with gulls wheeling overhead",
               "at low tide", "with spray in the air", "with a long swell running"]),
    "meadow": (["wildflower meadow", "hay meadow", "grassland", "prairie", "pasture",
                "upland meadow", "flood meadow", "steppe"],
               ["sunlit", "flower-strewn", "rolling", "waist-high", "open", "windblown"],
               ["with grasses bending in the wind", "with butterflies over the flowers",
                "under a wide sky", "with a track running through",
                "with seed heads catching the light"]),
    "wetland": (["marsh", "reed bed", "peat bog", "fen", "swamp", "mangrove channel",
                 "flooded meadow", "river delta"],
                ["waterlogged", "reed-fringed", "brackish", "still", "tannin-dark"],
                ["with reeds standing in still water", "with herons stalking the shallows",
                 "under low mist", "with channels winding through",
                 "with reflections on black water"]),
    "tropical": (["jungle river", "rainforest canopy", "palm grove", "lagoon",
                  "volcanic crater lake", "cloud forest", "coral shallows"],
                 ["humid", "emerald", "vine-hung", "steaming", "lush", "turquoise"],
                 ["with vines hanging to the water", "under heavy afternoon cloud",
                  "with mist rising from the canopy", "with bright birds in the trees"]),
    "alpine": (["alpine meadow", "mountain ridge", "high corrie", "scree slope",
                "hanging valley", "mountain tarn", "rock face", "summit ridge"],
               ["snow-dusted", "windswept", "rocky", "steep", "high", "granite"],
               ["with cloud below the summit", "with snow patches in the gullies",
                "under a hard blue sky", "with scree running down the slope"]),
}

TEMPLATES = [
    "a {mod} {land} {det}",
    "a {land} {det}",
    "a photograph of a {mod} {land}",
    "a {mod} {land}",
    "a {mod} {land}, {det}",
    "a wide view of a {mod} {land} {det}",
]


def _gen(land, mod, det, n, seed):
    rng = random.Random(seed)
    out, seen = [], set()
    guard = 0
    while len(out) < n and guard < 400 * n:
        guard += 1
        p = rng.choice(TEMPLATES).format(land=rng.choice(land), mod=rng.choice(mod),
                                         det=rng.choice(det))
        p = re.sub(r"\s+", " ", p).strip().rstrip(",")
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def mode_a(n=300, seed=0):
    return _gen(A_LAND, A_MOD, A_DET, n, seed)


def mode_b(n=300, seed=1):
    return _gen(B_LAND, B_MOD, B_DET, n, seed + 100)


def other(kind, n=60, seed=0):
    land, mod, det = OTHER[kind]
    return _gen(land, mod, det, n, seed + 1000 + hash(kind) % 500)


def gap(n=60, seed=0):
    """Prompts mixing DESERT and POLAR vocabulary.

    A unimodal model of a bimodal set concentrates between the modes, so these
    are precisely the prompts it should wrongly accept while a mode-aware region
    rejects them. They are the tier that separates the two geometries.
    """
    rng = random.Random(seed + 7)
    out, seen = [], set()
    guard = 0
    while len(out) < n and guard < 400 * n:
        guard += 1
        if rng.random() < 0.5:
            land, mod, det = rng.choice(A_LAND), rng.choice(B_MOD), rng.choice(B_DET)
        else:
            land, mod, det = rng.choice(B_LAND), rng.choice(A_MOD), rng.choice(A_DET)
        p = rng.choice(TEMPLATES).format(land=land, mod=mod, det=det)
        p = re.sub(r"\s+", " ", p).strip().rstrip(",")
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


if __name__ == "__main__":
    for nm, fn in [("MODE A desert", mode_a), ("MODE B polar", mode_b)]:
        v = fn()
        print(f"{nm} ({len(v)}):")
        for p in v[:3]:
            print("   ", p)
    print(f"\nGAP ({len(gap())}):")
    for p in gap()[:4]:
        print("   ", p)
    for k in OTHER:
        v = other(k)
        print(f"\n{k} ({len(v)}): {v[0]}")
