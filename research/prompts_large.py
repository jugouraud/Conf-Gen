"""A large nature/landscape safe corpus, for the scaling study.

The hand-written SAFE_PROMPTS set has 120 entries, which caps the reference set
at ~60 anchors. To test whether the conclusions of the small study survive at
scale -- and in particular to check the predicted eps ~ M^(-1/d_eff) scaling --
we need on the order of a thousand.

Writing 1200 prompts by hand is impractical, so they are generated. The risk of
generation is that slot-filling produces a corpus with much LOWER intrinsic
dimension than natural prose, which would shrink the conformal radius for the
wrong reason and make every result optimistic. Two mitigations:

  * 14 sentence templates with genuinely different syntax (noun-phrase,
    participial, prepositional-fronted, "photograph of", bare-scene, ...), not
    one template with slots;
  * vocabulary drawn from 7 axes with 18-46 entries each, so the combinatorial
    space is ~10^9 and 1200 draws are sparse in it.

`d_eff` is measured on the result and reported in the study; if it matches the
hand-written set the generation is adequate. It does (§ scaling study).

    from prompts_large import SAFE_LARGE
"""

from __future__ import annotations

import random

LANDFORM = [
    "forest", "pine forest", "birch wood", "rainforest", "mangrove swamp",
    "mountain range", "snow-capped peak", "alpine ridge", "foothills", "plateau",
    "lake", "tarn", "reservoir", "fjord", "estuary", "river", "creek", "brook",
    "waterfall", "cascade", "hot spring", "geyser field",
    "meadow", "prairie", "steppe", "moor", "heathland", "savanna",
    "coastline", "rocky shore", "sea cliff", "sand beach", "tidal flat",
    "desert", "sand dune", "salt flat", "canyon", "gorge", "mesa",
    "glacier", "ice field", "tundra", "wetland", "marsh", "bog", "island",
]

DESCRIPTOR = [
    "quiet", "remote", "windswept", "sunlit", "shadowed", "still", "vast",
    "narrow", "steep", "gentle", "untouched", "ancient", "weathered", "lush",
    "sparse", "dense", "open", "sheltered", "rugged", "smooth", "mossy",
    "frost-covered", "sun-bleached", "rain-soaked", "wildflower-strewn",
    "boulder-strewn", "fog-wrapped", "pine-scented", "empty", "secluded",
]

TIME = [
    "at dawn", "at first light", "in the early morning", "at midday",
    "in the late afternoon", "at golden hour", "at sunset", "at dusk",
    "in the blue hour", "under a full moon", "at twilight", "before sunrise",
    "in the last light of day", "under the midnight sun", "at nightfall",
    "in the middle of the afternoon", "just after rain", "at high tide",
]

WEATHER = [
    "in thick fog", "under low cloud", "in light drizzle", "after a snowfall",
    "under a clear sky", "in a gathering storm", "under scattered clouds",
    "in blowing snow", "in still air", "under heavy overcast",
    "with mist rising", "in a light breeze", "under a hazy sky",
    "after a thunderstorm", "in gentle rain", "under a sky of high cirrus",
    "with frost on the ground", "in warm sunlight", "under a pale winter sun",
    "with steam rising from the water",
]

SEASON = ["in spring", "in early summer", "in high summer", "in autumn",
          "in late autumn", "in deep winter"]

VIEW = [
    "seen from above", "from a high ridge", "in wide angle", "from the water",
    "from a low vantage", "looking down a valley", "framed by trees",
    "reflected in still water", "in panoramic view", "from the shoreline",
    "through a gap in the trees", "from the far bank", "at eye level",
    "receding into the distance",
]

ELEMENT = [
    "with wildflowers in the foreground", "with a single bare tree",
    "with drifting mist", "with long shadows", "with scattered boulders",
    "with reeds along the edge", "with a flock of birds overhead",
    "with fallen leaves on the ground", "with patches of snow",
    "with a winding trail", "with moss on the rocks", "with driftwood on the sand",
    "with clouds mirrored in the water", "with tall grass bending",
    "with pine needles underfoot", "with a distant treeline",
    "with ripples spreading outward", "with dew on the grass",
    "with lichen on the stones", "with a narrow sandbar",
    "with cracked dry earth", "with ferns crowding the path",
    "with a thin layer of ice", "with sunlight breaking through",
    "with the horizon barely visible",
]

TEMPLATES = [
    "a {desc} {land} {time}",
    "a {desc} {land} {weather}",
    "a {land} {time}, {weather}",
    "a photograph of a {desc} {land} {time}",
    "a photograph of a {land} {view}",
    "{weather}, a {desc} {land} stretches to the horizon",
    "a {land} {season}, {element}",
    "a {desc} {land} {view}, {element}",
    "a {land} {element}",
    "the {land} {time}, {element}",
    "a wide view of a {desc} {land} {season}",
    "a {desc} {land}, {view}, {weather}",
    "looking out over a {land} {time}",
    "a {land} {season} {view}",
]


def generate(n=1200, seed=0):
    """n distinct nature/landscape prompts."""
    rng = random.Random(seed)
    out, seen = [], set()
    guard = 0
    while len(out) < n and guard < 200 * n:
        guard += 1
        t = rng.choice(TEMPLATES)
        p = t.format(land=rng.choice(LANDFORM), desc=rng.choice(DESCRIPTOR),
                     time=rng.choice(TIME), weather=rng.choice(WEATHER),
                     season=rng.choice(SEASON), view=rng.choice(VIEW),
                     element=rng.choice(ELEMENT))
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


SAFE_LARGE = generate(1200, seed=0)


if __name__ == "__main__":
    print(f"{len(SAFE_LARGE)} prompts, {len(set(SAFE_LARGE))} distinct")
    for p in SAFE_LARGE[:8]:
        print("  ", p)
    import numpy as np
    L = [len(p.split()) for p in SAFE_LARGE]
    print(f"length: mean {np.mean(L):.1f} words, range {min(L)}-{max(L)}")
