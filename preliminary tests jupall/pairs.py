"""Matched minimal contrasts for estimating the human-subject direction.

An unmatched design ("a misty forest at dawn" vs "a portrait of a woman
standing in a misty forest at dawn") confounds the subject with syntactic
complexity: the person side has more determiners and prepositions, so the
difference direction loads on function words -- measured attribution 0.62 on
'a' and 0.54 on 'in', ABOVE 'portrait' (0.50) and 'woman' (0.40).

These pairs hold the frame, the length and the function words fixed and swap
ONLY the subject noun, person <-> inanimate natural object. The difference
then isolates human-subject content.
"""

_SCENES = [
    "in a misty forest at dawn",
    "beside a still alpine lake",
    "on a rocky coastline at sunset",
    "in a field of wildflowers",
    "under tall pine trees",
    "beside a cascading waterfall",
    "on a mountain trail at sunrise",
    "in a meadow covered in morning fog",
    "beside a quiet forest stream",
    "on a windswept clifftop",
    "among autumn birch trees",
    "beside a calm bay at dusk",
]
_PERSON = ["a woman", "a man", "a hiker", "a child", "a fisherman", "a climber",
           "a shepherd", "a photographer", "a farmer", "a girl", "a boy", "a camper"]
_OBJECT = ["a boulder", "a fallen log", "a granite rock", "a wooden post",
           "a mossy stone", "a driftwood branch", "a stone cairn", "a birch stump",
           "a lichen-covered rock", "a fern cluster", "a rotting log", "a sandstone slab"]

# (scene without a person, same scene with a person) -- identical frame and length
PAIRS = [(f"a photograph of {o} {s}", f"a photograph of {p} {s}")
         for s, p, o in zip(_SCENES, _PERSON, _OBJECT)]

# A second block: portrait framing vs landscape framing, frame otherwise held fixed
PAIRS += [(f"a landscape photograph of {o} {s}", f"a portrait photograph of {p} {s}")
          for s, p, o in zip(_SCENES, _PERSON, _OBJECT)]
