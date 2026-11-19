"""
Frozen seed lists, versioned. Never change a released list; add a new version instead.

v1.0 (the stage-8 frozen run):
- dev:  400-404  (used for method design from M1 to stage 7; no longer unseen)
- test: 500-504  (fresh seeds of the frozen final run; the published leaderboard)
v1.1 (this release, adds the realistic shape tracks):
- dev:  600-604  (tuning of new recipes/methods allowed)
- test: 700-704  (held out until a design is frozen; the v1.1 shape-track leaderboard)
Earlier, retired: 100-104 (first screen), 200 (design scenes), 300-304 (CTS v3 test).
"""
SEEDS = {
    "1.0": {"dev": tuple(range(400, 405)), "test": tuple(range(500, 505))},
    "1.1": {"dev": tuple(range(600, 605)), "test": tuple(range(700, 705))},
}
RETIRED = (tuple(range(100, 105)), (200,), tuple(range(300, 305)))
CURRENT = "1.1"


def seeds(split="test", version=CURRENT):
    return SEEDS[version][split]
