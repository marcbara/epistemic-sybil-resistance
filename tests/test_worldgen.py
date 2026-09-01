import sys
from pathlib import Path
from fractions import Fraction

import yaml
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import worldgen  # noqa: E402

CFG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


@pytest.fixture(scope="module")
def cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def test_solver_recombines_exactly_many_worlds(cfg):
    master_seed = cfg["master_seed"]
    for world_id in range(200):
        world = worldgen.generate_world(master_seed, world_id, cfg)
        doc = worldgen.generate_root_for_world(world, root_id=0, master_seed=master_seed, cfg=cfg)
        recovered = worldgen.solve_e_from_cues(doc.a_value, doc.p_pct, doc.q_value, doc.g_pct)
        assert recovered == doc.e_true, f"world {world_id}: solver mismatch"
        # exactness must hold to arbitrary precision, not just float closeness
        assert isinstance(recovered, Fraction)


def test_e_true_matches_document_text_cues(cfg):
    master_seed = cfg["master_seed"]
    world = worldgen.generate_world(master_seed, 7, cfg)
    doc = worldgen.generate_root_for_world(world, root_id=0, master_seed=master_seed, cfg=cfg)
    assert worldgen.format_money(doc.a_value) in doc.text
    assert str(doc.p_pct) in doc.text
    assert worldgen.format_money(doc.q_value) in doc.text
    assert str(doc.g_pct) in doc.text
    assert f"{float(doc.e_true):.1f}" not in doc.text  # E itself never stated verbatim


def test_format_money_never_emits_fraction_syntax(cfg):
    master_seed = cfg["master_seed"]
    for world_id in range(100):
        world = worldgen.generate_world(master_seed, world_id, cfg)
        doc = worldgen.generate_root_for_world(world, 0, master_seed, cfg)
        assert "/" not in doc.text


def test_e_true_tracks_theta_plus_eps_reasonably(cfg):
    # After the A/C split fix, E_true should stay in the same ballpark as the
    # target theta+eps -- large systematic drift would signal a design bug.
    master_seed = cfg["master_seed"]
    diffs = []
    for world_id in range(200):
        world = worldgen.generate_world(master_seed, world_id, cfg)
        doc = worldgen.generate_root_for_world(world, 0, master_seed, cfg)
        target = world.theta + doc.eps
        diffs.append(float(doc.e_true) - target)
    mean_abs_diff = sum(abs(d) for d in diffs) / len(diffs)
    assert mean_abs_diff < 5.0  # rounding-only residual, should be tiny vs sigma=50


def test_entity_names_avoid_blocklist(cfg):
    master_seed = cfg["master_seed"]
    for world_id in range(300):
        world = worldgen.generate_world(master_seed, world_id, cfg)
        assert not worldgen._contains_blocklisted_fragment(world.entity_name)


def test_worlds_are_reproducible(cfg):
    master_seed = cfg["master_seed"]
    w1 = worldgen.generate_world(master_seed, 42, cfg)
    w2 = worldgen.generate_world(master_seed, 42, cfg)
    assert w1.entity_name == w2.entity_name
    assert w1.theta == w2.theta

    d1 = worldgen.generate_root_for_world(w1, 0, master_seed, cfg)
    d2 = worldgen.generate_root_for_world(w2, 0, master_seed, cfg)
    assert d1.e_true == d2.e_true
    assert d1.text == d2.text


def test_theta_positive_and_in_plausible_range(cfg):
    master_seed = cfg["master_seed"]
    thetas = []
    for world_id in range(500):
        world = worldgen.generate_world(master_seed, world_id, cfg)
        assert world.theta > 0
        thetas.append(world.theta)
    mean_theta = sum(thetas) / len(thetas)
    assert 400 < mean_theta < 600  # loose sanity band around prior_mean=500


def test_distinct_worlds_get_distinct_documents(cfg):
    master_seed = cfg["master_seed"]
    texts = set()
    for world_id in range(50):
        world = worldgen.generate_world(master_seed, world_id, cfg)
        doc = worldgen.generate_root_for_world(world, 0, master_seed, cfg)
        texts.add(doc.text)
    assert len(texts) == 50  # no accidental collisions


def test_document_length_in_brief_target_range(cfg):
    # Section 2: "short memo (150-250 words)"
    master_seed = cfg["master_seed"]
    for world_id in range(60):
        world = worldgen.generate_world(master_seed, world_id, cfg)
        doc = worldgen.generate_root_for_world(world, 0, master_seed, cfg)
        n_words = len(doc.text.split())
        assert 140 <= n_words <= 260, f"world {world_id}: {n_words} words out of range"


def test_multiple_roots_per_world_differ(cfg):
    master_seed = cfg["master_seed"]
    world = worldgen.generate_world(master_seed, 3, cfg)
    docs = [worldgen.generate_root_for_world(world, r, master_seed, cfg) for r in range(5)]
    e_values = {d.e_true for d in docs}
    assert len(e_values) == 5  # distinct roots produce distinct primitive evidence
    assert all(d.theta == world.theta for d in docs)  # same underlying Theta
