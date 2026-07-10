"""Tests for embedding diff: identical spaces (overlap ~= 1), unrelated spaces
(overlap ~= 0), and a controlled partial perturbation (overlap in a known band,
movers surface the perturbed points). Also covers Procrustes alignment and
projection."""

from __future__ import annotations

import numpy as np

import synthetic
from ml_xray.embed import EmbeddingDiff
from ml_xray.embed.align import orthogonal_procrustes
from ml_xray.embed.project import project_2d


def test_identical_spaces_full_overlap():
    planted = synthetic.identical_spaces()
    report = EmbeddingDiff(k=10).fit(planted.emb_a, planted.emb_b).report()
    assert report.neighbor_overlap > 0.99
    assert report.cluster_stability > 0.99
    assert float(report.per_point_drift.max()) < 1e-9


def test_unrelated_spaces_near_zero_overlap():
    planted = synthetic.unrelated_spaces()
    report = EmbeddingDiff(k=10).fit(planted.emb_a, planted.emb_b).report()
    assert report.neighbor_overlap < 0.2
    assert report.cluster_stability < 0.2


def test_partial_perturbation_band_and_movers():
    planted = synthetic.partially_perturbed_spaces(n=300, moved=30)
    report = EmbeddingDiff(k=10).fit(planted.emb_a, planted.emb_b).report()
    # Most points unchanged, a minority moved -> overlap sits in a middle band.
    assert 0.5 < report.neighbor_overlap < 0.95

    moved = set(planted.moved_ids.tolist())
    top_movers = set(report.movers[: len(moved)])
    recall = len(top_movers & moved) / len(moved)
    assert recall >= 0.8, f"movers recall too low: {recall:.2f}"


def test_different_dimensionality_is_supported():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(120, 32))
    b = rng.normal(size=(120, 8))
    report = EmbeddingDiff(k=5).fit(a, b).report()
    assert 0.0 <= report.neighbor_overlap <= 1.0


def test_row_count_mismatch_raises():
    a = np.zeros((10, 4))
    b = np.zeros((9, 4))
    try:
        EmbeddingDiff().fit(a, b)
    except ValueError:
        return
    raise AssertionError("expected ValueError on row-count mismatch")


def test_procrustes_recovers_rotation():
    rng = np.random.default_rng(0)
    source = rng.normal(size=(50, 3))
    # A known rotation about the z-axis.
    theta = 0.7
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    target = source @ rot
    aligned, r = orthogonal_procrustes(source, target, allow_reflection=False)
    assert np.allclose(aligned, target, atol=1e-8)
    assert np.isclose(np.linalg.det(r), 1.0, atol=1e-6)


def test_project_2d_shared_basis_shapes():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(60, 10))
    b = rng.normal(size=(60, 10))
    pa, pb = project_2d(a, b, projector="pca")
    assert pa.shape == (60, 2)
    assert pb.shape == (60, 2)


def test_report_movers_use_ids():
    planted = synthetic.partially_perturbed_spaces(n=120, moved=10)
    ids = [f"item-{i}" for i in range(120)]
    report = EmbeddingDiff(k=8).fit(planted.emb_a, planted.emb_b, ids=ids).report()
    assert all(isinstance(m, str) for m in report.movers)
    moved_ids = {f"item-{i}" for i in planted.moved_ids.tolist()}
    assert len(set(report.movers[:10]) & moved_ids) >= 7
