import numpy as np

import config
import embedding


def test_cosine_identical_is_one():
    v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    assert embedding.cosine_similarity(v, v) > 0.9999


def test_cosine_orthogonal_is_zero():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(embedding.cosine_similarity(a, b)) < 1e-6


def test_cosine_opposite_is_negative_one():
    a = np.array([1.0, 1.0], dtype=np.float32)
    b = np.array([-1.0, -1.0], dtype=np.float32)
    assert embedding.cosine_similarity(a, b) < -0.9999


def test_verdict_bands():
    assert embedding.verdict(config.FACE_CONFIRMED + 0.01) == "confirmed"
    assert embedding.verdict((config.FACE_CONFIRMED + config.FACE_POSSIBLE) / 2) == "possible"
    assert embedding.verdict(config.FACE_REJECTED - 0.01) == "different"
    assert embedding.verdict(None) == "unknown"
