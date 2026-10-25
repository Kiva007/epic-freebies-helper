# -*- coding: utf-8 -*-
"""Static regression tests for the deterministic "complete the circle" solver.

The repository forbids test execution, so these cases are documentation of the
expected behaviour and are verified by targeted non-test scripts instead.
"""
from __future__ import annotations

import numpy as np
import pytest

from extensions.hcaptcha_adapter import (
    _is_ring_completion_question,
    _ransac_ring,
    _ring_template_variants,
)


class TestRingCompletionGate:
    @pytest.mark.parametrize(
        "question",
        ["Drag the icon to complete the circle", "Please DRAG THE ICON TO COMPLETE THE CIRCLE now"],
    )
    def test_matches_circle_questions(self, question: str) -> None:
        assert _is_ring_completion_question(question) is True

    @pytest.mark.parametrize(
        "question",
        [
            "Select the 3 ARROWS that are aimed at the shape",
            "Drag each piece to match the outline",
            "Please complete the line segment 4",
        ],
    )
    def test_rejects_other_questions(self, question: str) -> None:
        assert _is_ring_completion_question(question) is False


class TestRansacRing:
    def test_recovers_circle_from_evenly_spaced_points(self) -> None:
        centre = (200.0, 295.0)
        radius = 108.0
        # Eleven of twelve slots are filled; slot 1 is the gap.
        angles = [index * 30.0 for index in range(12) if index != 1]
        points = np.array(
            [
                [
                    centre[0] + radius * np.cos(np.radians(a)),
                    centre[1] + radius * np.sin(np.radians(a)),
                ]
                for a in angles
            ],
            dtype=float,
        )
        ring, cx, cy, fitted_radius = _ransac_ring(points)
        assert len(ring) == len(points)
        assert abs(cx - centre[0]) < 1.0
        assert abs(cy - centre[1]) < 1.0
        assert abs(fitted_radius - radius) < 1.0

    def test_ignores_stray_points_outside_the_ring(self) -> None:
        centre = (200.0, 295.0)
        radius = 108.0
        ring_points = np.array(
            [
                [
                    centre[0] + radius * np.cos(np.radians(a)),
                    centre[1] + radius * np.sin(np.radians(a)),
                ]
                for a in (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
            ],
            dtype=float,
        )
        noisy = np.vstack([ring_points, np.array([[441.0, 185.0], [30.0, 440.0]])])
        ring, cx, cy, fitted_radius = _ransac_ring(noisy)
        assert len(ring) == len(ring_points)
        assert abs(fitted_radius - radius) < 1.0

    def test_returns_none_for_too_few_points(self) -> None:
        assert _ransac_ring(np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 9.0]])) is None


class TestRingTemplateVariants:
    def test_produces_rotations_at_three_scales(self) -> None:
        icon = np.zeros((70, 70, 4), dtype=np.uint8)
        icon[:, :, 3] = 255
        variants = _ring_template_variants(icon)
        shapes = {variant.shape for variant, _angle in variants}
        assert shapes == {(60, 60), (70, 70), (80, 80)}
        assert len({angle for _variant, angle in variants}) == 36

    def test_alpha_is_composited_onto_neutral_gray(self) -> None:
        icon = np.zeros((70, 70, 4), dtype=np.uint8)
        icon[:, :, :3] = 255
        icon[:, :, 3] = 255
        variant, _angle = _ring_template_variants(icon)[36]  # scale 1.0, angle 0
        assert int(variant.min()) > 100
        assert int(variant.max()) <= 255
