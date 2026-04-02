"""Shared fixtures and markers for knockout-v2 tests."""

import numpy as np
import pymunk
import pytest


@pytest.fixture
def physics_space() -> pymunk.Space:
    """Create a clean Pymunk space for testing."""
    space = pymunk.Space()
    space.gravity = (0.0, 0.0)
    space.damping = 0.98
    return space


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(42)
