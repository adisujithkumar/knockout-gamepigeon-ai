"""Tests for ELO rating system."""

import pytest

from knockout.training.elo_rating import expected_score, update_elo, ELOTracker


class TestELOFunctions:
    """Tests for standalone ELO functions."""

    def test_expected_score_equal_ratings(self):
        """Equal ratings should give expected score of 0.5."""
        assert expected_score(1000.0, 1000.0) == pytest.approx(0.5)
        assert expected_score(1500.0, 1500.0) == pytest.approx(0.5)

    def test_expected_score_higher_rated(self):
        """Higher rated player should have expected score > 0.5."""
        score = expected_score(1200.0, 1000.0)
        assert score > 0.5
        assert score < 1.0

        # And the lower rated should be the complement
        score_low = expected_score(1000.0, 1200.0)
        assert score + score_low == pytest.approx(1.0)

    def test_update_elo_win(self):
        """Winning should increase ELO."""
        new_rating = update_elo(1000.0, 1000.0, score_a=1.0, k=32.0)
        assert new_rating > 1000.0
        # For equal ratings, expected is 0.5, so gain = 32 * (1.0 - 0.5) = 16
        assert new_rating == pytest.approx(1016.0)


class TestELOTracker:
    """Tests for ELOTracker class."""

    def test_elo_tracker_register_and_get(self):
        """Test registering agents and getting ratings."""
        tracker = ELOTracker(initial_rating=1000.0)

        tracker.register("alice")
        tracker.register("bob")

        assert tracker.get_rating("alice") == 1000.0
        assert tracker.get_rating("bob") == 1000.0

        # Unknown agent returns initial rating
        assert tracker.get_rating("unknown") == 1000.0

        # Registering again should not reset
        tracker.ratings["alice"] = 1200.0
        tracker.register("alice")
        assert tracker.get_rating("alice") == 1200.0

    def test_elo_tracker_update_and_leaderboard(self):
        """Test updating ratings and getting leaderboard."""
        tracker = ELOTracker(initial_rating=1000.0, k=32.0)

        tracker.register("alice")
        tracker.register("bob")

        # Alice wins
        new_a, new_b = tracker.update("alice", "bob", score_a=1.0)
        assert new_a > 1000.0
        assert new_b < 1000.0
        assert new_a == pytest.approx(1016.0)
        assert new_b == pytest.approx(984.0)

        # Check leaderboard
        leaderboard = tracker.get_leaderboard()
        assert leaderboard[0][0] == "alice"
        assert leaderboard[1][0] == "bob"
        assert leaderboard[0][1] > leaderboard[1][1]

        # Check history
        assert len(tracker.history) == 1
        assert tracker.history[0]["a"] == "alice"
        assert tracker.history[0]["score_a"] == 1.0
