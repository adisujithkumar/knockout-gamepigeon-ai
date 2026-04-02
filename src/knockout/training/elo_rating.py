"""ELO rating system for agent evaluation."""

from __future__ import annotations


def expected_score(rating_a: float, rating_b: float) -> float:
    """Compute expected score of player A."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating_a: float, rating_b: float, score_a: float, k: float = 32.0) -> float:
    """Update player A's ELO."""
    return rating_a + k * (score_a - expected_score(rating_a, rating_b))


class ELOTracker:
    """Track ELO ratings for multiple agents."""

    def __init__(self, initial_rating: float = 1000.0, k: float = 32.0):
        self.initial_rating = initial_rating
        self.k = k
        self.ratings: dict[str, float] = {}
        self.history: list[dict] = []

    def register(self, name: str) -> None:
        """Register a new agent with initial rating."""
        if name not in self.ratings:
            self.ratings[name] = self.initial_rating

    def get_rating(self, name: str) -> float:
        """Get current rating for an agent."""
        return self.ratings.get(name, self.initial_rating)

    def update(self, name_a: str, name_b: str, score_a: float) -> tuple[float, float]:
        """Update ratings after a match. Returns (new_a, new_b)."""
        self.register(name_a)
        self.register(name_b)

        old_a = self.ratings[name_a]
        old_b = self.ratings[name_b]

        self.ratings[name_a] = update_elo(old_a, old_b, score_a, self.k)
        self.ratings[name_b] = update_elo(old_b, old_a, 1.0 - score_a, self.k)

        self.history.append(
            {
                "a": name_a,
                "b": name_b,
                "score_a": score_a,
                "new_a": self.ratings[name_a],
                "new_b": self.ratings[name_b],
            }
        )

        return self.ratings[name_a], self.ratings[name_b]

    def get_leaderboard(self) -> list[tuple[str, float]]:
        """Return agents sorted by rating (highest first)."""
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
