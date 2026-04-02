"""Tests for training visualization utilities."""

import csv
import os
import tempfile

import pytest

from knockout.training.plots import plot_training_curves, plot_win_rates, save_logs_csv


class TestPlots:
    """Tests for plotting functions."""

    def test_plot_training_curves(self):
        """Test training curves are saved to file."""
        logs = [
            {"rollout": 0, "policy_loss": 0.5, "value_loss": 1.0, "entropy": 2.0},
            {"rollout": 1, "policy_loss": 0.4, "value_loss": 0.9, "entropy": 1.9},
            {"rollout": 2, "policy_loss": 0.3, "value_loss": 0.8, "entropy": 1.8},
        ]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name

        try:
            plot_training_curves(logs, save_path=path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_plot_win_rates(self):
        """Test win rate plot is saved to file."""
        logs = [
            {"generation": 0, "win_rate": 0.3},
            {"generation": 1, "win_rate": 0.5},
            {"generation": 2, "win_rate": 0.7},
        ]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name

        try:
            plot_win_rates(logs, save_path=path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_save_logs_csv(self):
        """Test CSV log saving."""
        logs = [
            {"rollout": 0, "policy_loss": 0.5, "value_loss": 1.0},
            {"rollout": 1, "policy_loss": 0.4, "value_loss": 0.9},
        ]

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name

        try:
            save_logs_csv(logs, path)
            assert os.path.exists(path)

            with open(path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2
            assert rows[0]["rollout"] == "0"
            assert rows[0]["policy_loss"] == "0.5"
        finally:
            os.unlink(path)
