"""Reward modules for knockout training."""

from knockout.reward.contrastive import (
    ContrastiveConfig,
    ContrastiveRewardShaper,
    ContrastiveTrainer,
)

from knockout.reward.attention_discovery import (
    AttentionConfig,
    AttentionRewardShaper,
    AttentionTrainer,
)

from knockout.reward.llm_architect import (
    LLMArchitectConfig,
    LLMArchitectTrainer,
)

from knockout.reward.coevolution import (
    CoevolutionConfig,
    RewardCoevolution,
)

__all__ = [
    "ContrastiveConfig",
    "ContrastiveRewardShaper",
    "ContrastiveTrainer",
    "AttentionConfig",
    "AttentionRewardShaper",
    "AttentionTrainer",
    "LLMArchitectConfig",
    "LLMArchitectTrainer",
    "CoevolutionConfig",
    "RewardCoevolution",
]
