"""Ranking waitlisted clients with an epsilon-greedy bandit.

Each client is an arm. When a slot opens we offer it to one client at a time,
and the reward is how fast they took it: 1.0 for an instant yes, falling to 0
at the end of the offer window, and 0 for a no or no reply. Offering first to
the clients most likely to say yes quickly fills slots sooner and texts fewer
people who will pass.

- Exploit: with probability 1 - epsilon, pick the highest estimated reward.
- Explore: with probability epsilon, pick a random remaining client, so a
  client who was slow once is not buried forever.
- Prior: every client starts at ``prior_mean`` with weight ``prior_weight``,
  so a new client is neither ignored nor trusted on one data point.
- Ties go to whoever joined the waitlist first, which keeps it fair.
"""

import random
import uuid
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Arm:
    client_id: uuid.UUID
    offers: int
    reward_sum: float
    waiting_since: float  # timestamp; earlier wins ties


def response_reward(seconds_to_yes: float | None, window_seconds: float) -> float:
    """1.0 for an instant yes, linearly down to 0.0 at the window's end."""
    if seconds_to_yes is None:
        return 0.0
    return max(0.0, 1.0 - seconds_to_yes / window_seconds)


class EpsilonGreedy:
    def __init__(
        self,
        epsilon: float = 0.1,
        *,
        prior_mean: float = 0.5,
        prior_weight: float = 1.0,
        rng: random.Random | None = None,
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")
        self.epsilon = epsilon
        self.prior_mean = prior_mean
        self.prior_weight = prior_weight
        self._rng = rng or random.Random()

    def estimate(self, arm: Arm) -> float:
        return (arm.reward_sum + self.prior_mean * self.prior_weight) / (
            arm.offers + self.prior_weight
        )

    def choose(self, arms: Sequence[Arm]) -> Arm:
        if not arms:
            raise ValueError("no arms to choose from")
        if self._rng.random() < self.epsilon:
            return self._rng.choice(list(arms))
        return max(arms, key=lambda a: (self.estimate(a), -a.waiting_since))
