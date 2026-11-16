"""The epsilon-greedy bandit: unit behaviour, and that it actually learns."""

import random
import uuid

import pytest

from frontdesk_agent.frontdesk.bandit import Arm, EpsilonGreedy, response_reward


def arm(offers: int = 0, reward: float = 0.0, since: float = 0.0) -> Arm:
    return Arm(uuid.uuid4(), offers, reward, since)


def test_reward_is_speed() -> None:
    assert response_reward(0, 900) == 1.0
    assert response_reward(450, 900) == 0.5
    assert response_reward(2000, 900) == 0.0
    assert response_reward(None, 900) == 0.0


def test_greedy_picks_best_estimate_and_ties_go_to_longest_waiting() -> None:
    b = EpsilonGreedy(0.0, rng=random.Random(1))
    slow, fast = arm(10, 1.0), arm(10, 9.0)
    assert b.choose([slow, fast]) is fast
    early, late = arm(since=1.0), arm(since=2.0)
    assert b.choose([late, early]) is early


def test_prior_keeps_one_lucky_answer_from_dominating() -> None:
    b = EpsilonGreedy(0.0, prior_mean=0.5, prior_weight=1.0)
    lucky_once = arm(1, 1.0)  # (1 + 0.5) / 2 = 0.75
    steady = arm(20, 16.0)  # (16 + 0.5) / 21 = 0.79
    assert b.choose([lucky_once, steady]) is steady


def test_epsilon_explores() -> None:
    b = EpsilonGreedy(1.0, rng=random.Random(3))
    arms = [arm(10, 9.0), arm(10, 0.0)]
    picks = {b.choose(arms).client_id for _ in range(50)}
    assert len(picks) == 2


def test_bad_epsilon_is_rejected() -> None:
    with pytest.raises(ValueError):
        EpsilonGreedy(1.5)


def simulate(strategy: str, rounds: int = 400, seed: int = 7) -> float:
    """Clients answer an offer with different probabilities and speeds. Each
    round one slot opens and we offer it until someone takes it; the score is
    the mean reward of the person offered *first*, the one the policy chose."""
    rng = random.Random(seed)
    p_yes = [0.9, 0.7, 0.3, 0.1, 0.05, 0.05]  # unknown to the policy
    speed = [0.2, 0.4, 0.6, 0.8, 0.9, 0.9]  # fraction of the window used
    offers = [0] * len(p_yes)
    rewards = [0.0] * len(p_yes)
    ids = [uuid.uuid4() for _ in p_yes]
    order = list(range(len(p_yes)))
    rng.shuffle(order)  # waitlist order is unrelated to responsiveness
    bandit = EpsilonGreedy(0.1, rng=random.Random(seed + 1))
    total = 0.0
    for _ in range(rounds):
        if strategy == "bandit":
            arms = [Arm(ids[i], offers[i], rewards[i], order.index(i)) for i in order]
            first = ids.index(bandit.choose(arms).client_id)
        elif strategy == "fifo":
            first = order[0]
        else:
            first = rng.randrange(len(p_yes))
        r = (1 - speed[first]) if rng.random() < p_yes[first] else 0.0
        offers[first] += 1
        rewards[first] += r
        total += r
    return total / rounds


def test_bandit_beats_first_come_first_served_and_random() -> None:
    bandit, fifo, rand = simulate("bandit"), simulate("fifo"), simulate("random")
    assert bandit > fifo and bandit > rand
    assert bandit > 0.6  # close to the best client's 0.9 * 0.8 = 0.72
