"""Configurable delayed communication for independent harbor platforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinkConfig:
    """Shared directed-link behavior for a communication experiment."""

    enabled: bool = True
    range: float = 20.0
    update_interval_steps: int = 1
    delay_steps: int = 0
    message_ttl_steps: int = 5
    dropout_probability: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.range <= 0.0:
            raise ValueError("communication range must be positive")
        if (
            self.update_interval_steps <= 0
            or self.delay_steps < 0
            or self.message_ttl_steps < 0
        ):
            raise ValueError(
                "communication interval must be positive; delay and TTL nonnegative"
            )
        if not 0.0 <= self.dropout_probability <= 1.0:
            raise ValueError("communication dropout_probability must be in [0, 1]")


@dataclass(frozen=True)
class AgentMessage:
    """Platform-neutral state and intent observation."""

    sender: str
    sent_step: int
    position: np.ndarray
    velocity: np.ndarray
    goal: np.ndarray
    cruise_speed: float


class CommunicationNetwork:
    """Broadcast network that changes information, never physical state."""

    def __init__(self, names: list[str], config: LinkConfig):
        self.names = tuple(names)
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self._pending: list[tuple[int, str, AgentMessage]] = []
        self._latest: dict[str, dict[str, AgentMessage]] = {
            receiver: {} for receiver in names
        }
        self.sent_count = 0
        self.delivered_count = 0
        self.dropped_count = 0

    def exchange(
        self,
        step: int,
        observations: dict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray, float]
        ],
    ) -> dict[str, dict[str, AgentMessage]]:
        """Broadcast due observations and return each receiver's latest inbox."""
        if self.config.enabled and step % self.config.update_interval_steps == 0:
            for sender, (position, velocity, goal, cruise_speed) in observations.items():
                message = AgentMessage(
                    sender=sender,
                    sent_step=step,
                    position=np.asarray(position, dtype=float).copy(),
                    velocity=np.asarray(velocity, dtype=float).copy(),
                    goal=np.asarray(goal, dtype=float).copy(),
                    cruise_speed=float(cruise_speed),
                )
                for receiver in self.names:
                    if receiver == sender:
                        continue
                    self.sent_count += 1
                    receiver_position = observations[receiver][0]
                    if np.linalg.norm(position - receiver_position) > self.config.range:
                        self.dropped_count += 1
                        continue
                    if self._rng.random() < self.config.dropout_probability:
                        self.dropped_count += 1
                        continue
                    self._pending.append(
                        (step + self.config.delay_steps, receiver, message)
                    )

        remaining = []
        for delivery_step, receiver, message in self._pending:
            if delivery_step <= step:
                self._latest[receiver][message.sender] = message
                self.delivered_count += 1
            else:
                remaining.append((delivery_step, receiver, message))
        self._pending = remaining
        for inbox in self._latest.values():
            expired = [
                sender
                for sender, message in inbox.items()
                if step - message.sent_step > self.config.message_ttl_steps
            ]
            for sender in expired:
                del inbox[sender]
        return {receiver: inbox.copy() for receiver, inbox in self._latest.items()}
