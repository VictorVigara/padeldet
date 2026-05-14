"""Kalman filter for 2D ball tracking with constant-velocity model."""

from __future__ import annotations

import numpy as np


class BallKalmanFilter:
    """Kalman filter with state [x, y, vx, vy].

    Constant-velocity transition model. Observes position (x, y) only.
    Designed for ball tracking in heatmap space (288x512).
    """

    def __init__(
        self,
        q_pos: float = 1.0,
        q_vel: float = 5.0,
        r_obs: float = 5.0,
    ):
        self._q_pos = q_pos
        self._q_vel = q_vel
        self._r_obs = r_obs

        # Transition matrix: constant velocity, dt=1 frame
        self._F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # Observation matrix: observe x, y
        self._H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Process noise
        self._Q = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float64)

        # Measurement noise
        self._R = np.diag([r_obs, r_obs]).astype(np.float64)

        # State
        self._x: np.ndarray | None = None  # [x, y, vx, vy]
        self._P: np.ndarray | None = None  # 4x4 covariance

        # Predicted state (after predict(), before update())
        self._x_pred: np.ndarray | None = None
        self._P_pred: np.ndarray | None = None
        self._S: np.ndarray | None = None  # innovation covariance

        self._consecutive_misses = 0

    @property
    def is_initialized(self) -> bool:
        return self._x is not None

    @property
    def consecutive_misses(self) -> int:
        return self._consecutive_misses

    @property
    def predicted_position(self) -> tuple[float, float] | None:
        if self._x_pred is not None:
            return (float(self._x_pred[0]), float(self._x_pred[1]))
        if self._x is not None:
            return (float(self._x[0]), float(self._x[1]))
        return None

    def initialize(self, x: float, y: float) -> None:
        """Initialize filter with first observation. Velocity starts at zero."""
        self._x = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        self._P = np.diag([self._r_obs, self._r_obs, 100.0, 100.0]).astype(np.float64)
        self._consecutive_misses = 0
        self._x_pred = None
        self._P_pred = None
        self._S = None

    def predict(self) -> tuple[float, float]:
        """Predict next state. Returns predicted (x, y).

        Must be called before update() or gating_distance() each frame.
        """
        self._x_pred = self._F @ self._x
        self._P_pred = self._F @ self._P @ self._F.T + self._Q
        self._S = self._H @ self._P_pred @ self._H.T + self._R
        return (float(self._x_pred[0]), float(self._x_pred[1]))

    def update(self, x: float, y: float) -> None:
        """Incorporate measurement and advance state."""
        z = np.array([x, y], dtype=np.float64)
        y_innov = z - self._H @ self._x_pred
        S_inv = np.linalg.inv(self._S)
        K = self._P_pred @ self._H.T @ S_inv

        self._x = self._x_pred + K @ y_innov
        I = np.eye(4)
        self._P = (I - K @ self._H) @ self._P_pred
        self._consecutive_misses = 0

    def missed_update(self) -> None:
        """No measurement this frame. State advances via prediction only."""
        self._x = self._x_pred
        self._P = self._P_pred
        self._consecutive_misses += 1

    def gating_distance(self, x: float, y: float) -> float:
        """Mahalanobis distance of (x, y) from predicted position.

        Lower = more consistent with predicted trajectory.
        Must call predict() first.
        """
        z = np.array([x, y], dtype=np.float64)
        y_innov = z - self._H @ self._x_pred
        S_inv = np.linalg.inv(self._S)
        d2 = float(y_innov @ S_inv @ y_innov)
        return np.sqrt(max(d2, 0.0))

    def reset(self) -> None:
        """Reset filter to uninitialized state."""
        self._x = None
        self._P = None
        self._x_pred = None
        self._P_pred = None
        self._S = None
        self._consecutive_misses = 0
