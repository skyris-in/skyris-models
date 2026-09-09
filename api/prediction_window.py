from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

ModelResult = tuple[int, float]
ModelDeque = deque[ModelResult]
VisionPrediction = dict[str, float | int]


@dataclass
class DeviceWindowState:
    windows: dict[str, ModelDeque]
    vision_preds: dict[str, VisionPrediction] = field(default_factory=dict)
    has_image: bool = False


class SlidingWindowManager:
    """Maintains per-device sliding windows for model outputs."""

    def __init__(self, window_size: int, model_names: list[str]):
        self.window_size: int = window_size
        self.model_names: list[str] = model_names
        self._devices: dict[str, DeviceWindowState] = {}

    def _get(self, device_id: str) -> DeviceWindowState:
        if device_id not in self._devices:
            self._devices[device_id] = DeviceWindowState(
                windows={name: deque(maxlen=self.window_size) for name in self.model_names}
            )
        return self._devices[device_id]

    def add_tabular_result(self, device_id: str, model_name: str, pred: int, prob: float) -> None:
        state = self._get(device_id)
        state.windows[model_name].append((pred, prob))

    def set_vision_predictions(self, device_id: str, vision_preds: dict[str, VisionPrediction]) -> None:
        state = self._get(device_id)
        state.vision_preds = vision_preds
        state.has_image = True

    def is_ready(self, device_id: str) -> bool:
        state = self._get(device_id)
        return all(len(q) >= self.window_size for q in state.windows.values())

    def compute_prediction(self, device_id: str) -> tuple[int, float] | tuple[None, None]:
        state = self._get(device_id)
        if not self.is_ready(device_id):
            return None, None

        tabular_votes = [max(pred for pred, _ in q) for q in state.windows.values()]
        all_votes = tabular_votes.copy()

        if state.has_image:
            all_votes += [v["predicted_class"] for v in state.vision_preds.values()]

        final_prediction = 1 if sum(all_votes) >= (len(all_votes) // 2 + 1) else 0

        probs = [sum(prob for _, prob in q) / len(q) for q in state.windows.values()]
        final_probability = sum(probs) / len(probs)

        # Keep deques for rolling window; image predictions are one-window hints.
        state.has_image = False

        return final_prediction, final_probability

    def current_window_size(self, device_id: str) -> int:
        state = self._get(device_id)
        return len(next(iter(state.windows.values())))
