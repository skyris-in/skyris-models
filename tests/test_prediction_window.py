import os
import sys
import unittest


CURRENT_DIR = os.path.dirname(__file__)
BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from api.prediction_window import SlidingWindowManager


class SlidingWindowManagerTests(unittest.TestCase):
    def test_prediction_starts_at_window_and_continues(self):
        manager = SlidingWindowManager(window_size=5, model_names=["a", "b", "c"])
        device = "node-1"

        # First 4 points -> not ready
        for _ in range(4):
            manager.add_tabular_result(device, "a", 1, 0.9)
            manager.add_tabular_result(device, "b", 0, 0.1)
            manager.add_tabular_result(device, "c", 1, 0.8)
            pred, prob = manager.compute_prediction(device)
            self.assertIsNone(pred)
            self.assertIsNone(prob)

        # 5th point -> ready
        manager.add_tabular_result(device, "a", 1, 0.9)
        manager.add_tabular_result(device, "b", 0, 0.2)
        manager.add_tabular_result(device, "c", 1, 0.7)
        pred, prob = manager.compute_prediction(device)
        self.assertIn(pred, (0, 1))
        self.assertIsInstance(prob, float)

        # 6th point -> still ready (sliding, no reset)
        manager.add_tabular_result(device, "a", 0, 0.3)
        manager.add_tabular_result(device, "b", 0, 0.4)
        manager.add_tabular_result(device, "c", 1, 0.8)
        pred2, prob2 = manager.compute_prediction(device)
        self.assertIn(pred2, (0, 1))
        self.assertIsInstance(prob2, float)
        self.assertEqual(manager.current_window_size(device), 5)

    def test_per_device_isolation(self):
        manager = SlidingWindowManager(window_size=3, model_names=["a", "b"])

        # Device A fills window
        for _ in range(3):
            manager.add_tabular_result("A", "a", 1, 0.8)
            manager.add_tabular_result("A", "b", 1, 0.7)

        # Device B has only one point
        manager.add_tabular_result("B", "a", 0, 0.2)
        manager.add_tabular_result("B", "b", 0, 0.3)

        pred_a, prob_a = manager.compute_prediction("A")
        pred_b, prob_b = manager.compute_prediction("B")

        self.assertIn(pred_a, (0, 1))
        self.assertIsInstance(prob_a, float)
        self.assertIsNone(pred_b)
        self.assertIsNone(prob_b)


if __name__ == "__main__":
    _ = unittest.main()
