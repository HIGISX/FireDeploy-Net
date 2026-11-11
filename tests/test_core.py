"""Checks of reward consistency, feasibility, policy gradients and state loading."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from examples.make_synthetic import make_instance
from firedeploy.data import Data, utility
from firedeploy.environment import BatchEnvironment
from firedeploy.model import FireDeployNet
from firedeploy.rollout import Runner
from firedeploy.runtime import evaluate, setup
from firedeploy.train import policy_update


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup(20261107, 1)

    def test_utility_thresholds(self):
        u = utility(np.array([0, 5, 10, 20, 25]))
        np.testing.assert_array_equal(u[[0, 1, 3, 4]], [1, 1, 0, 0])
        self.assertTrue(0 < u[2] < 1)

    def test_reward_and_directed_budget(self):
        for budget in [0.0, 0.3, 3.0]:
            with tempfile.TemporaryDirectory() as tmp, Data(make_instance(tmp, budget)) as data:
                env = BatchEnvironment(data, 4)
                rng = np.random.default_rng(19)
                for _ in range(3 * data.K):
                    legal, *_ = env.observe()
                    env.step(np.array([rng.choice(np.flatnonzero(row)) for row in legal]))
                values, layouts, distances = env.finish()
                for value, layout, distance in zip(values, layouts, distances):
                    result = data.evaluate(layout)
                    self.assertTrue(result["feasible"])
                    self.assertAlmostEqual(result["F"], value, places=11)
                    self.assertAlmostEqual(result["D_km"], distance, places=11)
                    if budget == 0:
                        self.assertEqual(set(layout[0]), set(layout[1]))
                        self.assertEqual(set(layout[1]), set(layout[2]))

    def test_policy_training_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp, Data(make_instance(Path(tmp) / "data")) as data:
            runner = Runner(data, "cpu")
            model = FireDeployNet()
            self.assertEqual(torch.count_nonzero(model.adapt_output.weight), 0)
            first = evaluate(model, runner)
            second = evaluate(model, runner)
            self.assertEqual(first["selected_ids"], second["selected_ids"])
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            original = {k: v.detach().clone() for k, v in model.state_dict().items()}
            record = policy_update(model, optimizer, runner, 4, 2)
            self.assertTrue(np.isfinite(record["loss"]))
            self.assertTrue(
                any(not torch.equal(original[k], v) for k, v in model.state_dict().items())
            )
            other = FireDeployNet()
            other.load_state_dict(model.state_dict(), strict=True)
            self.assertEqual(
                evaluate(model, runner)["selected_ids"], evaluate(other, runner)["selected_ids"]
            )


if __name__ == "__main__":
    unittest.main()
