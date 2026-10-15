"""CPU checks for the formal flow field, shared probes and conformal cutoff."""

import unittest
from fractions import Fraction
from types import SimpleNamespace

import torch

from backend.flow.calibration import _calculate_threshold
from backend.flow.features import _evaluate_probe_field
from backend.flow.model import FlowAdapter, model_signature
from backend.flow.probes import _collect, _time_grid


class FlowModelTests(unittest.TestCase):
    def test_model_signature_ignores_scheduler_default_order(self):
        metadata = {"model_id": "checkpoint", "guidance_scale": 7.5,
                    "scheduler": {"shift": 3.0,
                                  "_use_default_values": ["max_shift", "base_shift"]}}
        reordered = {**metadata, "scheduler": {**metadata["scheduler"],
                                                "_use_default_values": ["base_shift", "max_shift"]}}
        self.assertEqual(model_signature(metadata), model_signature(reordered))
        changed = {**metadata, "scheduler": {**metadata["scheduler"], "shift": 4.0}}
        self.assertNotEqual(model_signature(metadata), model_signature(changed))

    def test_transformer_output_is_reoriented_and_cfg_effect_subtracts_null(self):
        class Transformer:
            dtype = torch.float32

            def __call__(self, hidden_states, timestep, encoder_hidden_states,
                         pooled_projections, return_dict):
                self.timesteps = timestep.clone()
                values = encoder_hidden_states[:, 0, 0].reshape(-1, 1, 1, 1)
                return (values.expand_as(hidden_states),)

        adapter = FlowAdapter.__new__(FlowAdapter)
        adapter.pipe = SimpleNamespace(transformer=Transformer())
        adapter.device = "cpu"
        adapter.shape = (1, 1, 1)
        adapter.num_train_timesteps = 1000
        adapter.guidance_scale = 3.0
        adapter.null_embeddings = (torch.zeros(1, 1, 1), torch.zeros(1, 1))
        positive = (torch.full((1, 1, 1), 2.0), torch.ones(1, 1))
        x = torch.ones(2, 1, 1, 1)
        self.assertTrue(torch.allclose(adapter.prompt_effect(x, 0.25, positive),
                                       torch.full_like(x, -6.0)))
        self.assertTrue(torch.allclose(adapter.guided_velocity(x, 0.25, positive),
                                       torch.full_like(x, -6.0)))
        self.assertTrue(torch.equal(adapter.pipe.transformer.timesteps,
                                    torch.full((4,), 750.0)))

    def test_midpoint_grid_and_shared_trajectory_states(self):
        nodes, midpoints = _time_grid(3, 4)
        self.assertEqual(midpoints, [Fraction(1, 6), Fraction(1, 2), Fraction(5, 6)])
        self.assertTrue(set(midpoints).issubset(nodes))

        class Adapter:
            device = "cpu"
            shape = (1, 1, 1)

            def encode(self, prompt):
                return prompt

            def guided_velocity(self, x, t, embeddings):
                return torch.full_like(x, 2.0)

        cfg = {"probes": {"trajectories": 2, "time_points": 3,
                           "integration_steps": 4}, "experiment": {"seed": 7}}
        states, provenance, times = _collect(cfg, Adapter(), [{"id": "fit", "prompt": "p"}])
        self.assertEqual(times, [1/6, 1/2, 5/6])
        self.assertEqual([len(group) for group in states], [2, 2, 2])
        for trajectory_id in range(2):
            baseline = states[0][trajectory_id] - 2 * times[0]
            for time_id in range(3):
                self.assertTrue(torch.allclose(states[time_id][trajectory_id],
                                                baseline + 2 * times[time_id], atol=1e-6))
                record = provenance[time_id * 2 + trajectory_id]
                self.assertEqual(record["trajectory_id"], trajectory_id)
                self.assertEqual(record["prompt_id"], "fit")
                self.assertEqual(record["seed"], provenance[trajectory_id]["seed"])

    def test_signature_uses_common_midpoint_order(self):
        class Adapter:
            device = "cpu"

            def prompt_effect(self, states, time, embeddings):
                return states + time

        bank = {"states": torch.arange(4.0).reshape(4, 1, 1, 1),
                "midpoints": torch.tensor([0.25, 0.75]), "trajectories": 2}
        cfg = {"experiment": {"probe_batch_size": 1}}
        values = _evaluate_probe_field(cfg, Adapter(), bank, None)
        self.assertTrue(torch.allclose(values.flatten(),
                                       torch.tensor([0.25, 1.25, 2.75, 3.75])))

    def test_conformal_infinite_cutoff(self):
        cfg = {"experiment": {"alpha": 0.05}}
        rows = [{"distance": float(i)} for i in range(10)]
        result = _calculate_threshold(cfg, rows, "hash", ["fit"])
        self.assertIsNone(result["threshold"])
        rows = [{"distance": float(i)} for i in range(100)]
        self.assertEqual(_calculate_threshold(cfg, rows, "hash", ["fit"])["threshold"], 95.0)


if __name__ == "__main__":
    unittest.main()
