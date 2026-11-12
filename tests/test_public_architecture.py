"""Synthetic checks only; these settings and inputs are not the paper protocol."""
import dataclasses, inspect, sys, unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import numpy as np
import torch
from architecture_smoke import toy
from heidi import (HEIDI, FitConfig, ModelNumerics, ConditionalWGANSequenceCritic,
                   endpoint_recovery, endpoint_stats, mc_endpoint, fit_two_stage, construction_rates)
from heidi.metrics import combine_predictive_samples
from heidi.communication import Scenario, simulate_packets, _sample_burst_state
from heidi.evolution import fit_shared_qf

class PublicImplementationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(101)
        torch.set_num_threads(1)
        self.model, self.batch, self.edge = toy()

    def test_observations_and_endpoint_mean_scale(self):
        with torch.inference_mode():
            full = self.model(**self.batch, edge_index=self.edge, sample_bnn=False)
        mask = self.batch["input_mask"]
        self.assertTrue(torch.equal(full.completed_mean[mask], self.batch["input_values"][mask]))
        mean, scale = endpoint_stats(self.model, self.batch, self.edge, sample=False)
        torch.testing.assert_close(mean, full.completed_mean[:, -1], atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(scale, full.reconstruction_std[:, -1], atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(endpoint_recovery(self.model, self.batch, self.edge, sample=False), mean, atol=3e-6, rtol=3e-6)

    def test_sampled_endpoint_matches_full(self):
        torch.manual_seed(103)
        with torch.inference_mode():
            expected = self.model(**self.batch, edge_index=self.edge, sample_bnn=True)
        torch.manual_seed(103)
        mean, scale = endpoint_stats(self.model, self.batch, self.edge, sample=True)
        torch.testing.assert_close(mean, expected.completed_mean[:, -1], atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(scale, expected.reconstruction_std[:, -1], atol=3e-6, rtol=3e-6)

    def test_mc_reference_random_stream(self):
        torch.manual_seed(107)
        with torch.inference_mode():
            outputs = [self.model(**self.batch, edge_index=self.edge, sample_bnn=True) for _ in range(3)]
        expected = combine_predictive_samples([o.completed_mean[:, -1] for o in outputs],
                                               [o.reconstruction_std[:, -1] for o in outputs])
        torch.manual_seed(107)
        actual = mc_endpoint(self.model, self.batch, self.edge, samples=3, preserve_rng=True)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, atol=4e-6, rtol=4e-6)

    def test_causal_history(self):
        modified = {k: v.clone() for k, v in self.batch.items()}
        modified["input_values"][:, -1] += 11
        modified["edge_values"][:, -1] += 13
        modified["global_values"][:, -1] += 17
        with torch.inference_mode():
            first = self.model(**self.batch, edge_index=self.edge, sample_bnn=False)
            second = self.model(**modified, edge_index=self.edge, sample_bnn=False)
        torch.testing.assert_close(first.completed_mean[:, :-1], second.completed_mean[:, :-1], atol=0, rtol=0)

    def test_private_config_has_no_defaults(self):
        for cls in (FitConfig, ModelNumerics, Scenario):
            for field in dataclasses.fields(cls):
                self.assertIs(field.default, dataclasses.MISSING)
                self.assertIs(field.default_factory, dataclasses.MISSING)
        for key in ["hidden_dim", "graph_layers"]:
            self.assertIs(inspect.signature(HEIDI).parameters[key].default, inspect.Parameter.empty)

    def test_deadline_denominator_and_explicit_gate(self):
        rates = construction_rates(torch.tensor([1., 3., float("inf")]), torch.tensor([.9, .4, .99]),
            torch.tensor([0., 2., 4.]), acceptance_probability=.8, strict_acceptance=True)
        self.assertEqual(rates[0], 0)
        self.assertAlmostEqual(rates[1], 1/3, places=6)
        self.assertAlmostEqual(rates[2], 1/3, places=6)
        for strict, expected in [(True, 0.), (False, 1.)]:
            rate = construction_rates(torch.tensor([1.]), torch.tensor([.5]), torch.tensor([2.]),
                acceptance_probability=.5, strict_acceptance=strict)
            self.assertEqual(rate, [expected])

    def test_packet_visibility_and_shared_state(self):
        config = Scenario(name="synthetic", late_probability=0., drop_probability=0.,
            minimum_late_steps=2, maximum_late_steps=2, device_extra_delay_steps=0,
            device_extra_delay_jitter=0, burst_good_to_bad_probability=1.,
            burst_bad_to_good_probability=0., burst_late_probability=1.,
            burst_drop_probability=0., shared_burst_state=True)
        states = _sample_burst_state(np.random.default_rng(109), 5, 3, config)
        self.assertTrue(np.array_equal(states[:, 0], states[:, 1]))
        events, latency, visible = simulate_packets(time_steps=5, device_count=3,
            parameter_count=2, scenario=config, seed=109, deadline_steps=1, straggler_device_index=0)
        self.assertTrue(visible[0].all())
        self.assertFalse(visible[1:].any())
        self.assertTrue((latency[1:] == 2).all())

    def test_two_stage_training_and_freezing(self):
        config = FitConfig(ddgn_epochs=1, bnn_epochs=1, learning_rate=.0043,
            critic_learning_rate=.0027, weight_decay=.000031, critic_steps=1,
            gradient_penalty_weight=1.7, ddgn_reconstruction_weight=.8,
            ddgn_adversarial_weight=.13, bnn_reconstruction_weight=.9,
            bnn_forecast_weight=.7, bnn_kl_weight=.000023, validation_mc_samples=3,
            test_mc_samples=3, critic_beta1=.31, critic_beta2=.87, gradient_clip_norm=2.3)
        critic = ConditionalWGANSequenceCritic(node_feature_dim=2, edge_feature_dim=1,
            global_feature_dim=1, hidden_dim=7, negative_slope=.11)
        batch = dict(self.batch)
        batch.update(target_values=torch.randn_like(batch["input_values"]),
                     hidden_target_mask=~batch["input_mask"], forecast_target=torch.randn_like(batch["input_values"]))
        def evaluate(predictor, loader, samples):
            output = predictor(next(iter(loader)))
            value = float(output.completed_mean.detach().square().mean())
            return {"endpoint_reconstruction_rmse": value, "endpoint_forecast_rmse": value}
        result = fit_two_stage(self.model, critic, train_loader=[batch], validation_loader=[batch],
            edge_index=self.edge, device=torch.device("cpu"), config=config, evaluate=evaluate,
            select_validation=lambda scores: scores["endpoint_reconstruction_rmse"])
        self.assertEqual([row["stage"] for row in result.history], ["ddgn_wgan_gp", "bnn_refiner"])
        self.assertTrue(all(not p.requires_grad for p in self.model.ddgn.parameters()))
        self.assertTrue(all(not p.requires_grad for p in critic.parameters()))
        self.assertTrue(all(p.requires_grad for p in self.model.bnn.parameters()))

    def test_shared_evolution_explicit_calibration(self):
        data = SimpleNamespace(normalized_episodes={
            "train": [SimpleNamespace(target_values=torch.randn(6, 3, 2))],
            "val": [SimpleNamespace(target_values=torch.randn(5, 3, 2))]})
        qf = fit_shared_qf(data, epochs=1, learning_rate=.0047, hidden_dim=7, weight_decay=.000037,
            batch_size=2, calibration_quantile=.73, temperature_factor=.31, temperature_floor=.00017)
        self.assertEqual(tuple(qf.q_predict(torch.randn(2, 3, 2)).shape), (2, 3, 2))
        probability = qf.true_probability(torch.tensor([0., 1.]))
        self.assertTrue(((probability >= 0) & (probability <= 1)).all())

if __name__ == "__main__":
    unittest.main()
