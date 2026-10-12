from __future__ import annotations

import unittest

import torch

from heidi import EndToEndBayesianDT


class PublicArchitectureTest(unittest.TestCase):
    def test_forward_preserves_observed_values(self) -> None:
        model = EndToEndBayesianDT(2, 1, 1, 8, 3, 1)
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        values = torch.randn(1, 2, 2, 2)
        mask = torch.tensor([[[[True, False], [False, True]], [[True, True], [False, False]]]])
        output = model(
            values,
            mask,
            torch.randn(1, 2, 1, 1),
            torch.randn(1, 2, 1),
            edge_index,
        )
        self.assertTrue(torch.equal(output.completed_mean[mask], values[mask]))


if __name__ == "__main__":
    unittest.main()
