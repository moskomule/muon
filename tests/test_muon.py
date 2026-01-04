import pytest
import torch

from muon import Muon


@pytest.mark.parametrize(
    "backend",
    ["newton_schulz", "polar_express"],
)
def test_muon(backend):
    # Simple test to check if Muon optimizer works with different backends

    model = torch.nn.Linear(10, 10)
    optimizer = Muon(
        model.parameters(),
        backend=backend,
    )
    x = torch.randn(5, 10)
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    assert 1 == 1  # Dummy assertion to ensure the test runs without errors
