import pytest
import torch


@pytest.mark.parametrize(
    "backend",
    ["newton_schulz", "polar_express"],
)
def test_muon(backend):
    # Simple test to check if Muon optimizer works with different backends
    from muon import Muon

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


def test_grouped_optimizer():
    from muon.utils import GroupedOptimizer

    model = torch.nn.Linear(10, 10)
    optim1 = torch.optim.SGD([model.weight], lr=0.1)
    optim2 = torch.optim.Adam([model.bias], lr=0.1)
    grouped_optim = GroupedOptimizer(optim1, optim2)
    scheduler = torch.optim.lr_scheduler.StepLR(grouped_optim, step_size=1, gamma=0.1)
    x = torch.randn(5, 10)
    y = model(x)
    loss = y.sum()
    loss.backward()
    grouped_optim.step()
    grouped_optim.zero_grad()
    scheduler.step()

    assert scheduler.get_last_lr()[0] == 0.01  # Check if learning rate is adjusted correctly
