import torch


class GroupedOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        *optimizers: torch.optim.Optimizer,
    ):
        super().__init__(sum([optim.param_groups for optim in optimizers], []), {})
        self.optimizers = optimizers

    def step(self) -> None:
        for optim in self.optimizers:
            optim.step()

    def zero_grad(
        self,
        set_to_none: bool = True,
    ) -> None:
        for optim in self.optimizers:
            optim.zero_grad(set_to_none=set_to_none)
