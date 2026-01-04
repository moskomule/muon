import dataclasses
from collections import deque
from collections.abc import Callable, MutableMapping
from functools import partial

import torch
from muons.orthogonalization import (
    NEWTON_SCHULZ_DEFAULT_COEFFICIENTS,
    POLAR_EXPRESS_DEFAULT_COEFFICIENTS,
    newton_schulz,
    polar_express,
)
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.optim._muon import _adjust_lr
from torch.optim.optimizer import Optimizer, ParamsT

SUPPORTED_BACKENDS = {
    "newton_schulz": newton_schulz,
    "polar_express": polar_express,
}

DEFAULT_COEFFICIENTS = {
    "newton_schulz": NEWTON_SCHULZ_DEFAULT_COEFFICIENTS,
    "polar_express": POLAR_EXPRESS_DEFAULT_COEFFICIENTS,
}


class Muon(Optimizer):
    """Muon optimizer from PyTorch."""

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        coefficients: list[tuple[float, float, float]] = None,
        eps: float = 1e-7,
        steps: int = 5,
        adjust_lr_fn: str | None = None,
        backend: str | None = None,
    ) -> None:
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")
        if adjust_lr_fn is not None and adjust_lr_fn not in ["original", "match_rms_adamw"]:
            raise ValueError(f"Adjust learning rate function {adjust_lr_fn} is not supported")
        if backend is not None and backend not in ["newton_schulz", "polar_express"]:
            raise ValueError(f"Implementation {backend} is not supported")
        if coefficients is None:
            coefficients = DEFAULT_COEFFICIENTS[backend]

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "coefficients": coefficients,
            "eps": eps,
            "steps": steps,
            "adjust_lr_fn": adjust_lr_fn,
            "implementation": backend,
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon only supports 2D parameters whereas we found a parameter with size: {p.size()}"
                    )

    def _init_group(
        self,
        group: MutableMapping,
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        muon_momentum_bufs: list[Tensor],
    ):
        for p in group["params"]:
            if p.grad is None:
                continue

            if torch.is_complex(p):
                raise RuntimeError("Muon does not support complex parameters")
            if p.grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")

            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]

            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
            muon_momentum_bufs.append(state["momentum_buffer"])

        return False  # has_complex

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            orthogonalization = partial(
                SUPPORTED_BACKENDS[group["implementation"]],
                coefficients=group["coefficients"],
                steps=group["steps"],
                eps=group["eps"],
            )

            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            muon_momentum_bufs: list[Tensor] = []

            self._init_group(
                group,
                params_with_grad,
                grads,
                muon_momentum_bufs,
            )

            if isinstance(params_with_grad[0], DTensor):
                if params_with_grad[0].device_mesh.ndim != 1:
                    raise NotImplementedError()
                results = deque[DMuonResult]()

                for i, param in enumerate(params_with_grad):
                    world_size = param.device_mesh.size()
                    results.append(
                        DMuonResult(
                            param,
                            grads[i],
                            muon_momentum_bufs[i],
                            lr,
                            momentum,
                            weight_decay,
                            group["nesterov"],
                            group["adjust_lr_fn"],
                            orthogonalization,
                            index=i,
                        )
                    )
                    if len(results) >= world_size:
                        results.popleft().wait()

                for result in results:
                    result.wait()

            else:
                for i, param in enumerate(params_with_grad):
                    MuonResult(
                        param,
                        grads[i],
                        muon_momentum_bufs[i],
                        lr,
                        momentum,
                        weight_decay,
                        group["nesterov"],
                        group["adjust_lr_fn"],
                        orthogonalization,
                    ).wait()

        return loss


@dataclasses.dataclass
class MuonResult:
    param: Tensor
    grad: Tensor
    buf: Tensor
    lr: float | Tensor
    momentum: float | Tensor
    weight_decay: float | Tensor
    nesterov: bool
    adjust_lr_fn: str
    orthogonalization: Callable[[Tensor], Tensor]

    def __post_init__(self):
        self.buf.lerp_(self.grad, 1 - self.momentum)
        self.grad = self.grad.lerp(self.buf, self.momentum) if self.nesterov else self.buf

    def wait(self) -> None:
        update = self.orthogonalization(self.grad)
        adjusted_lr = _adjust_lr(self.lr, self.adjust_lr_fn, self.param.shape)
        self.param.mul_(1 - self.lr * self.weight_decay)
        self.param.add_(update, alpha=-adjusted_lr)


@dataclasses.dataclass
class DMuonResult(MuonResult):
    index: int

    def __post_init__(self):
        super().__post_init__()
        rank = self.grad.device_mesh.get_rank()
        world_size = self.grad.device_mesh.size()
        pg = self.grad.device_mesh.get_group()
        dest_rank = self.index % world_size

        local_grad = self.grad.to_local()

        if rank == dest_rank:
            gather_list = [torch.empty_like(local_grad) for _ in range(world_size)]
        else:
            gather_list = None

        self.gather_handle = torch.distributed.gather(
            local_grad,
            gather_list,
            group_dst=dest_rank,
            group=pg,
            async_op=True,
        )
        self.gather_list = gather_list
        self.dist_info = (rank, world_size, pg, dest_rank)

    def wait(self) -> None:

        rank, world_size, pg, dest_rank = self.dist_info
        self.gather_handle.wait()
        if rank == dest_rank:
            full_grad = torch.cat(self.gather_list, dim=0)
            full_grad.copy_(self.orthogonalization(full_grad))
            full_grad = full_grad.type_as(self.grad)
            chunks = list(full_grad.chunk(world_size, dim=0))
        else:
            chunks = None

        torch.distributed.scatter(
            self.grad.to_local(),
            chunks,
            src=dest_rank,
            group=pg,
            async_op=False,
        )
        adjusted_lr = _adjust_lr(self.lr, self.adjust_lr_fn, self.param.shape)
        self.param.mul_(1 - self.lr * self.weight_decay)
        self.param.add_(self.grad, alpha=-adjusted_lr)
