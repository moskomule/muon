import dataclasses
from collections import defaultdict, deque
from collections.abc import Callable, MutableMapping
from functools import partial
from typing import Literal

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.optim._muon import _adjust_lr
from torch.optim.optimizer import Optimizer, ParamsT

from .orthogonalization import (
    newton_schulz,
    newton_schulz_coefficients,
    polar_express_coefficients,
)

DEFAULT_COEFFICIENTS = {
    "newton_schulz": newton_schulz_coefficients,
    "polar_express": polar_express_coefficients,
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
        backend: Literal["newton_schulz", "polar_express"] = "newton_schulz",
        compile: bool = True,  # Whether to compile the orthogonalization function
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
        if backend not in ["newton_schulz", "polar_express"]:
            raise ValueError(f"Implementation {backend} is not supported")
        if coefficients is None:
            if backend == "newton_schulz":
                coefficients = DEFAULT_COEFFICIENTS["newton_schulz"](steps)
            elif backend == "polar_express":
                coefficients = DEFAULT_COEFFICIENTS["polar_express"](
                    l=1e-3, num_iters=steps, safety_factor_eps=1e-2, cushion=0.01
                )

        # force params: list[dict[str, Any]]
        params = list(params)
        if isinstance(params[0], Tensor):
            params = [{"params": params}]

        # group params by their size
        # e.g., [{"params": [...], "size": (128, 64), **kwargs}, {"params": [...], "size": (64, 32), **kwargs}]

        _params = []
        for group in params:
            size_to_group = defaultdict(list)
            for p in group["params"]:
                size = p.size()
                if len(size) != 2:
                    raise ValueError(f"Muon only supports 2D parameters whereas we found a parameter with size: {size}")
                size_to_group[size].append(p)
            for size, p_list in size_to_group.items():
                _params.append({**group, "params": p_list, "size": size})

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "eps": eps,
            "adjust_lr_fn": adjust_lr_fn,
        }

        super().__init__(_params, defaults)

        self.orthogonalization = partial(
            newton_schulz,
            coefficients=coefficients,
            eps=eps,
        )
        self.compile = compile

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
            # each group has same size params

            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

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
                            torch.compile(
                                self.orthogonalization,
                                disable=not self.compile,
                                fullgraph=True,
                            ),
                            index=i,
                        )
                    )
                    if len(results) >= world_size:
                        results.popleft().wait()

                for result in results:
                    result.wait()

            else:
                stacked_params = torch.stack(params_with_grad)
                MuonResult(
                    stacked_params,
                    torch.stack(grads),
                    torch.stack(muon_momentum_bufs),
                    lr,
                    momentum,
                    weight_decay,
                    group["nesterov"],
                    group["adjust_lr_fn"],
                    torch.compile(
                        torch.vmap(self.orthogonalization),
                        disable=not self.compile,
                        fullgraph=True,
                    ),
                ).wait()

                torch._foreach_copy_(params_with_grad, stacked_params.unbind())

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
