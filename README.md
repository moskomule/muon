# muon

PyTorch implementation of Muon Optimizer with Newton-Schulz and Polar Express methods supporting FSDP2.

![pytest](https://github.com/moskomule/muon/workflows/pytest/badge.svg)
[![document](https://img.shields.io/static/v1?label=doc&message=muon&color=blue)](https://moskomule.github.io/muon)

## Installation


```
uv pip install git+https://github.com/moskomule/muon.git
```

## Usage

### Basic Example

```python
from muon import Muon

model = ...  # Your PyTorch model

# alomost identical to torch.optim.Muon
optimizer = Muon(model.parameters(), 
                 lr=0.01, 
                 backend="newton_schulz", # or "polar_express"
                 )
```

### Advanced Example with FSDP2

You don't need to do anything special for FSDP2, so it may not be "Advanced".

```python
from muon import Muon
from torch.distributed.fsdp import fully_shard

model = ...  # Your PyTorch model
fully_shard(model, mesh, ...)

optimizer = Muon(model.parameters(), 
                 lr=0.01, 
                 backend="newton_schulz", # or "polar_express"
                 )
```

### Maybe useful helper class

```python
from muon import Muon
from muon.utils import GroupedOptimizer

model = ...  # Your PyTorch model
optimizer1 = Muon(model.muon_params(), ...)
optimizer2 = torch.optim.AdamW(model.non_muon_params(), ...)
optimizer = GroupedOptimizer(optimizer1, optimizer2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, ...)

for batch in data_loader:
    ...
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
```

## Acknowledgements

This implementation is heavily inspired by:
- [PyTorch's Muon](https://github.com/pytorch/pytorch/blob/main/torch/optim/_muon.py)
- [samsja's muon_fsdp_2](https://github.com/samsja/muon_fsdp_2/blob/main/src/muon/muon_fsdp2.py)