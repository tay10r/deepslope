from torch import nn

from deepslope.monitoring.monitor import Monitor


def compute_avg_grad(module: nn.Module) -> dict[str, float]:
    params = module.named_parameters()
    result: dict[str, float] = {}
    for name, param in params:
        result[name] = abs(param.grad).sum().item()
    return result


def visit_grad(module: nn.Module, monitor: Monitor):
    g = compute_avg_grad(module)
    for name, value in g.items():
        monitor.log_grad(name, value)
