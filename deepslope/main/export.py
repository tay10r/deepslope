from pathlib import Path
import sys

from loguru import logger

from torch import nn
import torch
import torch.onnx

from deepslope.main.state import State
from deepslope.main.config import Config
from deepslope.main.models import ModelRegistry

metadata_template = """{
  "input_size": %d,
  "t": %d,
  "beta_max": %f,
  "beta_min": %f
}
"""


def export(state: State, args):
    config = Config(**state.configs[state.current_config])
    best_path = Path(config.tmp_path) / 'best.pt'
    if not best_path.exists():
        logger.error('Best model does not exist.')
        sys.exit(1)
    registry = ModelRegistry()
    model_info = registry.info(config.model_name)
    model: nn.Module = registry.get(config.model_name)
    model.load_state_dict(torch.load(str(best_path), weights_only=True))
    model.eval()
    x = torch.randn(config.batch_size, 1, model_info.input_size, model_info.input_size, requires_grad=False)
    out = model(x)
    metadata_path = str(Path(config.tmp_path) / f'{config.model_name}.json')
    with open(metadata_path, 'w') as f:
        f.write(metadata_template % (model_info.input_size, config.diffusion_steps, config.beta_max, config.beta_min))
    model_path = str(Path(config.tmp_path) / f'{config.model_name}.onnx')
    torch.onnx.export(model,
                      x,
                      model_path,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['noisy_terrain'],
                      output_names=['denoised_terrain'],
                      dynamic_axes={'noisy_terrain': {0: 'batch_size'},
                                    'denoised_terrain': {0: 'batch_size'}})
