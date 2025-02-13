from dataclasses import dataclass


@dataclass
class Config:
    tmp_path: str = 'tmp'
    model_name: str = 'terrain-diffuse-64'
    seed: int = 0
    batch_size: int = 32
    learning_rate: float = 0.0001
    beta_max: float = 0.02
    beta_min: float = 0.0001
    diffusion_steps: int = 1000
