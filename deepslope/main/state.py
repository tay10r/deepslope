from dataclasses import dataclass, field


@dataclass
class State:
    configs: dict[str, dict] = field(default_factory=dict)
    current_config: str | None = None
    datasets: dict[str, str] = field(default_factory=dict)
    current_dataset: str | None = None
