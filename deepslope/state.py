from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class GlobalState:
    num_experiments: int = 0


def load_global_state(filename: str) -> GlobalState:
    s = GlobalState()
    path = Path(filename)
    if path.exists():
        with open(filename, 'r') as f:
            s = GlobalState(**json.load(f))
    return s


def store_global_state(filename: str, s: GlobalState):
    with open(filename, 'w') as f:
        json.dump(asdict(s), f, indent=2)
