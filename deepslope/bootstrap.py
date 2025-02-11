from pathlib import Path

from deepslope.config import get_config, Config
from deepslope.state import load_global_state, store_global_state

if __name__ == '__main__':
    # Also creates a config, if it does not exist.
    config = get_config()
    state_path = str(Path(config.tmp_path) / 'state.json')
    state = load_global_state(state_path)
    store_global_state(state_path, state)
