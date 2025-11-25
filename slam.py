from pathlib import Path

from src.entities.gaussian_slam import GaussianSLAM
from src.utils.io_utils import load_config
from src.utils.utils import setup_seed

def run_slam(config_path, queue):
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file {config_file} does not exist.")
    
    config = load_config(config_path)
    setup_seed(config.get("seed", 42))

    gslam = GaussianSLAM(config, queue)
    gslam.run()

    queue.put(None)     # signal consumers to exit
    queue.close()       # close the queue endpoint
    queue.join_thread() # flush the queue

    print("SLAM process finished.")
    print("Viewer is still running, please terminate it manually via Ctrl+C.")
    
