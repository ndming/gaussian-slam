import torch.multiprocessing as mp
from argparse import ArgumentParser

from slam import run_slam
from view import vis_slam


def main(args, queue):
    p_slam = mp.Process(target=run_slam, args=(args.config, queue))
    p_view = mp.Process(target=vis_slam, args=(args.port, queue))

    p_slam.start()
    p_view.start()

    p_slam.join()
    queue.put(None) # signal the viewer to stop
    p_view.join()


if __name__ == "__main__":
    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to the config file.")
    parser.add_argument("--port", type=int, default=8082, help="Port for the viewer server.")
    args = parser.parse_args()

    # Some multiprocessing setups
    mp.set_start_method("spawn", force=True)
    queue = mp.Queue(maxsize=1) # single snapshot buffer

    # Start SLAM and viewer processes
    main(args, queue)
