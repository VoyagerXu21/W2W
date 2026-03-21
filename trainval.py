import argparse
from utils.config import get_exp_config, print_arguments


def resolve_train_mode(args, cfg):
    """Resolve the requested training pipeline.

    Priority:
    1. CLI --train_mode
    2. config.train_mode
    3. legacy config.run_rl
    4. default to sft
    """
    if getattr(args, "train_mode", None):
        return args.train_mode.lower()

    cfg_mode = getattr(cfg, "train_mode", None)
    if cfg_mode:
        return str(cfg_mode).lower()

    if getattr(cfg, "run_rl", False):
        return "ppo"

    return "sft"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default="./config/config.json", type=str, help="Config file path.")
    parser.add_argument('--dataset', default="eth", type=str, help="Dataset name.", choices=["eth", "hotel", "univ", "zara1", "zara2"])
    parser.add_argument('--tag', default="LMTraj", type=str, help="Personal tag for the model.")
    parser.add_argument('--test', default=False, action='store_true', help="Evaluation mode.")
    parser.add_argument('--train_mode', default=None, type=str, choices=['sft', 'ppo'],
                        help="Training pipeline to launch. Overrides train_mode in config when provided.")

    args = parser.parse_args()

    print("===== Arguments =====")
    print_arguments(vars(args))

    print("===== Configs =====")
    cfg = get_exp_config(args.cfg)
    print_arguments(cfg)

    # Update configs
    cfg.dataset_name = args.dataset
    cfg.checkpoint_name = args.tag
    cfg.train_mode = resolve_train_mode(args, cfg)
    cfg.run_rl = cfg.train_mode == 'ppo'

    print(f"===== Resolved train mode: {cfg.train_mode} =====")

    if not args.test:
        # Training phase
        if cfg.train_mode == 'sft':
            from model.trainval_accelerator_SFT import trainval
        elif cfg.train_mode == 'ppo':
            from model.trainval_accelerator import trainval
        else:
            raise ValueError(f"Unsupported train_mode: {cfg.train_mode}. Expected 'sft' or 'ppo'.")
        trainval(cfg)

    else:
        # Evaluation phase
        from model.eval_accelerator import test
        test(cfg)
