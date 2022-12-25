import sys
import os
sys.path.append(os.path.dirname('/code/connectomics/'))
import argparse
import random
import numpy as np
import torch
from cilog import create_logger
torch.multiprocessing.set_start_method('spawn', force=True)
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from connectomics.config import load_cfg, save_all_cfg
from connectomics.engine import Trainer, SSL_Trainer


def get_args():
    parser = argparse.ArgumentParser(description="Model Training & Inference")
    parser.add_argument('--config-file', type=str,
                        help='configuration file (yaml)')
    parser.add_argument('--config-base', type=str,
                        help='base configuration file (yaml)', default=None)
    parser.add_argument('--inference', action='store_true',
                        help='inference mode')
    parser.add_argument('--distributed', action='store_true',
                        help='distributed training')
    parser.add_argument('--local_rank', type=int,
                        help='node rank for distributed training', default=None)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to load the checkpoint')
    parser.add_argument('--debug', action='store_true',
                        help='run the scripts in debug mode')
    # Merge configs from command line (e.g., add 'SYSTEM.NUM_GPUS 8').
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = get_args()
    args.local_rank = int(os.environ["LOCAL_RANK"]) if args.distributed else 0
    if args.local_rank == 0 or args.local_rank is None:
        print("Command line arguments: ", args)

    manual_seed = 0 if args.local_rank is None else args.local_rank
    init_seed(manual_seed)

    cfg = load_cfg(args)
    cfg_image_model = None
    args_image_model = None
    if cfg.MODEL.IMAGE_MODEL_CFG is not None:
        args_image_model = get_args()
        args_image_model.config_file = cfg.MODEL.IMAGE_MODEL_CFG
        args_image_model.checkpoint = cfg.MODEL.IMAGE_MODEL_CKPT
        assert args_image_model.checkpoint is not None
        args_image_model.inference = True
        cfg_image_model = load_cfg(args_image_model, merge_cmd=False)
    log_name = cfg.DATASET.OUTPUT_PATH + '.log'
    create_logger(name='l1', file=log_name, sub_print=True,
                  file_level='DEBUG')
    if args.local_rank == 0 or args.local_rank is None:
        # In distributed training, only print and save the
        # configurations using the node with local_rank=0.
        print("PyTorch: ", torch.__version__)
        print(cfg)

        if not os.path.exists(cfg.DATASET.OUTPUT_PATH):
            print('Output directory: ', cfg.DATASET.OUTPUT_PATH)
            os.makedirs(cfg.DATASET.OUTPUT_PATH)
            save_all_cfg(cfg, cfg.DATASET.OUTPUT_PATH)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        assert torch.cuda.is_available(), \
            "Distributed training without GPUs is not supported!"
        dist.init_process_group("nccl", init_method='env://')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Rank: {}. Device: {}".format(args.local_rank, device))
    cudnn.enabled = True
    cudnn.benchmark = True

    mode = 'test' if args.inference else 'train'
    if cfg.MODEL.SSL == 'none':
        trainer = Trainer(cfg, device, mode,
                          rank=args.local_rank,
                          checkpoint=args.checkpoint, cfg_image_model=cfg_image_model,
                          checkpoint_image_model=args_image_model.checkpoint if args_image_model is not None else None)
    else:
        trainer = SSL_Trainer(cfg, device, mode,
                              rank=args.local_rank,
                              checkpoint=args.checkpoint)

    # Start training or inference:
    if cfg.DATASET.DO_CHUNK_TITLE == 0 and not cfg.DATASET.DO_MULTI_VOLUME:
        test_func = trainer.test_one_neuron if cfg.INFERENCE.DO_SINGLY else trainer.test
        test_func() if args.inference else trainer.train()
    elif cfg.DATASET.DO_MULTI_VOLUME:
        trainer.run_multivolume(mode, rank=args.local_rank)
    else:
        trainer.run_chunk(mode)

    print("Rank: {}. Device: {}. Process is finished!".format(
          args.local_rank, device))


if __name__ == "__main__":
    main()
