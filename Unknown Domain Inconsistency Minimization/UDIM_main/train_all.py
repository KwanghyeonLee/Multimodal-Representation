import argparse
import collections
import random
import sys
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
from sconf import Config
from prettytable import PrettyTable

from domainbed.datasets import get_dataset
from domainbed import hparams_registry
from domainbed.lib import misc
from domainbed.lib.writers import get_writer
from domainbed.lib.logger import Logger
from domainbed.trainer import train, warmup_train, main_train


def main():
    parser = argparse.ArgumentParser(description="Domain generalization")
    parser.add_argument("name", type=str)
    parser.add_argument("configs", nargs="*")
    parser.add_argument("--data_dir", type=str, default="datadir/")
    parser.add_argument("--dataset", type=str, default="PACS")
    parser.add_argument("--algorithm", type=str, default="ERM")
    parser.add_argument(
        "--trial_seed",
        type=int,
        default=0,
        help="Trial number (used for seeding split_dataset and random_hparams).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for everything else")
    parser.add_argument(
        "--steps", type=int, default=None, help="Number of steps. Default is dataset-dependent."
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=None,
        help="Checkpoint every N steps. Default is dataset-dependent.",
    )
    parser.add_argument("--test_envs", type=int, nargs="+", default=None)  # sketch in PACS
    parser.add_argument("--is_single", type=bool, default=False) # single domain generalization
    parser.add_argument("--holdout_fraction", type=float, default=0.2)
    parser.add_argument("--model_save", default=None, type=int, help="Model save start step")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--tb_freq", default=10)
    parser.add_argument("--debug", action="store_true", help="Run w/ debug mode")
    parser.add_argument("--show", action="store_true", help="Show args and hparams w/o run")
    parser.add_argument(
        "--evalmode",
        default="fast",
        help="[fast, all]. if fast, ignore train_in datasets in evaluation time.",
    )
    parser.add_argument("--prebuild_loader", action="store_true", help="Pre-build eval loaders")

    '''IADA arguments'''
    parser.add_argument("--warmup_model_save", default=True, help='Save warmup checkpoint')
    parser.add_argument("--sam_warm_up_ratio", type=int,  default=3)
    parser.add_argument("--syn_lr", type=float,  default=0.005)
    parser.add_argument("--worst_weight", type=float,  default=0.5)
    parser.add_argument("--batch_size_change",  type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--batch_size_change_during", type=bool, default=False)
    parser.add_argument("--batch_size_during", type=int, default=32)
    parser.add_argument("--check_loss_value",  type=bool, default=False)
    parser.add_argument("--use_dpp",  type=bool, default=False)
    parser.add_argument("--domain_wise_inconsistency", type=bool, default=False)
    parser.add_argument("--advstyle", default=False, help='Save warmup checkpoint')



    args, left_argv = parser.parse_known_args()

    # setup hparams
    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)

    keys = ["config.yaml"] + args.configs
    keys = [open(key, encoding="utf8") for key in keys]
    hparams = Config(*keys, default=hparams)
    hparams.argv_update(left_argv)

    if args.batch_size_change:
        hparams["batch_size"] = args.batch_size

    # setup debug
    if args.debug:
        args.checkpoint_freq = 5
        args.steps = 10
        args.name += "_debug"
    torch.backends.cudnn.benchmark = True
    # timestamp = misc.timestamp()
    args.unique_name = args.algorithm+'_seed'+str(args.seed)



    if args.advstyle == 'True':
        args.advstyle = True
    else:
        args.advstyle = False


    if args.is_single == 'True':
        args.is_single = True
    elif args.is_single == True:
        args.is_single = True
    elif args.is_single == 'False':
        args.is_single = False
    else:
        args.is_single = False

    if args.is_single:
        args.domain_wise_inconsistency = False
    else:
        args.domain_wise_inconsistency = True

    if args.algorithm == 'IADA_DG' or args.algorithm == 'IADA_DG_FINAL':
        args.unique_name+='_syn_lr'+str(args.syn_lr)
        args.unique_name += '_lambda' + str(args.worst_weight)
        args.unique_name += '_warmup' + str(args.sam_warm_up_ratio)
        args.unique_name += '_batch_size'+ str(hparams["batch_size"])
        args.unique_name += '_steps'+ str(args.steps)
        args.unique_name += '_advstyle' + str(args.advstyle)


    if args.algorithm == 'ITTA' or args.algorithm == 'RIDG':
        args.unique_name+='_resnet18'
        args.unique_name += '_batch_size' + str(hparams["batch_size"])
    # path setup
    args.work_dir = Path(".")
    args.data_dir = Path(args.data_dir)

    if args.is_single:
        args.out_root = args.work_dir / Path("train_output/SDG") / args.dataset
    else:
        args.out_root = args.work_dir / Path("train_output/MDG") / args.dataset

    args.out_dir = args.out_root / args.unique_name
    args.out_dir.mkdir(exist_ok=True, parents=True)

    writer = get_writer(args.out_root / "runs" / args.unique_name)
    logger = Logger.get(args.out_dir / "log.txt")
    if args.debug:
        logger.setLevel("DEBUG")
    logger.setLevel("DEBUG")
    cmd = " ".join(sys.argv)
    logger.info(f"Command :: {cmd}")

    logger.nofmt("Environment:")
    logger.nofmt("\tPython: {}".format(sys.version.split(" ")[0]))
    logger.nofmt("\tPyTorch: {}".format(torch.__version__))
    logger.nofmt("\tTorchvision: {}".format(torchvision.__version__))
    logger.nofmt("\tCUDA: {}".format(torch.version.cuda))
    logger.nofmt("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    logger.nofmt("\tNumPy: {}".format(np.__version__))
    logger.nofmt("\tPIL: {}".format(PIL.__version__))

    # Different to DomainBed, we support CUDA only.
    assert torch.cuda.is_available(), "CUDA is not available"

    logger.nofmt("Args:")
    for k, v in sorted(vars(args).items()):
        logger.nofmt("\t{}: {}".format(k, v))

    logger.nofmt("HParams:")
    for line in hparams.dumps().split("\n"):
        logger.nofmt("\t" + line)

    if args.show:
        exit()

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = not args.deterministic

    # Dummy datasets for logging information.
    # Real dataset will be re-assigned in train function.
    # test_envs only decide transforms; simply set to zero.
    dataset, _in_splits, _out_splits = get_dataset([0], args, hparams)

    # print dataset information
    logger.nofmt("Dataset:")
    logger.nofmt(f"\t[{args.dataset}] #envs={len(dataset)}, #classes={dataset.num_classes}")
    for i, env_property in enumerate(dataset.environments):
        logger.nofmt(f"\tenv{i}: {env_property} (#{len(dataset[i])})")
    logger.nofmt("")

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
    logger.info(f"n_steps = {n_steps}")
    logger.info(f"checkpoint_freq = {checkpoint_freq}")

    org_n_steps = n_steps
    n_steps = (n_steps // checkpoint_freq) * checkpoint_freq + 1
    logger.info(f"n_steps is updated to {org_n_steps} => {n_steps} for checkpointing")

    if not args.test_envs:
        if args.is_single:
            args.test_envs = []
            for tr in range(len(dataset)):
                tmp = list(range(len(dataset)))
                tmp.remove(tr)
                args.test_envs.append(tmp)
            # batch size should be modified
            hparams["batch_size"] *= len(tmp) 
        else:
            args.test_envs = [[te] for te in range(len(dataset))]
    logger.info(f"Target test envs = {args.test_envs}")

    ###########################################################################
    # Run
    ###########################################################################
    all_records = []
    results = collections.defaultdict(list)

    for test_env in args.test_envs:
        if args.algorithm=='IADA_DG_FINAL':
            # warmup
            args.algorithm='SAM'
            _, records = warmup_train(
                test_env,
                args=args,
                hparams=hparams,
                n_steps=n_steps//args.sam_warm_up_ratio,
                checkpoint_freq=checkpoint_freq,
                logger=logger,
                writer=writer,
            )
            all_records.append(records)
            print('Our model starts')
            # main training
            args.algorithm='IADA_DG_FINAL'
            res, records = main_train(
                test_env,
                args=args,
                hparams=hparams,
                start_steps=n_steps//args.sam_warm_up_ratio,
                n_steps=n_steps,
                checkpoint_freq=checkpoint_freq,
                logger=logger,
                writer=writer,
            )
            all_records.append(records)
            for k, v in res.items():
                results[k].append(v)
        
        else:
            res, records = train(
                test_env,
                args=args,
                hparams=hparams,
                n_steps=n_steps,
                checkpoint_freq=checkpoint_freq,
                logger=logger,
                writer=writer,
            )
            all_records.append(records)
            for k, v in res.items():
                results[k].append(v)

    # log summary table
    logger.info("=== Summary ===")
    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.info("Unique name: %s" % args.unique_name)
    logger.info("Out path: %s" % args.out_dir)
    logger.info("Algorithm: %s" % args.algorithm)
    logger.info("Dataset: %s" % args.dataset)

    table = PrettyTable(["Selection"] + dataset.environments + ["Avg."])
    for key, row in results.items():
        row.append(np.mean(row))
        row = [f"{acc:.3%}" for acc in row]
        table.add_row([key] + row)
    logger.nofmt(table)


if __name__ == "__main__":
    main()
