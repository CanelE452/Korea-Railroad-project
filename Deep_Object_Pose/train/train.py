#!/usr/bin/python3

"""
Example usage:

 python -m torch.distributed.launch --nproc_per_node=1 train.py --data ../sample_data/ --object cracker
"""


import argparse
import datetime
import os
from queue import Queue
import random
import warnings
warnings.filterwarnings("ignore")

try:
    import configparser as configparser
except ImportError:
    import ConfigParser as configparser

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import sys
sys.path.insert(1, '../common')
from models import *
from utils import *
from geo_loss import GeometricLoss, StructuralLoss, ReliabilityLoss, VisibilityCoordLoss, SpatialSoftArgmax2D



def _runnetwork(net, optimizer, local_rank, epoch, train_loader, writer=None,
                geo_loss_module=None, geo_lambda=0.0, geo_warmup=5,
                struct_loss_module=None, struct_lambda=1.0, struct_warmup=10,
                rel_loss_module=None, rel_lambda=1.0, rel_warmup=0,
                vis_loss_module=None, vis_lambda=0.005, vis_warmup=0):
    loss_avg_to_log = {}
    loss_avg_to_log["loss"] = []
    loss_avg_to_log["loss_affinities"] = []
    loss_avg_to_log["loss_belief"] = []
    if geo_loss_module is not None:
        loss_avg_to_log["loss_geo"] = []
    if struct_loss_module is not None:
        loss_avg_to_log["loss_struct"] = []
    if rel_loss_module is not None:
        loss_avg_to_log["loss_rel"] = []
    if vis_loss_module is not None:
        loss_avg_to_log["loss_vis"] = []
    for batch_idx, targets in enumerate(train_loader):
        optimizer.zero_grad()

        data = Variable(targets["img"].cuda())
        target_belief = Variable(targets["beliefs"].cuda())
        target_affinities = Variable(targets["affinities"].cuda())

        output_belief, output_aff = net(data)

        # Shape check (first batch of first epoch only)
        if batch_idx == 0 and epoch == 0:
            print("target_belief:", tuple(target_belief.shape))
            print("target_aff   :", tuple(target_affinities.shape))
            print("output_belief:", tuple(output_belief[-1].shape))
            print("output_aff   :", tuple(output_aff[-1].shape))
            assert tuple(output_belief[-1].shape) == tuple(target_belief.shape), \
                f"Belief shape mismatch: output {tuple(output_belief[-1].shape)} vs target {tuple(target_belief.shape)}"
            assert tuple(output_aff[-1].shape) == tuple(target_affinities.shape), \
                f"Affinity shape mismatch: output {tuple(output_aff[-1].shape)} vs target {tuple(target_affinities.shape)}"

        loss = None

        loss_belief = torch.tensor(0).float().cuda()
        loss_affinities = torch.tensor(0).float().cuda()

        for stage in range(len(output_aff)):  # output, each belief map layers.
            loss_affinities += (
                (output_aff[stage] - target_affinities)
                * (output_aff[stage] - target_affinities)
            ).mean()

            if opt.symmetric_loss:
                # 180° Y-rotation swap: 0↔5, 1↔4, 2↔7, 3↔6, centroid(8) unchanged
                swap_idx = [5, 4, 7, 6, 1, 0, 3, 2, 8]
                target_swapped = target_belief[:, swap_idx]
                loss_bel_orig = (
                    (output_belief[stage] - target_belief)
                    * (output_belief[stage] - target_belief)
                ).mean()
                loss_bel_swap = (
                    (output_belief[stage] - target_swapped)
                    * (output_belief[stage] - target_swapped)
                ).mean()
                loss_belief += torch.min(loss_bel_orig, loss_bel_swap)
            else:
                loss_belief += (
                    (output_belief[stage] - target_belief)
                    * (output_belief[stage] - target_belief)
                ).mean()

        loss = loss_affinities + loss_belief

        # Geometric loss (soft-argmax + BPnP)
        loss_geo = torch.tensor(0.0).cuda()
        if geo_loss_module is not None and geo_lambda > 0:
            geo_total, geo_dict = geo_loss_module(
                output_belief[-1][:, :9], target_belief[:, :9],
                epoch=epoch, warmup=geo_warmup
            )
            loss_geo = geo_lambda * geo_total
            loss = loss + loss_geo

        # Visibility-aware coordinate loss
        loss_vis = torch.tensor(0.0).cuda()
        if vis_loss_module is not None and vis_lambda > 0:
            vis_weight = targets.get("visibility")
            if vis_weight is not None:
                vis_weight = vis_weight.cuda()
                vis_total, vis_dict = vis_loss_module(
                    output_belief[-1][:, :9], target_belief[:, :9],
                    vis_weight, epoch=epoch, warmup=vis_warmup
                )
                loss_vis = vis_lambda * vis_total
                loss = loss + loss_vis

        # Reliability-aware coordinate loss
        loss_rel = torch.tensor(0.0).cuda()
        if rel_loss_module is not None and rel_lambda > 0:
            rel_total, rel_dict = rel_loss_module(
                output_belief[-1][:, :9], target_belief[:, :9],
                epoch=epoch, warmup=rel_warmup
            )
            loss_rel = rel_lambda * rel_total
            loss = loss + loss_rel

        # Structural loss (flip equivariance + sparse edge + coord Huber)
        loss_struct = torch.tensor(0.0).cuda()
        if struct_loss_module is not None and struct_lambda > 0:
            # Create horizontally flipped input for flip equivariance
            data_flip = torch.flip(data, dims=[-1])  # flip width
            struct_total, struct_dict = struct_loss_module(
                output_belief[-1][:, :9], target_belief[:, :9],
                net=net, data_flip=data_flip,
                epoch=epoch, warmup=struct_warmup
            )
            loss_struct = struct_lambda * struct_total
            loss = loss + loss_struct

        if batch_idx == 0:
            post = "train"

            if writer is not None and local_rank == 0:
                for i_output in range(1):

                    # input images
                    writer.add_image(
                        f"{post}_input_{i_output}",
                        targets["img_original"][i_output],
                        epoch,
                        dataformats="CWH",
                    )

                    # belief maps gt
                    imgs = VisualizeBeliefMap(target_belief[i_output])
                    imgs[imgs == float('inf')] = 0
                    img, grid = save_image(
                        imgs, "belief_maps_gt.png", mean=0, std=1, nrow=3, save=False
                    )
                    writer.add_image(
                        f"{post}_belief_ground_truth_{i_output}",
                        grid,
                        epoch,
                        dataformats="CWH",
                    )

                    # belief maps guess
                    imgs = VisualizeBeliefMap(output_belief[-1][i_output])
                    imgs[imgs == float('inf')] = 0
                    img, grid = save_image(
                        imgs, "belief_maps.png", mean=0, std=1, nrow=3, save=False
                    )
                    writer.add_image(
                        f"{post}_belief_guess_{i_output}",
                        grid,
                        epoch,
                        dataformats="CWH",
                    )


        loss.backward()

        optimizer.step()

        # log the loss
        loss_avg_to_log["loss"].append(loss.item())
        loss_avg_to_log["loss_affinities"].append(loss_affinities.item())
        loss_avg_to_log["loss_belief"].append(loss_belief.item())
        if geo_loss_module is not None:
            loss_avg_to_log["loss_geo"].append(loss_geo.item())
        if struct_loss_module is not None:
            loss_avg_to_log["loss_struct"].append(loss_struct.item())
        if rel_loss_module is not None:
            loss_avg_to_log["loss_rel"].append(loss_rel.item())
        if vis_loss_module is not None:
            loss_avg_to_log["loss_vis"].append(loss_vis.item())

        # Belief peak health (every batch)
        with torch.no_grad():
            peak_vals = output_belief[-1][:, :9].view(output_belief[-1].shape[0], 9, -1).max(dim=-1).values
            if "belief_peak" not in loss_avg_to_log:
                loss_avg_to_log["belief_peak"] = []
            loss_avg_to_log["belief_peak"].append(peak_vals.mean().item())

        if batch_idx % opt.loginterval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.15f} \tLocal Rank: {}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                    local_rank,
                )
            )

    # log the loss values
    if writer is not None and local_rank == 0:
        mean_bel = np.mean(loss_avg_to_log["loss_belief"])
        writer.add_scalar(
            "loss/train_loss", np.mean(loss_avg_to_log["loss"]), epoch
        )
        writer.add_scalar(
            "loss/train_aff", np.mean(loss_avg_to_log["loss_affinities"]), epoch
        )
        writer.add_scalar(
            "loss/train_bel", mean_bel, epoch
        )
        if "loss_geo" in loss_avg_to_log and loss_avg_to_log["loss_geo"]:
            mean_geo = np.mean(loss_avg_to_log["loss_geo"])
            writer.add_scalar("loss/train_geo", mean_geo, epoch)
            if mean_bel > 1e-8:
                writer.add_scalar("loss/ratio_geo_bel", mean_geo / mean_bel, epoch)
        if "loss_struct" in loss_avg_to_log and loss_avg_to_log["loss_struct"]:
            mean_struct = np.mean(loss_avg_to_log["loss_struct"])
            writer.add_scalar("loss/train_struct", mean_struct, epoch)
            if mean_bel > 1e-8:
                writer.add_scalar("loss/ratio_struct_bel", mean_struct / mean_bel, epoch)
        if "loss_rel" in loss_avg_to_log and loss_avg_to_log["loss_rel"]:
            mean_rel = np.mean(loss_avg_to_log["loss_rel"])
            writer.add_scalar("loss/train_rel", mean_rel, epoch)
            if mean_bel > 1e-8:
                writer.add_scalar("loss/ratio_rel_bel", mean_rel / mean_bel, epoch)
        if "loss_vis" in loss_avg_to_log and loss_avg_to_log["loss_vis"]:
            mean_vis = np.mean(loss_avg_to_log["loss_vis"])
            writer.add_scalar("loss/train_vis", mean_vis, epoch)
            if mean_bel > 1e-8:
                writer.add_scalar("loss/ratio_vis_bel", mean_vis / mean_bel, epoch)

        # Belief peak health monitoring
        if "belief_peak" in loss_avg_to_log and loss_avg_to_log["belief_peak"]:
            writer.add_scalar("health/belief_peak_mean",
                              np.mean(loss_avg_to_log["belief_peak"]), epoch)


def main(opt):
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.gradcheck = False
    torch.backends.cudnn.benchmark = True

    local_rank = opt.local_rank

    # Validate Arguments
    if opt.use_s3 and (opt.train_buckets is None or opt.endpoint is None):
        raise ValueError(
            "--train_buckets and --endpoint must be specified if training with data from s3 bucket."
        )

    if not opt.use_s3 and opt.data is None:
        raise ValueError("--data field must be specified.")

    os.makedirs(opt.outf, exist_ok=True)

    random_seed = random.randint(1, 10000)
    if opt.manualseed is not None:
        random_seed = opt.manualseed

    # Save run parameters in a file
    with open(opt.outf + "/header.txt", "w") as file:
        file.write(str(opt) + "\n")
        file.write("seed: " + str(random_seed) + "\n")

    writer = None
    if local_rank == 0:
        writer = SummaryWriter(opt.outf + "/runs/")

    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.cuda.set_device(local_rank)
    # Windows 단일 GPU: distributed 우회
    if os.name == 'nt' or int(os.environ.get('WORLD_SIZE', '1')) <= 1:
        pass  # skip distributed init
    else:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")


    # Data Augmentation
    transform = transforms.Compose([
        transforms.Resize(opt.imagesize),
        transforms.ToTensor()
    ])

    # Load Model
    net = DopeNetwork()
    output_size = 50
    # sigma is controlled via --sigma CLI argument (default: 4.0)

    # Convert object names to lower-case for comparison later
    for idx in range(len(opt.object)):
        opt.object[idx] = opt.object[idx].lower()

    training_dataset = CleanVisiiDopeLoader(
        opt.data,
        sigma=opt.sigma,
        output_size=output_size,
        objects=opt.object,
        use_s3=opt.use_s3,
        buckets=opt.train_buckets,
        endpoint_url=opt.endpoint,
        truncation_aug_prob=opt.truncation_aug_prob,
    )
    training_data = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=opt.batchsize,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=True,
    )

    if not training_data is None:
        print("training data: {} batches".format(len(training_data)))

        print("Loading Model...")
        if os.name == 'nt' or int(os.environ.get('WORLD_SIZE', '1')) <= 1:
            net = net.cuda()
        else:
            net = torch.nn.parallel.DistributedDataParallel(
                net.cuda(),
                device_ids=[local_rank],
                output_device=local_rank
            )

    # Load any previous checkpoint (i.e. current job is a follow-up job)
    if opt.net_path is not None:
        net.load_state_dict(torch.load(opt.net_path))

    parameters = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.Adam(parameters, lr=opt.lr)

    print("ready to train!")
    start_time = datetime.datetime.now()
    print("start:", start_time.strftime("%m/%d/%Y, %H:%M:%S"))

    ckpt_q = None
    if opt.nb_checkpoints > 0:
        ckpt_q = Queue(maxsize=opt.nb_checkpoints)

    start_epoch = 0
    if opt.net_path is not None:
        # We started with a saved checkpoint, we start numbering checkpoints
        # after the loaded one
        try:
            start_epoch = int(os.path.splitext(os.path.basename(opt.net_path).split('_')[-1])[0]) + 1
        except:
            start_epoch = 1
        print(f"Starting at epoch {start_epoch}")

    # Geometric loss setup
    geo_loss_module = None
    if opt.geo_loss:
        from geo_loss import GeometricLoss
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'self_training'))
        from pnp_solver import make_pallet_keypoints_3d, make_camera_matrix
        kp3d = make_pallet_keypoints_3d(1.1, 1.1, 0.15)
        K = make_camera_matrix(opt.geo_fx, opt.geo_fy, opt.geo_cx, opt.geo_cy)
        geo_loss_module = GeometricLoss(
            kp3d, K,
            belief_size=output_size,
            input_size=opt.imagesize,
            orig_size=(opt.geo_img_w, opt.geo_img_h),
            temperature=opt.geo_temperature,
        ).cuda()
        print(f"[GEO] Geometric loss enabled (lambda={opt.geo_lambda}, warmup={opt.geo_warmup})")

    # Visibility-aware coordinate loss setup
    vis_loss_module = None
    if opt.vis_coord_loss:
        from geo_loss import VisibilityCoordLoss
        vis_loss_module = VisibilityCoordLoss(
            temperature=opt.geo_temperature,
            delta=0.03,
        ).cuda()
        print(f"[VIS] Visibility coord loss enabled (lambda={opt.vis_lambda}, warmup={opt.vis_warmup})")

    # Reliability loss setup
    rel_loss_module = None
    if opt.rel_loss:
        from geo_loss import ReliabilityLoss
        rel_loss_module = ReliabilityLoss(
            temperature=opt.geo_temperature,
            delta=opt.rel_delta,
            lambda_log=opt.rel_lambda_log,
        ).cuda()
        print(f"[REL] Reliability loss enabled (lambda={opt.rel_lambda}, "
              f"warmup={opt.rel_warmup}, delta={opt.rel_delta}, "
              f"lambda_log={opt.rel_lambda_log})")

    # Structural loss setup
    struct_loss_module = None
    if opt.struct_loss:
        from geo_loss import StructuralLoss, SpatialSoftArgmax2D
        soft_argmax = SpatialSoftArgmax2D(temperature=opt.geo_temperature)
        struct_lambdas = {
            'flip': opt.struct_flip,
            'edge': opt.struct_edge,
            'coord': opt.struct_coord,
            'vp': opt.struct_vp,
        }
        struct_loss_module = StructuralLoss(
            soft_argmax, lambdas=struct_lambdas, delta=opt.struct_delta
        ).cuda()
        print(f"[STRUCT] Structural loss enabled (lambda={opt.struct_lambda}, "
              f"warmup={opt.struct_warmup}, flip={opt.struct_flip}, "
              f"edge={opt.struct_edge}, coord={opt.struct_coord}, "
              f"vp={opt.struct_vp})")

    net.train()
    for epoch in range(start_epoch, opt.epochs + 1):
        _runnetwork(net, optimizer, local_rank, epoch, training_data, writer,
                    geo_loss_module=geo_loss_module,
                    geo_lambda=opt.geo_lambda,
                    geo_warmup=opt.geo_warmup,
                    struct_loss_module=struct_loss_module,
                    struct_lambda=opt.struct_lambda,
                    struct_warmup=opt.struct_warmup,
                    rel_loss_module=rel_loss_module,
                    rel_lambda=opt.rel_lambda,
                    rel_warmup=opt.rel_warmup,
                    vis_loss_module=vis_loss_module,
                    vis_lambda=opt.vis_lambda,
                    vis_warmup=opt.vis_warmup)

        try:
            if local_rank == 0 and epoch > 0 and epoch % opt.save_every == 0:
                out_fn = f"{opt.outf}/net_{opt.namefile}_{str(epoch).zfill(4)}.pth"
                torch.save(net.state_dict(), out_fn)

                # Clean up old checkpoints if we're limiting the number saved
                if ckpt_q is not None:
                    if ckpt_q.full():
                        to_del = ckpt_q.get()
                        os.remove(to_del)
                    ckpt_q.put(out_fn)

        except Exception as e:
            print(f"Encountered Exception: {e}")

    if local_rank == 0:
        torch.save(
            net.state_dict(),
            f"{opt.outf}/final_net_{opt.namefile}_{str(epoch).zfill(4)}.pth"
        )

    print("end:", datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
    print("Total time taken: ", str(datetime.datetime.now() - start_time).split(".")[0])
    return


if __name__ == "__main__":
    conf_parser = argparse.ArgumentParser(
        description=__doc__,  # printed with -h/--help
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Turn off help, so we print all options in response to -h
        add_help=False,
    )
    conf_parser.add_argument(
        "-c", "--config",
        help="Specify config file",
        metavar="FILE"
    )
    # Read the config but do not overwrite the args written
    args, remaining_argv = conf_parser.parse_known_args()


    parser = argparse.ArgumentParser()
    # Specify Training Data
    parser.add_argument(
        "--data",
        nargs="+",
        help="Path to training data"
    )
    parser.add_argument(
        "--use_s3",
        action="store_true",
        help="Use s3 buckets for training data"
    )
    parser.add_argument(
        "--train_buckets",
        nargs="+",
        default=[],
        help="s3 buckets containing training data. Can list multiple buckets separated by a space.",
    )
    parser.add_argument(
        "--endpoint",
        "--endpoint_url",
        type=str,
        default=None
    )

    # Specify Training Object
    parser.add_argument(
        "--object",
        nargs="+",
        required=True,
        default=[],
        help='Object to train network for. Must match "class" field in groundtruth .json file.'
        ' For best performance, only put one object of interest.',
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="number of data loading workers"
    )
    parser.add_argument(
        "--batchsize", "--batch_size",
        type=int,
        default=32,
        help="input batch size"
    )
    parser.add_argument(
        "--imagesize",
        type=int,
        default=448,
        help="the height / width of the input image to network",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate, default=0.0001"
    )
    parser.add_argument(
        "--net_path",
        default=None, help="path to net (to continue training)"
    )
    parser.add_argument(
        "--namefile",
        default="epoch",
        help="name to put on the file of the save weights"
    )
    parser.add_argument(
        "--manualseed",
        type=int,
        help="manual random number seed"
    )
    parser.add_argument(
        "--epochs",
        "--epoch",
        "-e",
        type=int,
        default=60,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--loginterval",
        type=int,
        default=100
    )
    parser.add_argument(
        "--outf",
        default="output/weights",
        help="folder to output images and model checkpoints",
    )
    parser.add_argument(
        "--nb_checkpoints",
        type=int,
        default=0,
        help="Number of checkpoints (.pth files) to save. Older ones will be "
        "deleted as new ones are saved. A value of 0 means an unlimited "
        "number will be saved"
    )
    parser.add_argument(
        '--save_every',
        type=int, default=1,
        help='How often (in epochs) to save a snapshot'
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=4.0,
        help="keypoint creation sigma (Gaussian std for belief map)")
    parser.add_argument(
        "--local-rank",
        type=int,
        default=0
    )

    parser.add_argument("--save", action="store_true", help="save a batch and quit")

    # On-the-fly truncation augmentation
    parser.add_argument("--truncation_aug_prob", type=float, default=0.0,
                        help="Probability per sample of applying on-the-fly "
                             "truncation crop+pad augmentation (0.0=off, "
                             "challenge pretrain uses 0.6)")

    # Symmetric loss (180° front-back swap)
    parser.add_argument("--symmetric_loss", action="store_true",
                        help="Use min(orig, 180°-swapped) belief loss for symmetric objects")

    # Geometric loss arguments
    parser.add_argument("--geo_loss", action="store_true",
                        help="Enable geometric loss (soft-argmax + BPnP)")
    parser.add_argument("--geo_lambda", type=float, default=0.1,
                        help="Weight for geometric loss (default: 0.1)")
    parser.add_argument("--geo_warmup", type=int, default=5,
                        help="Epochs before enabling PnP-based losses (default: 5)")
    parser.add_argument("--geo_temperature", type=float, default=1.0,
                        help="Soft-argmax temperature (default: 1.0)")
    parser.add_argument("--geo_fx", type=float, default=614.18)
    parser.add_argument("--geo_fy", type=float, default=614.31)
    parser.add_argument("--geo_cx", type=float, default=329.28)
    parser.add_argument("--geo_cy", type=float, default=234.53)
    parser.add_argument("--geo_img_w", type=int, default=640)
    parser.add_argument("--geo_img_h", type=int, default=480)

    # Visibility-aware coordinate loss arguments
    parser.add_argument("--vis_coord_loss", action="store_true",
                        help="Enable visibility-aware coordinate loss")
    parser.add_argument("--vis_lambda", type=float, default=0.005,
                        help="Weight for visibility coord loss (default: 0.005)")
    parser.add_argument("--vis_warmup", type=int, default=0,
                        help="Epochs before enabling vis coord loss (default: 0)")

    # Reliability loss arguments
    parser.add_argument("--rel_loss", action="store_true",
                        help="Enable reliability-aware coordinate loss")
    parser.add_argument("--rel_lambda", type=float, default=0.005,
                        help="Weight for reliability loss (default: 0.005)")
    parser.add_argument("--rel_warmup", type=int, default=0,
                        help="Epochs before enabling reliability loss (default: 0)")
    parser.add_argument("--rel_delta", type=float, default=0.03,
                        help="Huber delta for reliability loss (default: 0.03)")
    parser.add_argument("--rel_lambda_log", type=float, default=0.5,
                        help="Log-regularizer weight (default: 0.5)")

    # Structural loss arguments
    parser.add_argument("--struct_loss", action="store_true",
                        help="Enable structural losses (flip + edge + coord)")
    parser.add_argument("--struct_lambda", type=float, default=1.0,
                        help="Overall weight for structural loss (default: 1.0)")
    parser.add_argument("--struct_warmup", type=int, default=10,
                        help="Epochs before enabling structural losses (default: 10)")
    parser.add_argument("--struct_flip", type=float, default=0.02,
                        help="Flip equivariance loss weight (default: 0.02)")
    parser.add_argument("--struct_edge", type=float, default=0.05,
                        help="Sparse edge loss weight (default: 0.05)")
    parser.add_argument("--struct_coord", type=float, default=0.10,
                        help="Coordinate Huber loss weight (default: 0.10)")
    parser.add_argument("--struct_vp", type=float, default=0.0,
                        help="Vanishing-point concurrency loss weight (default: 0.0)")
    parser.add_argument("--struct_delta", type=float, default=0.03,
                        help="Huber delta for structural losses (default: 0.03)")

    opt = parser.parse_args(remaining_argv)

    main(opt)
