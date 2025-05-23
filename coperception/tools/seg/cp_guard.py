import argparse
from itertools import count
import os
from copy import deepcopy

import cv2
import torch.optim as optim
from tqdm import tqdm


from coperception.datasets import V2XSimSeg
from coperception.configs import Config, ConfigGlobal
from coperception.utils.SegMetrics import ComputeIoU
from coperception.utils.SegModule import *
from coperception.utils.loss import *
from coperception.models.seg import *
from torch.utils.data import DataLoader
from coperception.utils.data_util import apply_pose_noise
import random
from torch.autograd import Variable
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path

def print_and_write_log(log_str):
    """Print and write to log"""
    print(log_str)
    if args.log:
        saver.write(log_str + "\n")
        saver.flush()

def cal_robosac_consensus(num_agent, step_budget, num_attackers):
    """Calculate optimal consensus set size based on given step budget"""
    num_agent = num_agent - 1  # exclude ego agent
    eta = num_attackers / num_agent
    s = np.floor(np.log(1-np.power(1-0.99, 1/step_budget)) / np.log(1-eta)).astype(int)
    return s

def visualize(config, filename0, save_fig_path, segmodule, data, num_agent_list, padded_voxel_points, gt_max_iou, vis_tag):
    """Visualize segmentation results"""
    import matplotlib.pyplot as plt
    import cv2
    
    pred, labels = segmodule.step(data, num_agent_list[0][0], 1, loss=False)
    pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
    
    plt.clf()
    pred_map = np.zeros((256, 256, 3))
    gt_map = np.zeros((256, 256, 3))

    for k, v in config.class_to_rgb.items():
        pred_map[np.where(pred.cpu().numpy()[0] == k)] = v
        gt_map[np.where(labels.cpu().numpy()[0] == k)] = v

    filename = str(filename0[0][0])
    cut = filename[filename.rfind('agent') + 7:]
    seq_name = cut[:cut.rfind('_')]
    idx = cut[cut.rfind('_') + 1:cut.rfind('/')]
    seq_save = os.path.join(save_fig_path[0], seq_name)
    os.makedirs(seq_save, exist_ok=True)
    
    plt.imsave(
        f"{seq_save}/{idx}_voxel_points_{vis_tag}.png",
        np.asarray(
            np.max(padded_voxel_points.cpu().numpy()[0], axis=2), dtype=np.uint8
        ),
    )
    cv2.imwrite(f"{seq_save}/{idx}_pred_{vis_tag}.png", pred_map[:, :, ::-1])
    cv2.imwrite(f"{seq_save}/{idx}_gt_{vis_tag}.png", gt_map[:, :, ::-1])

#@torch.no_grad()
# We cannot use torch.no_grad() since we need to calculate the gradient for perturbation
def main(config, args):
    # Add global variables
    global saver  # For logging
    prev_result = None  # For history frame recording
    
    # Statistics variables
    succ = 0  # Number of successful consensus
    fail = 0  # Number of failed consensus
    steps = np.zeros(1000)  # Record sampling steps per frame
    ego_steps = np.zeros(1000)  # Record ego prediction steps per frame
    frame_seq = 0  # Current frame sequence number

    config.nepoch = args.nepoch
    num_epochs = args.nepoch
    need_log = args.log
    batch_size = args.batch
    num_workers = args.nworker
    compress_level = args.compress_level
    start_epoch = 1
    logpath = args.logpath
    pose_noise = args.pose_noise
    only_v2i = args.only_v2i

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    if args.com == "upperbound" or args.com == "lowerbound":
        flag = args.com
        config.com = None
    elif args.com == "when2com":
        flag = "when2com"
        if args.inference == "argmax_test":
            flag = "who2com"
        if args.warp_flag:
            flag = flag + "_warp"
    elif args.com in {"v2v", "disco", "sum", "mean", "max", "cat", "agent"}:
        flag = args.com
    else:
        raise ValueError(f"com: {args.com} is not supported")
    
    config.flag = flag

    num_agent = args.num_agent
    agent_idx_range = range(num_agent) if args.rsu else range(1, num_agent)
    valset = V2XSimSeg(
        dataset_roots=[args.data + "/agent%d" % i for i in agent_idx_range],
        config=config,
        split="val",
        val=True,
        com=args.com,
        bound="upperbound" if args.com == "upperbound" else "lowerbound",
        rsu=args.rsu,
    )
    valloader = DataLoader(
        valset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    print("Validation dataset size:", len(valset))

    checkpoint = torch.load(args.resume)

    config.flag = flag
    config.com = args.com
    # config.inference = args.inference
    config.split = "test"
    # build model
    if not args.rsu:
        num_agent -= 1
    if args.com.startswith("when2com") or args.com.startswith("who2com"):
        model = When2Com_UNet(
            config,
            in_channels=config.in_channels,
            n_classes=config.num_class,
            warp_flag=args.warp_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "v2v":
        model = V2VNet(
            config.in_channels,
            config.num_class,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "mean":
        model = MeanFusion(
            config.in_channels,
            config.num_class,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "max":
        model = MaxFusion(
            config.in_channels,
            config.num_class,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "sum":
        model = SumFusion(
            config.in_channels,
            config.num_class,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "cat":
        model = CatFusion(
            config.in_channels,
            config.num_class,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "agent":
        model = AgentWiseWeightedFusion(
            config.in_channels,
            config.num_class,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "disco":
        model = DiscoNet(
            config.in_channels,
            config.num_class,
            num_agent=num_agent,
            kd_flag=False,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    else:
        model = UNet(
            config.in_channels,
            config.num_class,
            num_agent=num_agent,
            compress_level=compress_level,
        )
    # model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    segmodule = SegModule(model, model, config, optimizer, False)
    segmodule.model.load_state_dict(checkpoint["model_state_dict"])


    # ==== eval ====
    segmodule.model.eval()
    compute_iou = ComputeIoU(num_class=config.num_class)  # num_class
    os.makedirs(logpath, exist_ok=True)
    logpath = os.path.join(logpath, f"{flag}_eval")
    os.makedirs(logpath, exist_ok=True)
    logpath = os.path.join(logpath, "with_rsu" if args.rsu else "no_rsu")
    os.makedirs(logpath, exist_ok=True)
    print("log path:", logpath)

    for k, v in segmodule.model.named_parameters():
        v.requires_grad = False  # fix parameters
    
    assert args.cpguard in ["upperbound", "lowerbound", "no_defense", "cpguard_validation", "cpguard_mAP", "fix_attackers", "performance_eval", "pasac", "robosac"]

   
    for idx, sample in enumerate(tqdm(valloader)):

        if args.com:
            (
                padded_voxel_points_list,
                padded_voxel_points_teacher_list,
                label_one_hot_list,
                trans_matrices,
                target_agent,
                num_sensor,
            ) = list(zip(*sample))
        else:
            (
                padded_voxel_points_list,
                padded_voxel_points_teacher_list,
                label_one_hot_list,
            ) = list(zip(*sample))

        

        if flag == "upperbound":
            padded_voxel_points = torch.cat(tuple(padded_voxel_points_teacher_list), 0)
        else:
            padded_voxel_points = torch.cat(tuple(padded_voxel_points_list), 0)

        label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
        # print('voxel', padded_voxel_points.size())  # batch*agent seq h w z
        # print('label', label_one_hot.size())

        data = {
            'ego_agent': args.ego_agent,
            'pert': None,
            'no_fuse': False,
            'attacker_list' : None,
            'collab_agent_list': None,
            'eps': args.eps,
            }
        data["bev_seq"] = padded_voxel_points.to(device).float()
        data["labels"] = label_one_hot.to(device)
        if args.com:
            trans_matrices = torch.stack(trans_matrices, 1)

            # add pose noise
            if pose_noise > 0:
                apply_pose_noise(pose_noise, trans_matrices)

            target_agent = torch.stack(target_agent, 1)
            num_sensor = torch.stack(num_sensor, 1)

            if not args.rsu:
                num_sensor -= 1

            data["trans_matrices"] = trans_matrices.to(device)
            data["target_agent"] = target_agent.to(device)
            data["num_sensor"] = num_sensor.to(device)
        

        # STEP 1:
        # get original ego agent class prediction of all pixels, without adv pert and fuse, return cls pred of all agents
        # cls_result, _ = segmodule.step(data, num_agent, batch_size, loss=False)
        # # change logits to one-hot
        # mean = torch.mean(cls_result, dim=2)
        # cls_result[:,:,0] = cls_result[:,:,0] > mean
        # cls_result[:,:,1] = cls_result[:,:,1] > mean
        # pseudo_gt = cls_result.clone().detach()

        if args.cpguard == 'upperbound':
            # no attacker is attacking and all agents are in collaboration, everything is just fine
            data['pert'] = None
            data['collab_agent_list'] = None
            data['no_fuse'] = False
            pred, labels = segmodule.step(data, num_agent, batch_size, loss=False)
            pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
            compute_iou(pred, labels)

            continue
        
        elif args.cpguard == 'lowerbound':
            # Suppose all neighboring agents are malicious, and only the ego agent is trusted
            # Each agent only use its own features to perform object detection
            data['pert'] = None
            data['collab_agent_list'] = None
            data['no_fuse'] = True
            pred, labels = segmodule.step(data, num_agent, batch_size, loss=False)
            pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
            compute_iou(pred, labels)

            continue

        else:
            # There are attackers among us: 

            # STEP 2:
            # generate adv perturb
            if args.adv_method == 'pgd':
                # PGD random init
                pert = torch.randn(6, 512, 32, 32) * 0.1
            elif args.adv_method == 'fgsm':
                # FGSM random init
                pert = torch.randn(6, 512, 32, 32) * 10
                args.adv_iter = 0
            elif args.adv_method == 'bim' or args.adv_method == 'cw-l2':
                # BIM/CW-L2 zero init
                pert = torch.zeros(6, 512, 32, 32)
            else:
                raise NotImplementedError
        
            ego_idx = args.ego_agent
            # Not including ego agent, since ego agent is always used.
            # Fix agent 2 as the attacker
            data['attacker_list'] = [2]
            data['no_fuse'] = False
            for i in range(args.adv_iter):
                pert.requires_grad = True
                # Introduce adv perturbation
                data['pert'] = pert.to(device)
                        
                # STEP 3: Use inverted classification ground truth, minimze loss wrt inverted gt, to generate adv attacks based on cls
                # NOTE: Actual ground truth is not always available especially in real-world attacks
                # We define the adversarial loss of the perturbed output with respect to an unperturbed output pseudo_gt instead of the ground truth
                _, cls_loss = segmodule.step(data, num_agent=num_agent, batch_size=1, invert_gt=True, adv_method=args.adv_method)

                pert = pert + args.pert_alpha * pert.grad.sign() * -1
                pert.detach_()

            # Detach and clone perturbations from Pytorch computation graph, in case of gradient misuse.
            pert = pert.detach().clone()
            # Apply the final perturbation to attackers' feature maps.
            data['pert'] = pert.to(device)

            if args.cpguard == 'no_defense':
                # attacker is always attacking and no defense is applied
                data['pert'] = pert.to(device)
                data['no_fuse'] = False
                data['collab_agent_list'] = None
                pred, labels = segmodule.step(data, num_agent, batch_size, loss=False, adv_method=args.adv_method)
                pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
                compute_iou(pred, labels)

                if args.visualization and idx % 20 == 0:  # render segmatic map
                    plt.clf()
                    pred_map = np.zeros((256, 256, 3))
                    gt_map = np.zeros((256, 256, 3))

                    for k, v in config.class_to_rgb.items():
                        pred_map[np.where(pred.cpu().numpy()[0] == k)] = v
                        gt_map[np.where(label_one_hot.numpy()[0] == k)] = v

                    plt.imsave(
                        f"{logpath}/{idx}_voxel_points.png",
                        np.asarray(
                            np.max(padded_voxel_points.cpu().numpy()[0], axis=2), dtype=np.uint8
                        ),
                    )
                    cv2.imwrite(f"{logpath}/{idx}_pred.png", pred_map[:, :, ::-1])
                    cv2.imwrite(f"{logpath}/{idx}_gt.png", gt_map[:, :, ::-1])

                continue

            elif args.cpguard == 'robosac':
                # First generate adversarial perturbation
                if args.adv_method == 'pgd':
                    # PGD random init
                    pert = torch.randn(6, 512, 32, 32) * 0.1
                elif args.adv_method == 'fgsm':
                    # FGSM random init
                    pert = torch.randn(6, 512, 32, 32) * 10
                    args.adv_iter = 0
                elif args.adv_method == 'bim' or args.adv_method == 'cw-l2':
                    # BIM/CW-L2 zero init
                    pert = torch.zeros(6, 512, 32, 32)
                else:
                    raise NotImplementedError

                # Given Step Budget N and Sampling Set Size s, perform predictions
                num_sensor = num_sensor[0, 0]
                ego_idx = args.ego_agent
                all_agent_list = [i for i in range(num_sensor)]
                # We always trust ourself
                all_agent_list.remove(ego_idx)
                
                print_and_write_log(f"\nFrame {frame_seq + 1}:")
                print_and_write_log(f"Available agents (excluding ego {ego_idx}): {all_agent_list}")
                
                # Check if there are enough agents for sampling
                available_agents = len(all_agent_list)
                if args.robosac_k is not None:
                    consensus_set_size = min(args.robosac_k, available_agents)
                else:
                    consensus_set_size = cal_robosac_consensus(num_agent, args.step_budget, args.number_of_attackers)
                    consensus_set_size = min(consensus_set_size, available_agents)
                
                print_and_write_log(f"Step Budget: {args.step_budget}, Consensus Set Size: {consensus_set_size}")
                if consensus_set_size < 1:
                    print_and_write_log('Expected Consensus Agent below 1. Exit.')
                    sys.exit()

                # Get reference prediction (ego only)
                print_and_write_log("Calculating ego-only reference prediction...")
                if args.use_history_frame and frame_seq > 0:
                    # Use history frame result
                    result_reference = prev_result
                    print_and_write_log("Using history frame as reference")
                else:
                    # Calculate ego only result
                    data['pert'] = None
                    data['collab_agent_list'] = None
                    data['no_fuse'] = True
                    data['attacker_list'] = []
                    pred_reference, labels = segmodule.step(data, num_agent, batch_size, loss=False)
                    # Save original logits for later computation
                    pred_reference = F.softmax(pred_reference, dim=1)
                    pred_reference_cls = torch.argmax(pred_reference, dim=1)
                    print_and_write_log("Using current ego prediction as reference")

                found = False
                final_pred = None
                final_labels = None
                
                for step in range(1, args.step_budget + 1):
                    print_and_write_log(f"\nStep {step}/{args.step_budget}:")
                    
                    # Sample collaborators
                    collab_agent_list = random.sample(all_agent_list, k=consensus_set_size)
                    print_and_write_log(f"Sampled collaborators: {collab_agent_list}")
                        
                    data['collab_agent_list'] = collab_agent_list
                    data['no_fuse'] = False
                    data['pert'] = pert.to(device)
                    # Set attacker_list as all non-collaborators
                    data['attacker_list'] = [i for i in all_agent_list if i not in collab_agent_list]
                    print_and_write_log(f"Potential attackers: {data['attacker_list']}")

                    # Get prediction with current collaborators
                    pred, labels = segmodule.step(data, num_agent, batch_size, loss=False)
                    # Save original logits for later computation
                    pred = F.softmax(pred, dim=1)
                    pred_cls = torch.argmax(pred, dim=1)

                    # Calculate IoU between reference and current prediction
                    intersection = torch.logical_and(pred_reference_cls, pred_cls)
                    union = torch.logical_or(pred_reference_cls, pred_cls)
                    iou = torch.sum(intersection) / torch.sum(union)

                    print_and_write_log(f"IoU with reference: {iou:.4f} (threshold: {args.th:.4f})")
                    
                    if iou < args.th:
                        print_and_write_log(f'Attacker(s) is(are) among {collab_agent_list}')
                    else:
                        sus_agent_list = [i for i in all_agent_list if i not in collab_agent_list]
                        print_and_write_log(f'Achieved consensus! Collaborating agents: {collab_agent_list}')
                        print_and_write_log(f'Potential attacker(s) among: {sus_agent_list}')
                        found = True
                        final_pred = pred_cls
                        final_labels = labels
                        break

                if not found:
                    print_and_write_log('No consensus achieved. Falling back to ego-only prediction.')
                    # Fall back to ego only result
                    data['pert'] = None
                    data['collab_agent_list'] = None
                    data['no_fuse'] = True
                    data['attacker_list'] = []
                    pred, labels = segmodule.step(data, num_agent, batch_size, loss=False)
                    pred = F.softmax(pred, dim=1)
                    final_pred = torch.argmax(pred, dim=1)
                    final_labels = labels
                
                # Save current result for next frame if using history
                if args.use_history_frame:
                    prev_result = final_pred

                # Ensure labels is torch.Tensor type
                if isinstance(final_labels, np.ndarray):
                    final_labels = torch.from_numpy(final_labels)
                final_labels = final_labels.to(device)

                # Compute IoU metrics
                compute_iou(final_pred, final_labels)

                # Record steps and performance statistics
                if found:
                    steps[frame_seq] = step
                    succ += 1
                else:
                    steps[frame_seq] = args.step_budget
                    ego_steps[frame_seq] = 1
                    fail += 1
                
                frame_seq += 1

                continue

            elif args.cpguard == 'pasac':
                # First generate adversarial perturbation, same as robosac
                if args.adv_method == 'pgd':
                    pert = torch.randn(6, 512, 32, 32) * 0.1
                elif args.adv_method == 'fgsm':
                    pert = torch.randn(6, 512, 32, 32) * 10
                    args.adv_iter = 0
                elif args.adv_method == 'bim' or args.adv_method == 'cw-l2':
                    pert = torch.zeros(6, 512, 32, 32)
                else:
                    raise NotImplementedError

                # Get all agent list
                num_sensor = num_sensor[0, 0]
                ego_idx = args.ego_agent
                all_agent_list = [i for i in range(num_sensor)]
                all_agent_list.remove(ego_idx)  # Remove self

                # First calculate ego BEV seg map
                data['pert'] = None
                data['collab_agent_list'] = None
                data['no_fuse'] = True
                data['attacker_list'] = []
                ego_pred, _ = segmodule.step(data, num_agent, batch_size, loss=False)
                ego_pred = torch.argmax(F.softmax(ego_pred, dim=1), dim=1)
                ego_pred_pr = np.zeros((256, 256))

                # 1: vehicle
                loc_k = np.where(ego_pred.cpu().numpy()[0] == 1)
                ego_pred_pr[loc_k] = len(loc_k[0])/(256*256)
                # 4: road
                loc_k = np.where(ego_pred.cpu().numpy()[0] == 4)
                ego_pred_pr[loc_k] = len(loc_k[0])/(256*256)

                def calculate_ccloss(agent_list):
                    """Calculate CCLoss for given agent list"""
                    if not agent_list:
                        return 0.0
                    
                    data['pert'] = pert.to(device)
                    data['no_fuse'] = False
                    data['collab_agent_list'] = [args.ego_agent] + agent_list
                    data['attacker_list'] = [i for i in all_agent_list if i not in agent_list]
                    
                    fused_pred, _ = segmodule.step(data, num_agent, batch_size, loss=False)
                    fused_pred = torch.argmax(F.softmax(fused_pred, dim=1), dim=1)
                    fused_pred_pr = np.zeros((256, 256))
                    
                    # 1: vehicle
                    loc_k = np.where(fused_pred.cpu().numpy()[0] == 1)
                    fused_pred_pr[loc_k] = len(loc_k[0])/(256*256)
                    # 4: road
                    loc_k = np.where(fused_pred.cpu().numpy()[0] == 4)
                    fused_pred_pr[loc_k] = len(loc_k[0])/(256*256)

                    # Calculate CCLoss
                    ccloss_a = np.sum(ego_pred_pr * fused_pred_pr)
                    ccloss_b = np.sum(ego_pred_pr + fused_pred_pr)
                    
                    return ccloss_a/ccloss_b if ccloss_b > 0 else 0.0

                def pasac_recursive(agent_list):
                    """Recursively execute PASAC algorithm"""
                    if len(agent_list) <= 1:
                        # If only one agent remains, directly calculate its CCLoss
                        ccloss = calculate_ccloss([agent_list[0]])
                        return [agent_list[0]] if ccloss > args.th else []
                    
                    # Split agent list into two halves
                    mid = len(agent_list) // 2
                    G1 = agent_list[:mid]
                    G2 = agent_list[mid:]
                    
                    # Calculate CCLoss for both groups
                    ccloss_G1 = calculate_ccloss(G1)
                    ccloss_G2 = calculate_ccloss(G2)
                    
                    benign_agents = []
                    
                    # Recursively process both subgroups
                    if ccloss_G1 > args.th:
                        benign_agents.extend(pasac_recursive(G1))
                    if ccloss_G2 > args.th:
                        benign_agents.extend(pasac_recursive(G2))
                    
                    return benign_agents

                # Execute PASAC algorithm
                benign_agents = pasac_recursive(all_agent_list)
                
                # Set final prediction based on results
                if not benign_agents:
                    print_and_write_log('No benign collaborators found, using ego only.')
                    data['pert'] = None
                    data['no_fuse'] = True
                    data['collab_agent_list'] = None
                    data['attacker_list'] = []
                else:
                    print_and_write_log(f'Found {len(benign_agents)} benign collaborators: {benign_agents}')
                    data['pert'] = pert.to(device)
                    data['no_fuse'] = False
                    data['collab_agent_list'] = [args.ego_agent] + benign_agents
                    data['attacker_list'] = [i for i in all_agent_list if i not in benign_agents]

                # Final prediction
                pred, labels = segmodule.step(data, num_agent, batch_size, loss=False)
                pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
                compute_iou(pred, labels)

                if args.visualization and idx % 20 == 0:  # render segmatic map
                    plt.clf()
                    pred_map = np.zeros((256, 256, 3))
                    gt_map = np.zeros((256, 256, 3))

                    for k, v in config.class_to_rgb.items():
                        pred_map[np.where(pred.cpu().numpy()[0] == k)] = v
                        gt_map[np.where(label_one_hot.numpy()[0] == k)] = v

                    plt.imsave(
                        f"{logpath}/{idx}_voxel_points.png",
                        np.asarray(
                            np.max(padded_voxel_points.cpu().numpy()[0], axis=2), dtype=np.uint8
                        ),
                    )
                    cv2.imwrite(f"{logpath}/{idx}_pred.png", pred_map[:, :, ::-1])
                    cv2.imwrite(f"{logpath}/{idx}_gt.png", gt_map[:, :, ::-1])

                continue



        # If has RSU, do not count RSU's output into evaluation
        if args.rsu:
            pred = pred[1:, :, :, :]
            labels = labels[1:, :, :]

        labels = labels.detach().cpu().numpy().astype(np.int32)

        # late fusion
        if args.apply_late_fusion:
            pred = torch.flip(pred, (2,))
            size = (1, *pred[0].shape)

            for ii in range(num_sensor[0, 0]):
                for jj in range(num_sensor[0, 0]):
                    if ii == jj:
                        continue

                    nb_agent = torch.unsqueeze(pred[jj], 0)
                    tfm_ji = trans_matrices[0, jj, ii]
                    M = (
                        torch.hstack((tfm_ji[:2, :2], -tfm_ji[:2, 3:4]))
                        .float()
                        .unsqueeze(0)
                    )  # [1,2,3]

                    mask = torch.tensor(
                        [[[1, 1, 4 / 128], [1, 1, 4 / 128]]], device=M.device
                    )

                    M *= mask
                    grid = F.affine_grid(M, size=torch.Size(size)).to(device)
                    warp_feat = F.grid_sample(nb_agent, grid).squeeze()
                    pred[ii] += warp_feat

            pred = torch.flip(pred, (2,))
        # ============

        pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
        compute_iou(pred, labels)


    print("iou:", compute_iou.get_ious())
    print("miou:", compute_iou.get_miou(ignore=0))
    log_file = open(f"{logpath}/log.txt", "w")
    log_file.write(f"iou: {compute_iou.get_ious()}\n")
    log_file.write(f"miou: {compute_iou.get_miou(ignore=0)}")

    # After main loop
    if args.cpguard == 'robosac':
        print_and_write_log("\nFinal Statistics:")
        print_and_write_log("Total frames processed: {}".format(frame_seq))
        print_and_write_log("Success rate: {:.2f}%".format(succ/frame_seq*100))
        print_and_write_log("Average steps: {:.2f}".format(np.mean(steps[:frame_seq])))
        print_and_write_log("Average ego steps: {:.2f}".format(np.mean(ego_steps[:frame_seq])))
        
        if args.log:
            saver.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-d",
        "--data",
        default="./datasets/V2X-Sim-seg/test",
        type=str,
        help="The path to the preprocessed sparse BEV training data",
    )
    parser.add_argument("--batch", default=1, type=int, help="The number of scene")
    parser.add_argument("--warp_flag", action="store_true", help="Whether to use pose info for When2com")
    parser.add_argument("--nepoch", default=10, type=int, help="Number of epochs")
    parser.add_argument("--nworker", default=2, type=int, help="Number of workers")
    parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument("--th", default=0.08, type=float, help="CCLoss threshold")
    parser.add_argument("--log", action="store_true", help="Whether to log")
    parser.add_argument("--logpath", default="./coperception/logs", help="The path to the output log file")
    parser.add_argument("--rsu", default=0, type=int, help="0: no RSU, 1: RSU")
    parser.add_argument(
        "--resume",
        default='./coperception/logs/v2v/no_rsu/epoch_100.pth',
        type=str,
        help="The path to the saved model that is loaded to resume training",
    )
    parser.add_argument(
        "--resume_teacher",
        default="",
        type=str,
        help="The path to the saved teacher model that is loaded to resume training",
    )
    parser.add_argument(
        "--kd_flag",
        default=0,
        type=int,
        help="Whether to enable distillation (only DiscNet is 1 )",
    )
    parser.add_argument("--kd_weight", default=100, type=int, help="KD loss weight")
    parser.add_argument(
        "--gnn_iter_times",
        default=3,
        type=int,
        help="Number of message passing for V2VNet",
    )
    parser.add_argument(
        "--visualization", default=False, action="store_true", help="Visualize validation result"
    )
    parser.add_argument(
        "--com", default="v2v", type=str, help="disco/when2com/v2v/sum/mean/max/cat/agent"
    )
    parser.add_argument(
        "--bound",
        type=str,
        default="both",
        help="The input setting: lowerbound -> single-view or upperbound -> multi-view",
    )
    parser.add_argument(
        "--no_cross_road", action="store_true", help="Do not load data of cross roads"
    )
    # scene_batch => batch size in each scene
    parser.add_argument(
        "--num_agent", default=6, type=int, help="The total number of agents"
    )

    parser.add_argument(
        "--compress_level",
        default=0,
        type=int,
        help="Compress the communication layer channels by 2**x times in encoder",
    )
    parser.add_argument(
        "--pose_noise",
        default=0,
        type=float,
        help="draw noise from normal distribution with given mean (in meters), apply to transformation matrix.",
    )
    parser.add_argument(
        "--apply_late_fusion",
        default=0,
        type=int,
        help="1: apply late fusion. 0: no late fusion",
    )
    parser.add_argument(
        "--only_v2i",
        default=0,
        type=int,
        help="1: only v2i, 0: v2v and v2i",
    )

    # Adversarial perturbation
    parser.add_argument('--pert_alpha', type=float, default=0.1, help='scale of the perturbation')
    parser.add_argument('--adv_method', type=str, default='pgd', help='pgd/bim/cw-l2/fgsm')
    parser.add_argument('--eps', type=float, default=0.5, help='epsilon of adv attack.')
    parser.add_argument('--adv_iter', type=int, default=15, help='adv iterations of computing perturbation')

    # Scene and frame settings
    parser.add_argument('--scene_id', type=list, default=[8], help='target evaluation scene') #Scene 8, 96, 97 has 6 agents.
    parser.add_argument('--sample_id', type=int, default=None, help='target evaluation sample')
    
    # CP-Guard modes and parameters
    parser.add_argument('--cpguard', type=str, default='no_defense', help='upperbound/lowerbound/no_defense/pasac/robosac')
    parser.add_argument('--ego_agent', type=int, default=1, help='id of ego agent')
    parser.add_argument('--ego_loss_only', action="store_true", help='only use ego loss to compute adv perturbation')
    parser.add_argument('--number_of_attackers', type=int, default=1, help='number of malicious attackers in the scene')
    parser.add_argument('--fix_attackers', action="store_true", help='if true, attackers will not change in different frames')
    parser.add_argument('--partial_upperbound', action="store_true", help='use with specifying ransan_k, to perform clean collaboration with a subset of teammates')
    
    # Robosac parameters
    parser.add_argument('--robosac_k', type=int, default=None, help='specify consensus set size if needed')
    parser.add_argument('--step_budget', type=int, default=5, help='sampling budget in a single frame')
    parser.add_argument('--use_history_frame', action="store_true", help='use history frame for computing the consensus')
    
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parser.parse_args()
    print(args)
    config = Config("train")
    main(config, args)


