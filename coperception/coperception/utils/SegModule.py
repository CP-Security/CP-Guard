import torch.nn.functional as F
import torch.nn as nn
import torch
from coperception.utils.detection_util import *


class SegModule(object):
    def __init__(self, model, teacher, config, optimizer, kd_flag):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.nepoch
        )
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200], gamma=0.5)
        self.criterion = nn.CrossEntropyLoss()
        self.teacher = teacher
        if kd_flag:
            for k, v in self.teacher.named_parameters():
                v.requires_grad = False  # fix parameters

        self.kd_flag = kd_flag

        self.com = config.com

    def resume(self, path):
        def map_func(storage, location):
            return storage.cuda()

        if os.path.isfile(path):
            if rank == 0:
                print("=> loading checkpoint '{}'".format(path))

            checkpoint = torch.load(path, map_location=map_func)
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)

            ckpt_keys = set(checkpoint["state_dict"].keys())
            own_keys = set(model.state_dict().keys())
            missing_keys = own_keys - ckpt_keys
            for k in missing_keys:
                print("caution: missing keys from checkpoint {}: {}".format(path, k))
        else:
            print("=> no checkpoint found at '{}'".format(path))
    
  

    def step(self, data, num_agent, batch_size, loss=True, invert_gt=False, adv_method=None):

        bev = data["bev_seq"]
        labels = data["labels"]
        
        self.optimizer.zero_grad()
        bev = bev.permute(0, 3, 1, 2).contiguous()

        
        if not self.com:
            filtered_bev = []
            filtered_label = []
            for i in range(bev.size(0)):
                if torch.sum(bev[i]) > 1e-4:
                    filtered_bev.append(bev[i])
                    filtered_label.append(labels[i])
            bev = torch.stack(filtered_bev, 0)
            labels = torch.stack(filtered_label, 0)

        if self.kd_flag:
            data["bev_seq_teacher"] = (
                data["bev_seq_teacher"].permute(0, 3, 1, 2).contiguous()
            )

        if self.com:
            # if self.kd_flag:
            #     pred, x9, x8, x7, x6, x5, fused_layer = self.model(
            #         bev, data["trans_matrices"], data["num_sensor"]
            #     )
            # elif self.config.flag.startswith("when2com") or self.config.flag.startswith(
            #     "who2com"
            # ):
            #     if self.config.split == "train":
            #         pred = self.model(
            #             bev, data["trans_matrices"], data["num_sensor"], training=True,
            #         )
            #     else:
            #         pred = self.model(
            #             bev,
            #             data["trans_matrices"],
            #             data["num_sensor"],
            #             inference=self.config.inference,
            #             training=False,
            #         )
            # else:
                pred = self.model(bev, data["trans_matrices"], data["num_sensor"], 
                                  pert=data['pert'], 
                                  eps=data['eps'], 
                                  no_fuse=data['no_fuse'], 
                                  attacker_list=data['attacker_list'], 
                                  collab_agent_list=data['collab_agent_list'],
                                  ego_agent=data['ego_agent'],
                                  adv_method=adv_method)
        else:
            pred = self.model(bev)

        if self.com:
            filtered_pred = []
            filtered_label = []
            for i in range(bev.size(0)):
                if torch.sum(bev[i]) > 1e-4:
                    filtered_pred.append(pred[i])
                    filtered_label.append(labels[i])
            pred = torch.stack(filtered_pred, 0)
            labels = torch.stack(filtered_label, 0)
        if not loss:
            return pred, labels

        

        kd_loss = (
            self.get_kd_loss(batch_size, data, fused_layer, num_agent, x5, x6, x7)
            if self.kd_flag
            else 0
        )

        if adv_method == 'pgd' or adv_method == 'bim':
            if invert_gt:
                # invert labels for adv loss
                # labels = labels.bool()
                # labels = ~labels
                # print(labels)
                # more dedicated label processing methods can be applied here, invert label for targeted pgd attack
                labels = 7 - labels
                labels = labels.float()
            loss = self.criterion(pred, labels.long()) + kd_loss
        
        elif adv_method == 'fgsm':
            if invert_gt:
                idx = torch.randperm(labels.nelement()) # random permutation, 随机排列
                labels = labels.view(-1)[idx].view(labels.size())
                labels = labels.float()
            loss = self.criterion(pred, labels.long()) + kd_loss

        elif adv_method == 'cw-l2':

            def convert_to_one_hot(labels, num_classes):
                # Expand the labels to one-hot encoding
                batch_size, height, width = labels.size()
                one_hot = torch.zeros((batch_size, height, width, num_classes), device=labels.device)
                one_hot = one_hot.scatter_(3, labels.unsqueeze(3), 1)
                return one_hot.float()

            def cw_loss(result, labels, kappa=0, c=0.1):
                # Convert labels to one-hot encoding
                one_hot_labels = convert_to_one_hot(labels, num_classes=result.size(1))

                # Flatten the result and one-hot labels for computation
                flattened_result = result.permute(0, 2, 3, 1).reshape(-1, result.size(1))
                flattened_labels = one_hot_labels.permute(0, 2, 3, 1).reshape(-1, result.size(1))

                # Define f-function
                def f(x):
                    # Calculate the CW loss component wise
                    real_class = torch.max(flattened_labels * x, dim=1)  # Confidence of the true class
                    other_class = torch.max((1 - flattened_labels) * x, dim=1)  # Max confidence of non-true classes
                    return torch.clamp(real_class.values - other_class.values, min=-kappa)

                # Apply tanh activation to ensure the output range [-1, 1]
                a = 1/2 * (torch.tanh(flattened_result) + 1)

                # Calculate the CW loss
                loss2 = torch.sum(c * f(a))
                return loss2

            loss = cw_loss(pred,labels.long())

        else:
            loss = self.criterion(pred, labels.long()) + kd_loss

        if isinstance(self.criterion, nn.DataParallel):
            loss = loss.mean()

        loss_data = loss.data.item()
        if np.isnan(loss_data):
            raise ValueError("loss is nan while training")
        
        loss.requires_grad_(True)
        loss.backward()
        self.optimizer.step()

        return pred, loss_data

   
    def get_kd_loss(self, batch_size, data, fused_layer, num_agent, x5, x6, x7):
        if not self.kd_flag:
            return 0

        bev_seq_teacher = data["bev_seq_teacher"].type(torch.cuda.FloatTensor)
        kd_weight = data["kd_weight"]
        (
            logit_teacher,
            x9_teacher,
            x8_teacher,
            x7_teacher,
            x6_teacher,
            x5_teacher,
            x4_teacher,
        ) = self.teacher(bev_seq_teacher)
        kl_loss_mean = nn.KLDivLoss(size_average=True, reduce=True)

        target_x5 = x5_teacher.permute(0, 2, 3, 1).reshape(
            num_agent * batch_size * 16 * 16, -1
        )
        student_x5 = x5.permute(0, 2, 3, 1).reshape(
            num_agent * batch_size * 16 * 16, -1
        )
        kd_loss_x5 = kl_loss_mean(
            F.log_softmax(student_x5, dim=1), F.softmax(target_x5, dim=1)
        )

        target_x6 = x6_teacher.permute(0, 2, 3, 1).reshape(
            num_agent * batch_size * 32 * 32, -1
        )
        student_x6 = x6.permute(0, 2, 3, 1).reshape(
            num_agent * batch_size * 32 * 32, -1
        )
        kd_loss_x6 = kl_loss_mean(
            F.log_softmax(student_x6, dim=1), F.softmax(target_x6, dim=1)
        )

        target_x7 = x7_teacher.permute(0, 2, 3, 1).reshape(
            num_agent * batch_size * 64 * 64, -1
        )
        student_x7 = x7.permute(0, 2, 3, 1).reshape(
            num_agent * batch_size * 64 * 64, -1
        )
        kd_loss_x7 = kl_loss_mean(
            F.log_softmax(student_x7, dim=1), F.softmax(target_x7, dim=1)
        )

        target_x4 = x4_teacher.permute(0, 2, 3, 1).reshape(
            num_agent * batch_size * 32 * 32, -1
        )
        student_x4 = fused_layer.permute(0, 2, 3, 1).reshape(
            num_agent * batch_size * 32 * 32, -1
        )
        kd_loss_fused_layer = kl_loss_mean(
            F.log_softmax(student_x4, dim=1), F.softmax(target_x4, dim=1)
        )

        return kd_weight * (kd_loss_x5 + kd_loss_x6 + kd_loss_x7 + kd_loss_fused_layer)
