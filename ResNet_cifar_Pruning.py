import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from ResNet_cifar_Model import *

from ResNet_cifar_Train import all_test
from utils.Channel_selection import channel_selection
from utils.Data_loader import Data_Loader_CIFAR
from utils.Functions import *

# ---------------------------------------
activations = []


def acvition_hook(model, input, output):
    activations.append(output.clone().detach().sum(dim=0))
    return


def Compute_activation_scores(activations_):
    """
    :argument 计算每个通道评价标准(重要性)
    :param activations_: [c,h,w]
    :return: [c]
    """
    activations_scores = []
    for activation in activations_:
        # activation = F.leaky_relu(activation)
        # activation = F.relu(activation)
        # 一阶范数
        # activations_scores.append(activation.cpu().norm(dim=(1, 2), p=1).cuda())
        # 二阶范数
        activations_scores.append(activation.cpu().norm(dim=(1, 2), p=2).cuda())
    return activations_scores


def Compute_activation_thresholds(activations_scores, percent):
    """
    :argument 通过channel的重要性水平，算出阈值
    :param activations_scores: 通道重要性
    :param percent: 剪枝比例(剪掉的比率)
    :return: 输出对每个bn层的阈值
    """

    thresholds = []
    for tensor in activations_scores:
        sorted_tensor, index = torch.sort(tensor)

        total = len(sorted_tensor)
        threshold_index = int(total * percent)
        threshold = sorted_tensor[threshold_index]

        # threshold = sorted_tensor.mean()

        thresholds.append(threshold)

    return thresholds


def Compute_layer_mask(imgs, model, percent, device):
    """
    :argument 根据输入图片计算masks
    :param percent: 剪枝比例(剪掉的比率)
    :argument 根据输入图片获取模型的mask
    :param imgs: 输入图片tensor
    :param model:
    :return: masks 维度为 [layer_num, c]
    """

    # 此处需要把模型更改为eval状态，否则在计算layer_mask时输入的数据会改变bn层参数，导致正确率下降
    model.eval()
    with torch.no_grad():
        imgs_masks = []
        one_img_mask = []
        hooks = []

        for img in imgs:
            activations.clear()
            one_img_mask.clear()
            hooks.clear()

            img = torch.unsqueeze(img, 0)

            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    hook = module.register_forward_hook(acvition_hook)
                    hooks.append(hook)

            _ = model(img)

            for hook in hooks:
                hook.remove()

            # 计算每个通道评价标准(重要性) [layer_num, c, h, w] => [layer_num, c]
            activations_scores = Compute_activation_scores(activations)
            # 计算阈值thresholds [layer_num, c] => [layer_num, 1]
            thresholds = Compute_activation_thresholds(activations_scores, percent)

            # 计算掩码 mask []
            for i in range(len(thresholds)):
                # [c]
                layer_mask = activations_scores[i].gt(thresholds[i]).to(device)
                # [layer_num, c]
                one_img_mask.append(layer_mask)

            imgs_masks.append(one_img_mask)

        # 合并 [image_num, layer_num, c] => [layer_num, c]
        img_num = len(imgs_masks)
        layer_num = len(imgs_masks[0])
        masks = imgs_masks[0]

        for i in range(layer_num):
            for j in range(1, img_num):
                masks[i] = masks[i] | imgs_masks[j][i]

        return masks


def pre_processing_Pruning(model: nn.Module, masks):
    """
    :argument: 根据输入的mask，计算生成新模型所需的cfg，以及对应的新的layer_mask
              （和原本mask比其实知识把前两层全部置为1，前两层不剪枝）
    :param model:输入的预剪枝模型
    :param masks:剪枝用到的mask
    :return:
    """
    model.eval()
    cfg = []  # 新的网络机构参数
    count = 0  # 层计数
    cfg_mask = []  # 计算新的mask
    pruned = 0  # 计算剪掉的通道数
    total = 0  # 总通道数

    for index, module in enumerate(model.modules()):

        if isinstance(module, nn.BatchNorm2d):

            mask = masks[count]
            # 前两层不剪枝
            if count <= 1:
                mask = mask | True

            # mask中0对应位置置0
            # module.weight.data.mul_(mask)
            # module.bias.data.mul_(mask)

            # 处理一下通道剩余0的情况
            if torch.sum(mask) == 0:
                mask[0] = 1

            # 当前层剩余通道
            cfg.append(int(torch.sum(mask)))
            # 当前层对应的mask向量
            cfg_mask.append(mask.clone())

            # 总通道数
            total += len(mask)

            # 总数减去保留的数量=剪掉的通道数
            pruned += len(mask) - torch.sum(mask)

            pruned_ratio = pruned / total

            count += 1

    return cfg, cfg_mask, pruned_ratio.detach().item()


def Real_Pruning(old_model: nn.Module, new_model: nn.Module, cfg_masks, reserved_class):
    """
    :argument 根据cfg_mask即每个bn层的mask，将原始模型的参数拷贝至新模型，同时调整新模型的cs层和linear层
              每个cs层通过设置index来实现剪枝
    :param old_model:
    :param new_model:
    :param cfg_masks:
    :param reserved_class: 保留下的类
    :return:返回剪枝后，拷贝完参数的模型，多余的类被剪掉
    """

    old_model.eval()
    new_model.eval()
    old_modules_list = list(old_model.named_modules())
    new_modules_list = list(new_model.named_modules())

    bn_idx = 0  # bn计数
    conv_idx = 0  # conv计数

    current_mask = torch.ones(16)  # 记录当前bn层的mask
    next_mask = cfg_masks[bn_idx]  # 记录下一层bn的mask

    # 因为上面用了list(name_modules) 所以其中 [0]表示name  [1]表示module
    for idx, (old, new) in enumerate(zip(old_modules_list, new_modules_list)):

        old_name = old[0]
        new_name = new[0]
        old_module = old[1]
        new_module = new[1]

        if isinstance(old_module, nn.BatchNorm2d):

            current_mask = next_mask
            next_mask = cfg_masks[bn_idx + 1 if bn_idx + 1 < len(cfg_masks) else bn_idx]

            # 如果下一层是cs层，曾调整cs层indexes以实现cs层剪枝
            if isinstance(old_modules_list[idx + 1][1], channel_selection):
                new_module.weight.data = old_module.weight.data.clone()
                new_module.bias.data = old_module.bias.data.clone()
                new_module.running_mean = old_module.running_mean.clone()
                new_module.running_var = old_module.running_var.clone()

                # 调整cs层index
                new_modules_list[idx + 1][1].indexes.data = current_mask

            # 下一层不是cs，则对bn层剪枝
            else:
                # True的位置保留， False位置直接移除
                # 输入对齐
                new_module.weight.data = old_module.weight.data.clone()[current_mask]
                new_module.bias.data = old_module.bias.data.clone()[current_mask]
                new_module.running_mean = old_module.running_mean.clone()[current_mask]
                new_module.running_var = old_module.running_var.clone()[current_mask]

            bn_idx += 1

        # 注意卷积层bias全部关掉，不用拷贝
        if isinstance(old_module, nn.Conv2d):

            # 第一个conv层为外部conv层，不剪枝
            if conv_idx == 0:
                new_module.weight.data = old_module.weight.data.clone()
                conv_idx += 1

            # 当前conv层前两层不是cs层也不是bn层(表示该层为downsample层) 不剪枝 直接拷贝
            elif not isinstance(old_modules_list[idx - 2][1], channel_selection) and \
                    not isinstance(old_modules_list[idx - 2][1], nn.BatchNorm2d):
                # print(old_name, new_name)
                new_module.weight.data = old_module.weight.data.clone()

            # 当前conv层根据其前面bn层进行剪枝
            else:
                # weight结构为[out_channel, in_channel, _, _]
                # 输出对齐
                conv_weight = old_module.weight.data.clone()[:, current_mask, :, :]

                # 每个block最后一层的输出不变
                if conv_idx % 3 != 0:
                    # 输出对齐
                    conv_weight = conv_weight[next_mask, :, :, :]

                new_module.weight.data = conv_weight

                # print(conv_weight.size())
                # print(new_module.weight.data.size())
                # print(old_module.weight.data.size())
                # print()

                conv_idx += 1

        # 对齐最后linear层于卷积的输出
        if isinstance(old_module, nn.Linear):
            # 替换掉原始fc
            input_size = sum(current_mask)
            out_size = len(reserved_class)
            new_model.fc = nn.Linear(input_size, out_size)

            # 原模型fc数据拷贝
            # 删除剪掉的类
            out_mask = torch.full([old_module.weight.data.size(0)], False)
            for i in reserved_class:
                out_mask[i] = True

            # 改变输入size
            fc_data = old_module.weight.data.clone()[:, current_mask]
            # 改变输出size
            fc_data = fc_data[out_mask, :]

            new_model.fc.weight.data = fc_data
            new_model.fc.bias.data = old_module.bias.data.clone()[out_mask]

    # test
    # aa = torch.randn(2, 3, 32, 32)
    # aa = aa.to(device)
    # print(new_model)
    # out1 = old_model(aa)
    # out2 = new_model(aa)
    return new_model


if __name__ == '__main__':
    # dataSet_name = 'CIFAR100'
    dataSet_name = 'CIFAR10'

    torch.manual_seed(1)
    np.random.seed(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = Data_Loader_CIFAR(train_batch_size=512, test_batch_size=1024, dataSet=dataSet_name)

    # model = resnet56(num_classes=data_loader.dataset_num_class).to(device)
    # model = torch.load(f='./models/ResNet/resnet32_before_9393.pkl').to(device)
    model = torch.load(f='./models/ResNet/resnet56_before_9423.pkl').to(device)
    # model = torch.load(f='../input/resnet-pruning-cifar-code/models/ResNet/resnet56_before_9423.pkl').to(device)

    # --------------------------------------------- 剪枝前模型测试
    # epoch_acc, epoch_class_correct, epoch_class_acc = all_test(model, data_loader.test_data_loader,
    #                                                            data_loader.dataset_num_class, device,
    #                                                            finnal_test=True)
    # print('\n', '1each class corrects: ', epoch_class_correct, '\n',
    #       '1each class accuracy: ', epoch_class_acc, '\n', '1total accuracy: ', epoch_acc)

    # --------------------------------------------- 预剪枝
    # 此处需要把模型更改为eval状态，否则在计算layer_mask时输入的数据会改变bn层参数，导致原模型正确率下降
    model.eval()

    reserved_classes_list = [[2, 4],
                             [2, 4, 7],
                             [2, 4, 7, 9],
                             [1, 2, 4, 7, 9],
                             [1, 2, 3, 4, 7, 9],
                             [1, 2, 3, 4, 6, 7, 9],
                             [1, 2, 3, 4, 6, 7, 8, 9],
                             [1, 2, 3, 4, 5, 6, 7, 8, 9],
                             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

    reserved_classes_list = [[0, 1, 2, 3, 4]]

    version_id = 2  # 指定
    model_id = 0    # 保存模型的id

    fine_tuning_epoch = 80
    manual_radio = 0.562

    fine_tuning_lr = 0.01
    fine_tuning_batch_size = 512
    fine_tuning_pics_num = 1024

    use_KL_divergence = False
    divide_radio = 4
    redundancy_num = 1024

    frozen = False

    version_msg = "版本备注:无"
    # version_msg = "版本备注:train的时候用eval训练,保持bn层数据不变"
    # version_msg = "版本备注:冻结除bn和fc的其他层,bn层冻结"

    reserved_classes = [0, 1, 2, 3, 4]

    # ----------------------------------------------------------------------
    # --------------
    # ----------------------------------------------------------------------

    imgs = read_Img_by_class(target_class=reserved_classes, pics_num=100,
                             data_loader=data_loader.test_data_loader, device=device)
    layer_masks = Compute_layer_mask(imgs=imgs, model=model, percent=manual_radio, device=device)
    # --------------------------------------------- 预剪枝,计算mask
    cfg, cfg_masks, pruned_radio = pre_processing_Pruning(model, layer_masks)

    # for reserved_classes in reserved_classes_list:
    # redundancy_num_list = [128, 256, 512, 1024]
    fine_tuning_epoch_list = [30, 40, 50, 60, 70, 80, 90, 100]
    for fine_tuning_epoch in fine_tuning_epoch_list:

        new_model = resnet56(data_loader.dataset_num_class, cfg=cfg).to(device)
        # --------------------------------------------- 正式剪枝,参数拷贝
        model_after_pruning = Real_Pruning(old_model=model, new_model=new_model,
                                           cfg_masks=cfg_masks, reserved_class=reserved_classes)
        print("model_id: " + str(model_id) + "  运行：")

        # --------------------------------------------- 微调
        # model_save_path = '/kaggle/working/version'
        model_save_path = './models/ResNet/version'

        model_save_path += str(version_id) + '_resnet56_after_model_' + str(model_id) + '.pkl'

        fine_tuning_loader = get_fine_tuning_data_loader(reserved_classes,
                                                         pics_num=fine_tuning_pics_num,
                                                         batch_size=fine_tuning_batch_size,
                                                         data_loader=data_loader.train_data_loader,
                                                         use_KL=use_KL_divergence,
                                                         redundancy_num=redundancy_num,
                                                         divide_radio=divide_radio)

        best_acc = fine_tuning(model_after_pruning, reserved_classes,
                               EPOCH=fine_tuning_epoch, lr=fine_tuning_lr,
                               device=device,
                               train_data_loader=fine_tuning_loader,
                               test_data_loader=data_loader.test_data_loader,
                               model_save_path=model_save_path,
                               use_all_data=False,
                               frozen=frozen)

        print("model_id:---" + str(model_id) +
              " best_acc:----" + str(best_acc) +
              " reserved_classes:---" + str(reserved_classes) +
              " manual_radio:---" + str(manual_radio) +
              " pruned_radio:---" + str(pruned_radio) +
              '\n')

        msg_save_path = "./model_msg3.txt"
        # msg_save_path = "/kaggle/working/model_msg.txt"
        with open(msg_save_path, "a") as fp:
            # fp.write("version_id:---" + str(version_id) +
            #          "  model_id:---" + str(model_id) +
            #          "  best_acc:----" + str(best_acc) +
            #          "  fine_tuning_batch_size:---" + str(fine_tuning_batch_size) +
            #          "  fine_tuning_pics_num:---" + str(fine_tuning_pics_num) +
            #          "  fine_tuning_epoch:---" + str(fine_tuning_epoch) +
            #          "  fine_tuning_lr:---" + str(fine_tuning_lr) +
            #          "  manual_radio:---" + str(manual_radio) +
            #          "  pruned_radio:---" + str(pruned_radio) +
            #          "  reserved_classes:---" + str(reserved_classes) +
            #          "  model_save_path---" + model_save_path +
            #          "\n")
            space = " "

            fp.write(str(version_id) + space +
                     str(model_id) + space +
                     str(round(best_acc + 0.0001, 4)) + space +
                     str(fine_tuning_batch_size) + space +
                     str(fine_tuning_pics_num) + space +
                     str(fine_tuning_epoch) + space +
                     str(fine_tuning_lr) + space +
                     str(redundancy_num) + space +
                     str(divide_radio) + space +
                     str(use_KL_divergence) + space +
                     str(round(manual_radio, 3)) + space +
                     str(round(pruned_radio, 4)) + space +
                     str(reserved_classes) + space +
                     version_msg + space +
                     model_save_path + space +
                     "\n")

        model_id += 1
