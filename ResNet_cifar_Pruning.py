import winnt

from models.ResNet_cifar_Model import *

from utils.Data_loader import Data_Loader_CIFAR
from utils.Functions import *

# ---------------------------------------
activations = []
record_activations = []


# 计算mask使用的activation_hook
def mask_activation_hook(module, input, output):
    global activations
    activations.append(output.clone().detach().cpu())
    return

# 记录前向推理使用的activation_hook
def forward_activation_hook(module, input, output):
    global record_activations
    # input_size [b, 256, 8, 8]
    # output_size [b, 256, 1, 1]
    record = output.clone().detach().view(output.size(0), -1).cpu()
    record_activations.append(record)
    return


def Compute_layer_mask(imgs, model, percent, device, activation_func):
    """
    :argument 根据输入图片计算masks
    :param percent: 保留的比例
    :argument 根据输入图片获取模型的mask
    :param imgs: 输入图片tensor
    :param model:
    :param activation_func:
    :return: masks 维度为 [layer_num, c]
    """

    percent = 1 - percent  # 通过保留比例 计算出需要剪掉的比例percent

    global activations
    # 此处需要把模型更改为eval状态，否则在计算layer_mask时输入的数据会改变bn层参数，导致正确率下降
    model.eval()

    with torch.no_grad():
        imgs_masks = []
        hooks = []
        activations.clear()

        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                hook = module.register_forward_hook(mask_activation_hook)
                hooks.append(hook)

        imgs = imgs.to(device)
        _ = model(imgs)

        for hook in hooks:
            hook.remove()

        # ------版本2 一层一层处理
        masks = []
        for layer_activations in activations:
            if activation_func is not None:
                layer_activations = activation_func(layer_activations)
            # [img_num, c, h, w] => [img_num, c] --- [800, 64, 32, 32] => [800, 64]
            layer_activations_score = layer_activations.norm(dim=(2, 3), p=2)
            # [img_num, c]  eg [800, 64]
            layer_masks = torch.empty_like(layer_activations_score, dtype=torch.bool)
            # [image_num, c] 计算每一张图片的mask
            for idx, imgs_activations_score in enumerate(layer_activations_score):
                # [c]
                sorted_tensor, index = torch.sort(imgs_activations_score)
                threshold_index = int(len(sorted_tensor) * percent)
                threshold = sorted_tensor[threshold_index]
                one_img_mask = imgs_activations_score.gt(threshold)
                layer_masks[idx] = one_img_mask

            # 2.2 统计true数量
            # [c]  [64]
            score_num = torch.sum(layer_masks, dim=0)
            sorted_tensor, _ = torch.sort(score_num)
            score_threshold_index = int(len(sorted_tensor) * percent)
            score_threshold = sorted_tensor[score_threshold_index]
            one_layer_mask = score_num.gt(score_threshold)

            masks.append(one_layer_mask)

        return masks


def pre_processing_Pruning(model: nn.Module, masks, jump_layers=2):
    """

    :argument: 根据输入的mask，计算生成新模型所需的cfg，以及对应的新的layer_mask
              （和原本mask比其实知识把前两层全部置为1，前两层不剪枝）
    :param model:输入的预剪枝模型
    :param masks:剪枝用到的mask
    :param jump_layers: 剪枝跳过几层
    :return:
    """
    model.eval()
    cfg = []  # 新的网络结构参数
    count = 0  # bn层计数
    cfg_mask = []  # 计算新的mask
    pruned = 0  # 计算剪掉的通道数
    total = 0  # 总通道数

    for index, module in enumerate(model.modules()):

        if isinstance(module, nn.BatchNorm2d):

            mask = masks[count]
            # 前两层不剪枝
            if count <= jump_layers:
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

            count += 1

    pruned_ratio = pruned / total

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
                new_modules_list[idx + 1][1].indexes.data = current_mask.clone()

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

                new_module.weight.data = conv_weight.clone()

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

            new_model.fc.weight.data = fc_data.clone()
            new_model.fc.bias.data = old_module.bias.data.clone()[out_mask]

    return new_model


if __name__ == '__main__':
    # dataSet_name = 'CIFAR100'
    dataSet_name = 'CIFAR10'

    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    # vis = Visdom()
    # # 加一行能显示标签
    # vis.line([0], [0], win='acc', name='line', opts=dict(legend=['']))
    # vis.line([0], [0], win='loss', name='line', opts=dict(legend=['']))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = Data_Loader_CIFAR(train_batch_size=1024, test_batch_size=1024,
                                    dataSet=dataSet_name, use_data_Augmentation=False,
                                    download=False, train_shuffle=True)

    # model = torch.load(f='./models/ResNet/resnet32_before_9393.pkl').to(device)
    model = torch.load(f='./models/ResNet/resnet56_before_9423.pkl').to(device)
    # model = torch.load(f='../input/resnet-pruning-cifar-code/models/ResNet/resnet56_before_9423.pkl').to(device)
    model.eval()

    # --------------------------------------------- 剪枝前模型测试
    # test_reserved_classes(model=model, device=device, reserved_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #                       test_data_loader=data_loader.test_data_loader, is_print=True, test_class=True)

    # reserved_classes_list = [[2, 4],
    #                          [2, 4, 7],
    #                          [2, 4, 7, 9],
    #                          [1, 2, 4, 7, 9],
    #                          [1, 2, 3, 4, 7, 9],
    #                          [1, 2, 3, 4, 6, 7, 9],
    #                          [1, 2, 3, 4, 6, 7, 8, 9],
    #                          [1, 2, 3, 4, 5, 6, 7, 8, 9],
    #                          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

    version_id = 18  # 指定
    model_id = 0  # 保存模型的id

    fine_tuning_epoch = 50
    manual_radio = 0.9

    fine_tuning_lr = 0.001
    fine_tuning_batch_size = 128
    fine_tuning_pics_num = 128

    use_KL_divergence = True
    divide_radio = 1
    redundancy_num = 0

    record_imgs_num = 512
    record_batch = 128

    frozen = False

    version_msg = "版本备注:方法2 kl选取 使用最大 图片池改为256 mile stone 20 40 batch size 256"

    msg_save_path = "./msg/model_msg2.txt"
    model_save_path = './models/ResNet/version'
    # msg_save_path = "/kaggle/working/model_msg.txt"
    # model_save_path = '/kaggle/working/version'
    model_save_path += str(version_id) + '_resnet56_after_model_' + str(model_id) + '.pkl'

    reserved_classes = [0, 1, 2, 3, 4, 5, 6, 7]
    max_kc = None
    min_kc = None
    FLOPs_radio = 0.00
    Parameters_radio = 0.00
    # ----------------------------------------------------------------------
    # --------------
    # ----------------------------------------------------------------------

    # ----------第一步：进行一定数量的前向推理forward，并记录图片和中间激活值
    record_dataloader = read_Img_by_class(pics_num=record_imgs_num,
                                          batch_size=record_batch,
                                          target_class=reserved_classes,
                                          data_loader=data_loader.train_data_loader,
                                          shuffle=False)

    for module in model.modules():
        if isinstance(module, nn.AvgPool2d):
            hook = module.register_forward_hook(forward_activation_hook)

    # 模拟实际前向推理(剪枝前)，记录数据
    model.eval()
    with torch.no_grad():
        for forward_x, _ in record_dataloader:
            forward_x = forward_x.to(device)
            _ = model(forward_x)

    hook.remove()
    # ----------第二部：根据前向推理图片获取保留的类，并计算mask，进行剪枝，获得剪枝后的新模型
    # 从记录数据中选取一部分计算mask
    imgs_tensor = choose_mask_imgs(target_class=reserved_classes,
                                   data_loader=record_dataloader,
                                   pics_num=10)
    layer_masks = Compute_layer_mask(imgs=imgs_tensor, model=model, percent=manual_radio, device=device,
                                     activation_func=nn.ReLU())

    # # 对比不同数量图片计算mask的作图
    # rv_pics_num_list = [1, 2, 5, 10, 20, 40, 80, 120]
    # for rv_pics_num in rv_pics_num_list:
    #
    #     imgs_tensor = choose_mask_imgs(target_class=reserved_classes,
    #                                    data_loader=record_dataloader,
    #                                    pics_num=rv_pics_num)
    #     layer_masks = Compute_layer_mask(imgs=imgs_tensor, model=model, percent=manual_radio, device=device)
    #
    #     for idx, mask_ in enumerate(layer_masks):
    #         mask = mask_.clone().cpu().numpy().reshape(1, -1)
    #         try:
    #             old_mask = np.load('./mask_numpy/ResNet56/layer' + str(idx + 1) + '.npy')
    #             new_mask = np.vstack((old_mask, mask))
    #             print(new_mask.shape)
    #             np.save('./mask_numpy/ResNet56/layer' + str(idx + 1) + '.npy', new_mask)
    #         except:
    #             np.save('./mask_numpy/ResNet56/layer' + str(idx + 1) + '.npy', mask)
    # exit(0)

    # 预剪枝,计算mask
    cfg, cfg_masks, filter_remove_radio = pre_processing_Pruning(model, layer_masks, jump_layers=3)
    filter_remain_radio = 1 - filter_remove_radio

    redundancy_num_list = [0, 16, 32, 64, 80, 100, 128, 200, 256, 320]
    for redundancy_num in redundancy_num_list:
        for i in range(3):

            # 根据cfg生成模型
            new_model = resnet56(data_loader.dataset_num_class, cfg=cfg).to(device)

            # 正式剪枝,参数拷贝
            model_after_pruning = Real_Pruning(old_model=model, new_model=new_model,
                                               cfg_masks=cfg_masks, reserved_class=reserved_classes)

            # 多GPU微调
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                model_after_pruning = nn.DataParallel(model_after_pruning)
            model_after_pruning.to(device)

            # 计算多种压缩率标准
            model.eval()
            model_after_pruning.eval()
            old_FLOPs, old_parameters = cal_FLOPs_and_Parameters(model, device)
            new_FLOPs, new_parameters = cal_FLOPs_and_Parameters(model_after_pruning, device)
            FLOPs_radio = new_FLOPs / old_FLOPs
            Parameters_radio = new_parameters / old_parameters

            # 剪枝后模型测试
            # test_reserved_classes(model=model_after_pruning, device=device, reserved_classes=reserved_classes,
            #                       test_data_loader=data_loader.test_data_loader, test_class=True, is_print=True)

            print("model_id:" + str(model_id)
                  + " ---filter_remain_radio:" + str(filter_remain_radio)
                  + " ---FLOPs_radio:" + str(FLOPs_radio)
                  + " ---Parameters_radio:" + str(Parameters_radio)
                  + "  运行：")

            # ----------第三步：从前向推理记录的图片中，使用算法选取一部分进行微调
            fine_tuning_loader, max_kc, min_kc = get_fine_tuning_data_loader2(record_activations,
                                                                              reserved_classes,
                                                                              pics_num=fine_tuning_pics_num,
                                                                              batch_size=fine_tuning_batch_size,
                                                                              data_loader=record_dataloader,
                                                                              redundancy_num=redundancy_num,
                                                                              divide_radio=divide_radio,
                                                                              use_max=True)
            # 微调
            best_acc, acc_list, loss_list = fine_tuning(model_after_pruning, reserved_classes,
                                                        EPOCH=fine_tuning_epoch, lr=fine_tuning_lr,
                                                        device=device,
                                                        train_data_loader=fine_tuning_loader,
                                                        test_data_loader=data_loader.test_data_loader,
                                                        model_save_path=model_save_path,
                                                        use_all_data=False,
                                                        frozen=frozen)

            # 画图
            # draw_acc_loss(acc_list=acc_list, loss_list=loss_list, line_id=model_id, vis=vis)

            print("model_id:---" + str(model_id) +
                  " best_acc:----" + str(best_acc) +
                  " reserved_classes:---" + str(reserved_classes) +
                  " manual_radio:---" + str(manual_radio) +
                  " filter_remain_radio:---" + str(filter_remain_radio) +
                  '\n')

            with open(msg_save_path, "a") as fp:
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
                         str(round(manual_radio, 4)) + space +
                         str(round(FLOPs_radio, 4)) + space +
                         str(round(Parameters_radio, 4)) + space +
                         str(round(filter_remain_radio, 4)) + space +
                         str(reserved_classes) + space +
                         str(max_kc) + space +
                         str(min_kc) + space +
                         version_msg + space +
                         model_save_path + space +
                         "\n")

            model_id += 1

    with open(msg_save_path, "a") as fp:
        fp.write("\n")
