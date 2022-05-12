import random
import sys
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset, dataloader
from torchvision import datasets, transforms
from utils.Channel_selection import channel_selection

from copy import deepcopy
import matplotlib.pyplot as plt
from fvcore.nn import FlopCountAnalysis

class myDataset(Dataset):
    def __init__(self, img_list, label_list):
        """
        :argument 将图片数据和label数据转换为dataset类型，可以让fine-tuning直接调用。
        :param img_list:
        :param label_list:
        """
        self.img_list = img_list
        self.label_list = label_list

    def __getitem__(self, idx):
        img, label = self.img_list[idx], self.label_list[idx]
        return img, label

    def __len__(self):
        return len(self.img_list)


def read_Img_by_class(target_class, pics_num, data_loader, batch_size, shuffle=True):
    """
    :argument 模拟前向推理记录的数据集
    :param batch_size:
    :param target_class: 读取类的标签 eg: [0, 1, 2, 3]
    :param pics_num:  每个类图片数量
    :param data_loader:  数据集 data_loader
    :return: data_loader
    :param shuffle:
    """

    counts = []
    inputs = []
    labels = []

    for idx in range(len(target_class)):
        counts.append(0)
        inputs.append([])
        labels.append([])

    # image_num表示微调时选区的图片数量
    for data, label in data_loader:

        if sum(counts) == len(target_class) * pics_num:
            break

        for idx in range(len(label)):

            if label[idx] in target_class:
                list_idx = target_class.index(label[idx])
            else:
                continue

            if counts[target_class.index(label[idx])] < pics_num:
                inputs[list_idx].append(data[idx])
                labels[list_idx].append(label[idx])
                counts[list_idx] += 1

    imgs = []
    targets = []
    for i, j in zip(inputs, labels):
        imgs += i
        targets += j

    mydataset = myDataset(imgs, targets)
    record_dataloader = dataloader.DataLoader(mydataset, batch_size=batch_size, shuffle=shuffle)

    return record_dataloader


def choose_mask_imgs(target_class, pics_num, data_loader):
    """
    :argument 根据前向推理数据集，选取计算mask使用的imgs
    :param target_class:
    :param pics_num:  每个类图片数量
    :param data_loader:  数据集 data_loader
    :return: imgs
    """

    counts = []
    inputs = []

    for idx in range(len(target_class)):
        counts.append(0)

    # image_num表示微调时选区的图片数量
    for data, label in data_loader:

        if sum(counts) == len(target_class) * pics_num:
            break

        for idx in range(len(label)):
            if counts[target_class.index(label[idx])] < pics_num:
                inputs.append(data[idx])
                counts[target_class.index(label[idx])] += 1

    imgs = torch.stack(inputs, dim=0)

    return imgs


def test_reserved_classes(model, reserved_classes, test_data_loader,
                          device, is_print=True, test_class=True):
    """

    :param model:
    :param reserved_classes:
    :param test_data_loader:
    :param device:
    :param is_print:   是否输出
    :param test_class: 是否测试每个类
    :return:
    """

    model.to(device)
    model.eval()

    # 计算有多少类别
    dataset_num_class = max(test_data_loader.dataset.targets) + 1

    if test_class:
        class_correct = []
        class_num = []
        for _ in range(dataset_num_class):
            class_correct.append(0)
            class_num.append(0)

    with torch.no_grad():
        correct = 0
        num_data_all = 0
        for data, label in test_data_loader:
            input = data.to(device)
            target = label.to(device)  # [b]

            masks = torch.full([len(target)], False)

            for idx, i in enumerate(target):
                if i in reserved_classes:
                    masks[idx] = True

            input = input[masks, :, :, :]
            target = target[masks]

            output = model(input)
            pred = torch.argmax(output, 1)

            for idx, item in enumerate(pred):
                pred[idx] = reserved_classes[int(item)]

            if test_class:
                for index in range(len(target)):
                    if pred[index] == target[index]:
                        class_correct[target[index]] += 1
                    class_num[target[index]] += 1

            correct += (pred == target).sum()
            num_data_all += len(target)

        total_acc = float(correct / num_data_all)

        if test_class:
            class_acc = []
            for correct, nums in zip(class_correct, class_num):
                # 排除除0错误
                if nums == 0:
                    nums = 1
                class_acc.append(correct / nums)

        if is_print:
            if test_class:
                print('\n',
                      'each class corrects: ', class_correct, '\n',
                      'each class accuracy: ', class_acc, '\n',
                      'total accuracy: ', total_acc)
            else:
                print('\n', 'total accuracy: ', total_acc)

        if test_class:
            return round(total_acc, 4), class_correct, class_acc
        else:
            return round(total_acc, 4), None, None


def fine_tuning(model, reserved_classes, EPOCH, lr, model_save_path,
                train_data_loader, test_data_loader, device,
                use_all_data=True, frozen=False):

    # TODO: 冻结一部分？
    # if frozen:
    #     for param in model.parameters():
    #         param.requires_grad = False
    #
    #     conv_count = 10
    #     conv_idx = 0
    #
    #     for module in model.modules():
    #         if isinstance(module, nn.Linear):
    #             module.weight.requires_grad = True
    #             module.bias.requires_grad = True
    #         if isinstance(module, nn.Conv2d):
    #             module.weight.requires_grad = True
    #             # module.bias.requires_grad = True


    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=lr, weight_decay=5e-4, momentum=0.9)
    # optimizer = optim.AdamW(params=model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

    optimizer.zero_grad()

    best_acc = 0

    acc_list = []
    loss_list = []

    for epoch in range(EPOCH):
        model.train()

        epoch_loss = 0
        item_times = 0

        for idx, (data, label) in enumerate(tqdm(train_data_loader, desc='fine_tuning: ', file=sys.stdout)):
            data = data.to(device)
            label = label.to(device)

            if use_all_data:
                masks = torch.full([len(label)], False)

                for idx0, i in enumerate(label):
                    if i in reserved_classes:
                        masks[idx0] = True

                data = data[masks, :, :, :]
                label = label[masks]

            for idx0, item in enumerate(label):
                label[idx0] = reserved_classes.index(int(item))

            output = model(data)
            loss = loss_func(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().item()
            item_times += 1

        scheduler.step()

        epoch_acc, _, _ = test_reserved_classes(model, reserved_classes,
                                                test_data_loader,
                                                device,
                                                test_class=False,
                                                is_print=False)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            # print('model save')
            torch.save(model, model_save_path)

        if epoch % 10 == 0 and epoch != 0:
            test_reserved_classes(model, reserved_classes,
                                  test_data_loader,
                                  device,
                                  test_class=True,
                                  is_print=True)
        else:
            print("epoch:" + str(epoch) + "\tepoch_acc: "
                  + str(epoch_acc) + "\tepoch_loss: " + str(round(epoch_loss / item_times, 5)))

        acc_list.append(epoch_acc)
        loss_list.append(round(epoch_loss / item_times, 5))

    return best_acc, acc_list, loss_list


def get_fine_tuning_data_loader1(reserved_classes, pics_num, data_loader, batch_size,
                                use_KL=False, divide_radio=4, redundancy_num=50, use_norm=False):
    """

    :param reserved_classes: 读取类的标签 eg: [0, 1, 2, 3]
    :param pics_num:  每个类图片数量
    :param data_loader:
    :param batch_size: 输出data_loader的batch_size

    :param use_KL:
    :param divide_radio:
    :param redundancy_num:
    :param use_norm: KL-div前面加上norm

    :return: 返回微调数据的data_loader
    """

    counts = []
    img_list = []
    label_list = []
    redundancy_counts = []

    for idx in range(len(reserved_classes)):
        counts.append(0)
        redundancy_counts.append(0)
        img_list.append([])
        label_list.append([])

    if use_KL:
        image_Kc_list = np.zeros([len(reserved_classes), pics_num])

    for data, label in tqdm(data_loader, desc='choosing fine tuning data: ', file=sys.stdout):

        # 若没有使用KL-div 则能直接跳过and后面的判断
        if sum(counts) == len(reserved_classes) * pics_num \
                and ((not use_KL) or sum(redundancy_counts) == len(reserved_classes) * redundancy_num):
            break

        for idx in range(len(label)):

            if label[idx] in reserved_classes:
                list_idx = reserved_classes.index(label[idx])
            else:
                continue

            if counts[list_idx] < pics_num:

                # 使用kl-divergence 且图片还未满
                if use_KL:
                    dim = -1
                    # 如果是第一张图片 则将其Kc值置为100 很大的值
                    if counts[list_idx] == 0:
                        image_Kc_list[list_idx][0] = 100.0

                    # 不是第一张图
                    else:
                        # 小于划分阈值 则全部计算
                        if counts[list_idx] < pics_num / divide_radio:
                            KL_all = 0
                            for image_ in img_list[list_idx]:
                                if not use_norm:
                                    KL_all += F.kl_div(data[idx].softmax(dim=dim).log(), image_.softmax(dim=dim),
                                                       reduction='batchmean')
                                else:
                                    data1 = data[idx].norm(dim=(1, 2), p=2)
                                    data2 = image_.norm(dim=(1, 2), p=2)
                                    KL_all += F.kl_div(data1.softmax(dim=0).log(), data2.softmax(dim=0),
                                                       reduction='batchmean')

                                # x_down = F.adaptive_avg_pool2d(x[idx_], 1).squeeze()
                                # KL_all +=
                            Kc = KL_all / counts[list_idx]

                        # 大于划分阈值 则随机选择计算
                        else:
                            KL_all = 0
                            samples = [ig for ig in range(counts[list_idx])]
                            sample = random.sample(samples, int(pics_num / divide_radio))

                            for random_i in sample:
                                # data[idx]当前图片 img_list[list_idx][random_i]已存图片随机选择一张
                                if not use_norm:
                                    KL_all += F.kl_div(data[idx].softmax(dim=dim).log(),
                                                       img_list[list_idx][random_i].softmax(dim=dim),
                                                       reduction='batchmean')
                                else:
                                    data1 = data[idx].norm(dim=(1, 2), p=2)
                                    data2 = img_list[list_idx][random_i].norm(dim=(1, 2), p=2)
                                    KL_all += F.kl_div(data1.softmax(dim=0).log(), data2.softmax(dim=0),
                                                       reduction='batchmean')
                            Kc = KL_all / len(sample)

                        # 储存当前图片的Kc值
                        image_Kc_list[list_idx][counts[list_idx]] = Kc

                img_list[list_idx].append(data[idx])
                label_list[list_idx].append(label[idx])
                counts[list_idx] += 1

            # 使用kl且图片已满，冗余
            elif use_KL and counts[list_idx] == pics_num and redundancy_counts[list_idx] < redundancy_num:

                # Kc_max = max(image_Kc_list[list_idx])
                # Kc_max_idx = np.argmax(image_Kc_list[list_idx])

                Kc_min = min(image_Kc_list[list_idx])
                Kc_min_idx = np.argmin(image_Kc_list[list_idx])

                samples = [ig for ig in range(counts[list_idx])]
                sample = random.sample(samples, int(pics_num / divide_radio))

                KL_all = 0
                for random_i in sample:
                    # x[idx_]当前图片 image_data_list[list_idx][random_i]已存图片随机选择一张
                    if not use_norm:
                        KL_all += F.kl_div(data[idx].softmax(dim=dim).log(),
                                           img_list[list_idx][random_i].softmax(dim=dim),
                                           reduction='batchmean')
                    else:
                        data1 = data[idx].norm(dim=(1, 2), p=2)
                        data2 = img_list[list_idx][random_i].norm(dim=(1, 2), p=2)
                        KL_all += F.kl_div(data1.softmax(dim=0).log(), data2.softmax(dim=0),
                                           reduction='batchmean')

                Kc = KL_all / len(sample)

                # if Kc < Kc_max:
                #     image_Kc_list[list_idx][Kc_max_idx] = Kc
                #     img_list[list_idx][Kc_max_idx] = data[idx]
                #     label_list[list_idx][Kc_max_idx] = label[idx]
                #     redundancy_counts[list_idx] += 1

                if Kc > Kc_min:
                    image_Kc_list[list_idx][Kc_min_idx] = Kc
                    img_list[list_idx][Kc_min_idx] = data[idx]
                    label_list[list_idx][Kc_min_idx] = label[idx]
                    redundancy_counts[list_idx] += 1

    imgs = []
    labels = []
    for i, j in zip(img_list, label_list):
        imgs += i
        labels += j

    mydataset = myDataset(imgs, labels)

    new_data_loader = dataloader.DataLoader(mydataset, batch_size=batch_size, shuffle=True)

    return new_data_loader


def get_fine_tuning_data_loader2(record_features, reserved_classes, pics_num, data_loader, batch_size,
                                 divide_radio=1, redundancy_num=50, use_max=True):
    """
    :argument 根据模型前向推理记录的图片数据和特征，使用kl_div选取一部分图片来进行微调

    :param record_features: (list)前向推理中记录的特征，和data_loader里面的数据顺序一致
    :param reserved_classes: 读取类的标签 eg: [0, 1, 2, 3]
    :param pics_num:  每个类图片数量
    :param data_loader: 前向推理保存的数据
    :param batch_size: 输出data_loader的batch_size

    :param divide_radio:
    :param redundancy_num:
    :param use_max: True保留最大值，丢掉最小值   False保留最小值，丢掉最大值

    :return: 返回微调数据的data_loader
    """

    counts = []
    img_list = []
    label_list = []
    redundancy_counts = []
    feature_list = []

    for idx in range(len(reserved_classes)):
        counts.append(0)
        redundancy_counts.append(0)
        img_list.append([])
        label_list.append([])
        feature_list.append([])

    image_Kc_list = np.zeros([len(reserved_classes), pics_num])

    for (data, label), features in tqdm(zip(data_loader, record_features), desc='choosing fine tuning data: ', file=sys.stdout):

        if sum(counts) == len(reserved_classes) * pics_num and sum(redundancy_counts) == len(reserved_classes) * redundancy_num:
            break

        for idx in range(len(label)):

            list_idx = reserved_classes.index(label[idx])

            # 图片还未满
            if counts[list_idx] < pics_num:

                # 如果是第一张图片 则将其Kc值置为100 很大的值
                if counts[list_idx] == 0:
                    if use_max:
                        image_Kc_list[list_idx][0] = 1.0
                    else:
                        image_Kc_list[list_idx][0] = 0.0001

                # 不是第一张图
                else:
                    # 小于划分阈值 则全部计算
                    if counts[list_idx] < pics_num / divide_radio:

                        old_features = torch.stack(feature_list[list_idx])
                        Kc = F.kl_div(features[idx].softmax(dim=0).log(), old_features.softmax(dim=1),
                                      reduction='batchmean') / len(old_features)

                    # 大于划分阈值 则随机选择计算
                    else:
                        samples = random.sample([ig for ig in range(counts[list_idx])], int(pics_num / divide_radio))
                        feature_mask = torch.full([counts[list_idx]], False)
                        for i in range(counts[list_idx]):
                            if i in samples:
                                feature_mask[i] = True
                        old_features = torch.stack(feature_list[list_idx])[feature_mask]

                        Kc = F.kl_div(features[idx].softmax(dim=0).log(), old_features.softmax(dim=1),
                                      reduction='batchmean') / len(old_features)

                    # 储存当前图片的Kc值
                    image_Kc_list[list_idx][counts[list_idx]] = Kc

                img_list[list_idx].append(data[idx])
                label_list[list_idx].append(label[idx])
                feature_list[list_idx].append(features[idx])
                counts[list_idx] += 1

            # 使用kl且图片已满，冗余
            elif counts[list_idx] == pics_num and redundancy_counts[list_idx] < redundancy_num:

                if use_max:
                    Kc_min = min(image_Kc_list[list_idx])
                    Kc_min_idx = np.argmin(image_Kc_list[list_idx])
                else:
                    Kc_max = max(image_Kc_list[list_idx])
                    Kc_max_idx = np.argmax(image_Kc_list[list_idx])

                samples = random.sample([ig for ig in range(counts[list_idx])], int(pics_num / divide_radio))
                feature_mask = torch.full([counts[list_idx]], False)
                for i in range(counts[list_idx]):
                    if i in samples:
                        feature_mask[i] = True
                old_features = torch.stack(feature_list[list_idx])[feature_mask]

                Kc = F.kl_div(features[idx].softmax(dim=0).log(), old_features.softmax(dim=1),
                              reduction='batchmean') / len(old_features)

                # 使用最大，替换最小
                if use_max:
                    if Kc > Kc_min:
                        image_Kc_list[list_idx][Kc_min_idx] = Kc
                        img_list[list_idx][Kc_min_idx] = data[idx]
                        label_list[list_idx][Kc_min_idx] = label[idx]
                        feature_list[list_idx][Kc_min_idx] = features[idx]
                        redundancy_counts[list_idx] += 1

                # 使用最小，替换最大
                else:
                    if Kc < Kc_max:
                        image_Kc_list[list_idx][Kc_max_idx] = Kc
                        img_list[list_idx][Kc_max_idx] = data[idx]
                        label_list[list_idx][Kc_max_idx] = label[idx]
                        feature_list[list_idx][Kc_max_idx] = features[idx]
                        redundancy_counts[list_idx] += 1


    imgs = []
    labels = []
    for i, j in zip(img_list, label_list):
        imgs += i
        labels += j

    max_kc = []
    min_kc = []
    for kk in image_Kc_list:
        kk = list(kk)
        if use_max:
            kk.remove(max(kk))
        else:
            kk.remove(min(kk))
        max_kc.append(round(max(kk), 6))
        min_kc.append(round(min(kk), 6))

    mydataset = myDataset(imgs, labels)

    new_data_loader = dataloader.DataLoader(mydataset, batch_size=batch_size, shuffle=True)

    return new_data_loader, max_kc, min_kc


def get_fine_tuning_data_loader3(record_features, reserved_classes, pics_num, data_loader, batch_size,
                                 redundancy_num=50, use_max=False):
    """
    :argument 根据模型前向推理记录的图片数据和特征，使用均值选取一部分图片来进行微调

    :param record_features: (list)前向推理中记录的特征，和data_loader里面的数据顺序一致
    :param reserved_classes: 读取类的标签 eg: [0, 1, 2, 3]
    :param pics_num:  每个类图片数量
    :param data_loader: 前向推理保存的数据
    :param batch_size: 输出data_loader的batch_size
    :param use_max: True保留大均值误差， 丢掉小均值
                    False保留小均值误差，丢掉大均值
    :param redundancy_num:

    :return: 返回微调数据的data_loader
    """

    # [class_num, c]  [8, 256]
    mean_list = cal_feature_mean(record_features, reserved_classes)

    counts = []
    img_list = []
    label_list = []
    redundancy_counts = []

    for idx in range(len(reserved_classes)):
        counts.append(0)
        redundancy_counts.append(0)
        img_list.append([])
        label_list.append([])

    image_Kc_list = np.zeros([len(reserved_classes), pics_num])

    for (data, label), features in tqdm(zip(data_loader, record_features), desc='choosing fine tuning data: ', file=sys.stdout):

        if sum(counts) == len(reserved_classes) * pics_num and sum(redundancy_counts) == len(reserved_classes) * redundancy_num:
            break

        for idx in range(len(label)):

            list_idx = reserved_classes.index(label[idx])

            # 图片还未满
            if counts[list_idx] < pics_num:

                # 计算与均值差值绝对值之和，作为kc值
                image_Kc_list[list_idx][counts[list_idx]] = (features[idx] - mean_list[list_idx]).norm(dim=0, p=1)

                img_list[list_idx].append(data[idx])
                label_list[list_idx].append(label[idx])
                counts[list_idx] += 1

            # 使用kl且图片已满，冗余
            elif counts[list_idx] == pics_num and redundancy_counts[list_idx] < redundancy_num:

                if use_max:
                    Kc_min = min(image_Kc_list[list_idx])
                    Kc_min_idx = np.argmin(image_Kc_list[list_idx])
                else:
                    Kc_max = max(image_Kc_list[list_idx])
                    Kc_max_idx = np.argmax(image_Kc_list[list_idx])

                Kc = (features[idx] - mean_list[list_idx]).norm(dim=0, p=1)


                # 使用最大，替换最小
                if use_max:
                    if Kc > Kc_min:
                        image_Kc_list[list_idx][Kc_min_idx] = Kc
                        img_list[list_idx][Kc_min_idx] = data[idx]
                        label_list[list_idx][Kc_min_idx] = label[idx]
                        redundancy_counts[list_idx] += 1

                # 使用最小，替换最大
                else:
                    if Kc < Kc_max:
                        image_Kc_list[list_idx][Kc_max_idx] = Kc
                        img_list[list_idx][Kc_max_idx] = data[idx]
                        label_list[list_idx][Kc_max_idx] = label[idx]
                        redundancy_counts[list_idx] += 1


    imgs = []
    labels = []
    for i, j in zip(img_list, label_list):
        imgs += i
        labels += j

    max_kc = []
    min_kc = []
    for kk in image_Kc_list:
        kk = list(kk)
        max_kc.append(round(max(kk), 6))
        min_kc.append(round(min(kk), 6))

    mydataset = myDataset(imgs, labels)

    new_data_loader = dataloader.DataLoader(mydataset, batch_size=batch_size, shuffle=True)

    return new_data_loader, max_kc, min_kc


def get_fine_tuning_data_loader4(record_features, reserved_classes, pics_num, data_loader, batch_size,
                                 redundancy_num=50, use_max=False):
    """
    :argument 根据模型前向推理记录的图片数据和特征，使用均值选取一部分图片来进行微调
    :argument ！！和3比较，使用更加随机的方法来找到均值接近的图片池

    :param record_features: (list)前向推理中记录的特征，和data_loader里面的数据顺序一致
    :param reserved_classes: 读取类的标签 eg: [0, 1, 2, 3]
    :param pics_num:  每个类图片数量
    :param data_loader: 前向推理保存的数据
    :param batch_size: 输出data_loader的batch_size
    :param use_max: True保留大均值误差， 丢掉小均值
                    False保留小均值误差，丢掉大均值
    :param redundancy_num:.

    :return: 返回微调数据的data_loader
    """

    # [class_num * pics_num, c] => [8, 256]
    mean_list = cal_feature_mean(record_features, reserved_classes)

    counts = []
    img_list = []
    label_list = []
    redundancy_counts = []
    feature_list = []

    for idx in range(len(reserved_classes)):
        counts.append(0)
        redundancy_counts.append(0)
        img_list.append([])
        label_list.append([])
        feature_list.append([])

    # image_Kc_list = np.zeros([len(reserved_classes), pics_num])

    for (data, label), features in tqdm(zip(data_loader, record_features), desc='choosing fine tuning data: ', file=sys.stdout):

        if sum(counts) == len(reserved_classes) * pics_num and sum(redundancy_counts) == len(reserved_classes) * redundancy_num:
            break

        for idx in range(len(label)):

            list_idx = reserved_classes.index(label[idx])

            # 图片还未满
            if counts[list_idx] < pics_num:

                # 计算与均值差值绝对值之和，作为kc值
                # image_Kc_list[list_idx][counts[list_idx]] = (features[idx] - mean_list[list_idx]).norm(dim=0, p=1)

                img_list[list_idx].append(data[idx])
                label_list[list_idx].append(label[idx])
                feature_list[list_idx].append(features[idx])
                counts[list_idx] += 1

            # 使用kl且图片已满，冗余
            elif counts[list_idx] == pics_num and redundancy_counts[list_idx] < redundancy_num:

                old_score = (mean_list[list_idx] - torch.mean(torch.stack(feature_list[list_idx], dim=0), dim=0)).norm(dim=0, p=1)

                scores = []
                for feature_idx in range(len(feature_list[list_idx])):
                    new_feature_list = deepcopy(feature_list[list_idx])
                    new_feature_list[feature_idx] = features[idx]

                    score = (mean_list[list_idx] - torch.mean(torch.stack(new_feature_list, dim=0), dim=0)).norm(dim=0, p=1)
                    scores.append(score)

                if use_max:
                    score_max = max(scores)
                    score_idx = np.argmax(scores)
                    if old_score > score_max:
                        continue

                else:
                    score_min = min(scores)
                    score_idx = np.argmin(scores)
                    if old_score < score_min:
                        continue

                img_list[list_idx][score_idx] = data[idx]
                label_list[list_idx][score_idx] = label[idx]
                feature_list[list_idx][score_idx] = features[idx]
                redundancy_counts[list_idx] += 1

    imgs = []
    labels = []
    for i, j in zip(img_list, label_list):
        imgs += i
        labels += j

    max_kc = []
    min_kc = []
    # for kk in image_Kc_list:
    #     kk = list(kk)
    #     max_kc.append(round(max(kk), 6))
    #     min_kc.append(round(min(kk), 6))

    mydataset = myDataset(imgs, labels)

    new_data_loader = dataloader.DataLoader(mydataset, batch_size=batch_size, shuffle=True)

    return new_data_loader, max_kc, min_kc



def get_fine_tuning_data_loader5(record_features, reserved_classes, pics_num, data_loader, batch_size,
                                 redundancy_num=50, use_max=False):
    """
    :argument 根据模型前向推理记录的图片数据和特征，使用均值选取一部分图片来进行微调
    :argument ！！和3比较，使用更加随机的方法来找到均值接近的图片池

    :param record_features: (list)前向推理中记录的特征，和data_loader里面的数据顺序一致
    :param reserved_classes: 读取类的标签 eg: [0, 1, 2, 3]
    :param pics_num:  每个类图片数量
    :param data_loader: 前向推理保存的数据
    :param batch_size: 输出data_loader的batch_size
    :param use_max: True保留大均值误差， 丢掉小均值
                    False保留小均值误差，丢掉大均值
    :param redundancy_num:.

    :return: 返回微调数据的data_loader
    """

    # [class_num, c]  [8, 256]
    mean_list = cal_feature_mean(record_features, reserved_classes)

    counts = []
    img_list = []
    label_list = []
    redundancy_counts = []
    feature_list = []

    for idx in range(len(reserved_classes)):
        counts.append(0)
        redundancy_counts.append(0)
        img_list.append([])
        label_list.append([])
        feature_list.append([])

    # image_Kc_list = np.zeros([len(reserved_classes), pics_num])

    for (data, label), features in tqdm(zip(data_loader, record_features), desc='choosing fine tuning data: ', file=sys.stdout):

        if sum(counts) == len(reserved_classes) * pics_num and sum(redundancy_counts) == len(reserved_classes) * redundancy_num:
            break

        for idx in range(len(label)):

            list_idx = reserved_classes.index(label[idx])

            # 图片还未满
            if counts[list_idx] < pics_num:

                # 计算与均值差值绝对值之和，作为kc值
                # image_Kc_list[list_idx][counts[list_idx]] = (features[idx] - mean_list[list_idx]).norm(dim=0, p=1)

                img_list[list_idx].append(data[idx])
                label_list[list_idx].append(label[idx])
                feature_list[list_idx].append(features[idx])
                counts[list_idx] += 1

            # 使用kl且图片已满，冗余
            elif counts[list_idx] == pics_num and redundancy_counts[list_idx] < redundancy_num:

                old_score = (mean_list[list_idx] - torch.mean(torch.stack(feature_list[list_idx], dim=0), dim=0)).norm(dim=0, p=1)

                scores = []
                for feature_idx in range(len(feature_list[list_idx])):
                    new_feature_list = deepcopy(feature_list[list_idx])
                    new_feature_list[feature_idx] = features[idx]

                    score = (mean_list[list_idx] - torch.mean(torch.stack(new_feature_list, dim=0), dim=0)).norm(dim=0, p=1)
                    scores.append(score)

                if use_max:
                    score_max = max(scores)
                    score_idx = np.argmax(scores)
                    if old_score > score_max:
                        continue

                else:
                    score_min = min(scores)
                    score_idx = np.argmin(scores)
                    if old_score < score_min:
                        continue

                img_list[list_idx][score_idx] = data[idx]
                label_list[list_idx][score_idx] = label[idx]
                feature_list[list_idx][score_idx] = features[idx]
                redundancy_counts[list_idx] += 1

    imgs = []
    labels = []
    for i, j in zip(img_list, label_list):
        imgs += i
        labels += j

    max_kc = []
    min_kc = []
    # for kk in image_Kc_list:
    #     kk = list(kk)
    #     max_kc.append(round(max(kk), 6))
    #     min_kc.append(round(min(kk), 6))

    mydataset = myDataset(imgs, labels)

    new_data_loader = dataloader.DataLoader(mydataset, batch_size=batch_size, shuffle=True)

    return new_data_loader, max_kc, min_kc


def show_imgs(imgs: torch.Tensor):
    """
    :argument: 显示图片
    :param imgs: [b, c, h, w]
    :return:
    """
    unloader = transforms.ToPILImage()
    for idx, img in enumerate(imgs):
        image = img.clone().cpu()
        image = unloader(image)

        plt.subplot(5, 10, idx+1)
        plt.title(str(idx))
        plt.axis('off')
        plt.imshow(image)

    plt.show()


def cal_feature_mean(record_features, reserved_classes):
    """
    :argument 计算feature均值
    :param record_features:
    :param reserved_classes:
    :return:
    """
    means = []
    for i in record_features:
        means.append(torch.mean(i, dim=0))
    divide_num = int(len(means) / len(reserved_classes))

    mean_list = []
    for i in range(len(reserved_classes)):
        mean_list.append(torch.mean(torch.stack(means[i*divide_num:(i+1)*divide_num], dim=0), dim=0))

    return mean_list


def cal_FLOPs_and_Parameters(model, device):
    """
    :argument 模型计算FLOPs和参数量
    :param model:
    :param device:
    :return: FLOPs:G 参数量:M
    """

    tensor = torch.rand(1, 3, 32, 32).to(device)

    flops = FlopCountAnalysis(model, (tensor, ))

    flops_total = flops.total() / 10e9  # G
    parameters_total = sum([param.nelement() for param in model.parameters()]) / 10e6   # M

    return flops_total, parameters_total




