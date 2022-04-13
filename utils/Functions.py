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

import matplotlib.pyplot as plt

def read_Img_by_class(target_class, pics_num, data_loader, device):
    """
    :param target_class: 读取类的标签 eg: [0, 1, 2, 3]
    :param pics_num:  每个类图片数量
    :param data_loader:  数据集 data_loader
    :param device:
    :return:
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
            if label[idx] in target_class and counts[target_class.index(label[idx])] < pics_num:
                inputs.append(data[idx].to(device))
                counts[target_class.index(label[idx])] += 1

    imgs = torch.empty(len(inputs), 3, 32, 32).to(device)

    for idx, img in enumerate(inputs):
        imgs[idx] = img

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

    optimizer.zero_grad()

    best_acc = 0

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

    return best_acc


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


def get_fine_tuning_data_loader(reserved_classes, pics_num, data_loader, batch_size,
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
                        image_Kc_list[list_idx][0] = 0.001

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

                Kc_max = max(image_Kc_list[list_idx])
                Kc_max_idx = np.argmax(image_Kc_list[list_idx])

                # Kc_min = min(image_Kc_list[list_idx])
                # Kc_min_idx = np.argmin(image_Kc_list[list_idx])

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

                if Kc < Kc_max:
                    image_Kc_list[list_idx][Kc_max_idx] = Kc
                    img_list[list_idx][Kc_max_idx] = data[idx]
                    label_list[list_idx][Kc_max_idx] = label[idx]
                    redundancy_counts[list_idx] += 1

                # if Kc > Kc_min:
                #     image_Kc_list[list_idx][Kc_min_idx] = Kc
                #     img_list[list_idx][Kc_min_idx] = data[idx]
                #     label_list[list_idx][Kc_min_idx] = label[idx]
                #     redundancy_counts[list_idx] += 1

    imgs = []
    labels = []
    for i, j in zip(img_list, label_list):
        imgs += i
        labels += j

    mydataset = myDataset(imgs, labels)

    new_data_loader = dataloader.DataLoader(mydataset, batch_size=batch_size, shuffle=True)

    return new_data_loader


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
