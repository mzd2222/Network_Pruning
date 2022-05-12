import sys
from tqdm import tqdm
import torch.optim as optim
# from torchsummary import summary
from models.ResNet_cifar_Model import *

from utils.Data_loader import Data_Loader_CIFAR

# ----------------------------------------------- 训练用test
def all_test(model, test_data_loader, device, finnal_test=False):
    model.to(device)
    model.eval()

    # 计算有多少类别
    dataset_num_class = max(test_data_loader.dataset.targets) + 1

    correct = 0
    if finnal_test:
        class_correct = []
        class_num = []
        for _ in range(dataset_num_class):
            class_correct.append(0)
            class_num.append(0)

    with torch.no_grad():
        for _, (data, label) in enumerate(tqdm(test_data_loader, desc='testing: ', file=sys.stdout)):
            input = data.to(device)
            target = label.to(device)

            # print(target.size())

            output = model(input)
            pred = torch.argmax(output, 1)

            if finnal_test:
                for index in range(len(target)):
                    if pred[index] == target[index]:
                        class_correct[target[index]] += 1
                    class_num[target[index]] += 1

            correct += (pred == target).sum()

        total_acc = float(correct / len(test_data_loader.dataset))

        model.train()

        if finnal_test:
            class_acc = []
            for correct, nums in zip(class_correct, class_num):
                class_acc.append(correct / nums)
            return round(total_acc, 4), class_correct, class_acc

        else:

            return round(total_acc, 4), None, None



# ----------------------------------------------- train
def train_new_model(model, data_loader, device, EPOCH, lr, model_save_path):

    model.to(device)
    model.train()

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=lr, weight_decay=5e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.3, verbose=True)

    optimizer.zero_grad()

    best_acc = 0

    for epoch in range(EPOCH):
        epoch_loss = 0
        item_times = 0

        for idx, (data, label) in enumerate(tqdm(data_loader.train_data_loader, desc='training: ', file=sys.stdout)):
            data = data.cuda()
            label = label.cuda()

            output = model(data)
            loss = loss_func(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            item_times += 1

        scheduler.step()

        epoch_acc, _, _ = all_test(model, data_loader.test_data_loader, device, finnal_test=False)

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            # print('model save')
            torch.save(model, model_save_path)

        if epoch % 10 == 0 and epoch != 0:
            epoch_acc, epoch_class_correct, epoch_class_acc = all_test(model, data_loader.test_data_loader,
                                                                       device, finnal_test=True)
            print('\n',
                  'epoch: ' + str(epoch), '\n',
                  'each class corrects: ', epoch_class_correct, '\n',
                  'each class accuracy: ', epoch_class_acc, '\n',
                  'total accuracy: ', epoch_acc)
        else:
            print("epoch:" + str(epoch) + "\tepoch_acc: "
                       + str(epoch_acc) + "\tepoch_loss: " + str(round(epoch_loss / item_times, 5)))


if __name__ == '__main__':

    # dataSet = 'CIFAR100'
    dataSet = 'CIFAR10'

    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = Data_Loader_CIFAR(test_batch_size=512, train_batch_size=512)
    # data_loader = Data_Loader_CIFAR(train_batch_size=128, test_batch_size=512, dataSet=dataSet)
    # model = resnet32(data_loader.dataset_num_class).to(device)
    # model_save_path = './models/ResNet/resnet32_before.pkl'
    #
    # train_new_model(model, data_loader, device, EPOCH=100, lr=0.1, model_save_path=model_save_path)

    model = torch.load(f='./models/ResNet/resnet56_before_9423.pkl').to(device)
    epoch_acc, epoch_class_correct, epoch_class_acc = all_test(model, data_loader.test_data_loader,
                                                               device, finnal_test=True)
    print('\n', '1each class corrects: ', epoch_class_correct, '\n',
          '1each class accuracy: ', epoch_class_acc, '\n', '1total accuracy: ', epoch_acc)