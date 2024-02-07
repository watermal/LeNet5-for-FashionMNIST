import time

import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

from LeNet import Net


def test_dataloader():
    test_data = FashionMNIST(root='./data',
                             train=False,
                             download=True,
                             transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]))

    test_Loader = Data.DataLoader(dataset=test_data,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=0)
    print("Test DataLoader Loaded.")
    return test_Loader, test_data.classes


def test_net(net, test_loaders):
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net.to(device)

    # Init
    test_correct, test_num = 0, 0
    with torch.no_grad():
        for step, (images, labels) in enumerate(test_loaders):
            images, labels = images.to(device), labels.to(device)
            model.eval()
            output = model(images)
            predicted = torch.argmax(output, dim=1)
            test_correct += 1 if predicted == labels else 0
            test_num += images.size(0)
            print("Predicted: {} ,\t Actual: {}"
                  .format(classes[predicted.item()], classes[labels.item()]))
    test_acc = test_correct / test_num
    print("Accuracy of the network on the {} test".format(test_num))
    return test_acc


if __name__ == '__main__':
    test_data_loader, _ = test_dataloader()
    since = time.time()
    pre_model = Net()
    # pre_model.load_state_dict(torch.load('./model/maxpool_best_model_20240207_125746.pth')) # 72.1%
    pre_model.load_state_dict(torch.load('./model/best_model_20240207_120921.pth'))  # 87.3%
    correct = test_net(pre_model, test_data_loader)
    print("Test Accuracy is : {:.2f} % , Spend time is : {:.3f}s"
          .format(correct * 100, time.time() - since))
