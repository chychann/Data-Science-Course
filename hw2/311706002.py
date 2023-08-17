import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as Fun
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torchvision.models as models
from torchsummary import summary
from torchvision.models import resnet18
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# Load Data
## Copied from sample predicted.py
def load_data():
  # transform
  transform = transforms.Compose(
      [transforms.Grayscale(num_output_channels=3),  # gray to 3 channel
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))])
  #


  # load Fashion MNIST
  trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                  download=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                  shuffle=True, num_workers=0)

  testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                  download=True, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                  shuffle=False, num_workers=0)
  return trainloader, testloader
# Knowledge Distillation
## Load Teacher Model
# Copied from sample predicted.py
class ResNet(nn.Module):
      def __init__(self):
            super(ResNet, self).__init__()
            self.resnet50 = resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            num_ftrs = self.resnet50.fc.in_features
            self.resnet50.fc = nn.Linear(num_ftrs, 10)

      def forward(self, x):
            x = self.resnet50.conv1(x)
            x = self.resnet50.bn1(x)
            x = self.resnet50.relu(x)
            x = self.resnet50.maxpool(x)

            x = self.resnet50.layer1(x)
            x = self.resnet50.layer2(x)
            x = self.resnet50.layer3(x)
            x = self.resnet50.layer4(x)

            x = self.resnet50.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.resnet50.fc(x)
            return x
## Student model - CNN
class _ConvLayer(nn.Sequential):
  def __init__(self, input_features, output_features):
    super(_ConvLayer, self).__init__()
    self.add_module('conv', nn.Conv2d(input_features, output_features,
                    kernel_size=3, stride=1, padding=1, bias=False))
    self.add_module('relu', nn.ReLU(inplace=True))
    self.add_module('norm', nn.BatchNorm2d(output_features))

  def forward(self, x):
    x = super(_ConvLayer, self).forward(x)
    return x

class CNN(nn.Module):
  def __init__(self):
    # Inheritance cnn
    super(CNN, self).__init__()
    self.features = nn.Sequential()
    #add_module(name,module)
    # Convolution 1 
    self.features.add_module('convlayer1', _ConvLayer(3, 32))
    self.features.add_module('maxpool', nn.MaxPool2d(2, 2))
    self.features.add_module('convlayer2', _ConvLayer(32, 64))
    self.features.add_module('avgpool', nn.AvgPool2d(2, 2))
    self.features.add_module('convlayer3', _ConvLayer(64, 128))
    # 10 classes
    self.classifier = nn.Linear(128, 10)
                
  def forward(self, x):
    features = self.features(x)
    # input_shape=(3,28,28)
    out = Fun.avg_pool2d(features, kernel_size=7, stride=1).view(features.size(0), -1)
    out = self.classifier(out)
    return out

## Training
def Train(epochs, Teacher, Student, trainloader):
  
  optimizer = optim.Adam(Student.parameters(), lr=0.001)
  criterion1 = nn.CrossEntropyLoss().to(device)
  criterion2 = nn.KLDivLoss()

  for epoch in range(epochs):
    print(epoch)
    Student.train()
    Teacher.train()
    alpha = 0.90
    train_acc = []
    train_loss = []
    #total = 0
    for i, data in enumerate(tqdm(trainloader)):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        soft_target = Teacher(inputs)
        optimizer.zero_grad()
        outputs = Student(inputs)

        loss1 = criterion1(outputs, labels)

        T = 10
        outputs_S = Fun.log_softmax(5*outputs/T, dim=1)
        outputs_T = Fun.softmax(5*soft_target/T, dim=1)
        loss2 = T*T*criterion2(outputs_S, outputs_T)

        if epoch < 10:
          loss = loss1*(1-alpha) + loss2*alpha
        else:
          loss = loss1

        loss.backward()
        optimizer.step()
        train_acc.append(correct)
        train_loss.append(loss.item())
        classifications = torch.argmax(outputs, dim = 1)
        correct = (classifications == labels).float().mean()
        #total += labels.size(0)
    # print(total)
    # print(train_loss)
    print(f"Accuracy: {100*sum(train_acc)/len(train_acc)} %")
    print(f"Loss: {sum(train_loss)/len(train_loss)}")

  return Student


def test(testloader, Student):
  # test 
  correct = 0
  total = 0
  pred_arr = []
  all_pred = []
  all_labels = []
  all_out = []
  with torch.no_grad():
    Student.eval()
    for data in tqdm(testloader):
          images1, labels = data
          images1, labels = images1.to(device), labels.to(device)
          outputs = Student(images1)

          all_out.append(outputs.data)
          all_labels.append(labels)
          _ , predicted = torch.max(outputs.data, 1)
          all_pred.append(predicted)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
          pred_arr.append(predicted.item())
  
  accuracy = 100 * correct / total
  return accuracy,pred_arr


if __name__ == "__main__":
  try:
    trainloader, testloader = load_data()
    # Teacher = ResNet().to(device)
    # checkpoint = torch.load("./resnet-50.pth")
    # Teacher.load_state_dict(checkpoint['model_state_dict'])

    Student = CNN().to(device)
    #Student = Train(10, Teacher, Student, trainloader)
    checkpoint_S = torch.load("./CNN.pth")
    Student.load_state_dict(checkpoint_S)
    print(summary(Student,(3,28,28)))
    
    #torch.save(Student.state_dict(), './CNN.pth')
    acc,pred_arr  = test(testloader, Student)
    print(acc,' %')
    pred_data = {"pred":pred_arr}
    df_pred = pd.DataFrame(pred_data)
    df_pred.to_csv('311706002_pred.csv', index_label='id')
  except:
    print('Error!')