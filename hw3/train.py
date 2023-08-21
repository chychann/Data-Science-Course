import pickle
import numpy as np
import torch
import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels,FusedNWaysKShots
from learn2learn.vision.transforms import RandomClassRotation
# from PIL.Image import LANCZOS
from collections import defaultdict
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from tqdm.auto import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

with open("./train.pkl", "rb") as f:
    train_data = pickle.load(f) # a dictionary
with open("./validation.pkl", "rb") as f:
    val_data = pickle.load(f) # a dictionary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

Nways = 5
Kshots = 5
img_size = 84
meta_lr=0.003
fast_lr=0.1
meta_batch_size=64
adaptation_steps=5
num_iterations=1000

# Dataset
class My_Dataset(Dataset):
    def __init__(self, imgs, labels, train = True):
        self.data = imgs
        self.labels = labels
        self.train = train
        
    def __getitem__(self, index):
        img = np.transpose(self.data[index], (1, 2, 0))
        img = Image.fromarray((img*255).astype('uint8'))
        tfms = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        return tfms(img), self.labels[index]
    def __len__(self):
        return len(self.data)

# acc function
def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)
# adapt
def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        # print(adaptation_data.shape)
        # print(adaptation_labels.shape)
        #print(learner_output)
        adaptation_error = loss(learner(adaptation_data), adaptation_labels)
        # print(adaptation_error.shape)
        learner.adapt(adaptation_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    evaluation_error = loss(predictions, evaluation_labels)
    evaluation_accuracy = accuracy(predictions, evaluation_labels)
    return evaluation_error, evaluation_accuracy

if __name__ == '__main__':

    save_path = './mini_test_maml_weights.pth'
    # Train dataset
    train_dataset = My_Dataset(train_data["images"], train_data["labels"])
    x = l2l.data.MetaDataset(train_dataset)
    transforms_train = [
        l2l.data.transforms.NWays(x, n=Nways),
        l2l.data.transforms.KShots(x, k=Kshots*2),
        l2l.data.transforms.LoadData(x),
        l2l.data.transforms.RemapLabels(x),
        l2l.data.transforms.ConsecutiveLabels(x),
    ]
    train_set = l2l.data.TaskDataset(x, transforms_train)
    # Validation dataset
    a = My_Dataset(val_data["images"], val_data["labels"])
    a = l2l.data.MetaDataset(a)
    transforms_val = [
        l2l.data.transforms.NWays(a, n=Nways),
        l2l.data.transforms.KShots(a, k=Kshots*2),
        l2l.data.transforms.LoadData(a),
        l2l.data.transforms.RemapLabels(a),
        l2l.data.transforms.ConsecutiveLabels(a),
    ]
    val_set = l2l.data.TaskDataset(a, transforms_val)
    # Load model
    model = l2l.vision.models.MiniImagenetCNN(output_size=Nways)
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr = fast_lr, first_order = True)
    # set optimizier and loss function
    opt = torch.optim.Adam(maml.parameters(),meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')
    # Count the acc
    best_accuracy = 0
    num_iterations = 1 #1000
    
    #Start to Train
    for iteration in range(num_iterations):
      model.train()
      opt.zero_grad()
      meta_train_error = 0.0
      meta_train_accuracy = 0.0
      meta_valid_error = 0.0
      meta_valid_accuracy = 0.0
      # model.train()
      for task in range(meta_batch_size):
          # Compute meta-training loss
          learner = maml.clone(first_order = True)
          batch = train_set.sample()
          evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                              learner,
                                                              loss,
                                                              adaptation_steps,
                                                              Nways,
                                                              Kshots,
                                                              device)
          evaluation_error.backward()
          meta_train_error += evaluation_error.item()
          meta_train_accuracy += evaluation_accuracy.item()

          # Compute meta-validation loss
          learner = maml.clone(first_order = True)
          batch = val_set.sample()
          evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                              learner,
                                                              loss,
                                                              adaptation_steps,
                                                              Nways,
                                                              Kshots,
                                                              device)
          meta_valid_error += evaluation_error.item()
          meta_valid_accuracy += evaluation_accuracy.item()
      if iteration < 10 or iteration%10 == 0:
          # Print some metrics
          print('\n')
          print('Iteration', iteration)
          print('Meta Train Loss:    ', meta_train_error / meta_batch_size)
          print('Meta Train Accuracy: ', meta_train_accuracy / meta_batch_size)
          print('Meta Valid Loss:    ', meta_valid_error / meta_batch_size)
          print('Meta Valid Accuracy: ', meta_valid_accuracy / meta_batch_size)
      if (meta_valid_accuracy / meta_batch_size) > best_accuracy:
          # save_path = '/content/drive/MyDrive/Colab Notebooks/HW3/mini_0423_maml_weights.pth'
          torch.save(maml.state_dict(), save_path)
          best_accuracy = meta_valid_accuracy / meta_batch_size
          print('save model!\n')
      # Average the accumulated gradients and optimize
      for p in maml.parameters():
          p.grad.data.mul_(1.0 / meta_batch_size)
      opt.step()