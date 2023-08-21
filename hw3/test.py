import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import learn2learn as l2l
from PIL import Image
import pickle
from torchvision import transforms

Nways = 5
Kshots = 5
meta_lr = 0.003
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing on {device}")
adaptation_steps = 5


def load_maml(file_path, fast_lr):
  model = l2l.vision.models.MiniImagenetCNN(output_size=Nways)
  model.to(device)

  maml = l2l.algorithms.MAML(model, lr=fast_lr)

  checkpoint = torch.load(file_path, map_location=device)
  maml.load_state_dict(checkpoint)

  return maml

class TestDataset(Dataset):
    def __init__(self, imgs, labels = None):
        self.data = imgs
        if labels is not None:
            self.labels = labels
        else:
            self.labels = None
    def __getitem__(self, index):
        img = np.transpose(self.data[index], (1, 2, 0))
        img = Image.fromarray((img*255).astype('uint8'))
        tfms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        if self.labels is not None:
            return tfms(img), self.labels[index]
        else:
            return tfms(img)

    def __len__(self):
        return len(self.data)

def metasgd_pred(file_path, fast_lr, adapt_steps, output_path):
    model_test = load_maml(file_path, fast_lr)
    # sup_acc = 0.0
    # pred_arr = []
    result = []
    learner = model_test.clone()
    optimizer = torch.optim.Adam(model_test.parameters(), meta_lr)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    for (img, label), qry_img in zip(sup_loader, qry_loader):
        learner = model_test.clone(first_order=True)
        sup_img, sup_label = img.to(device), label.to(device)
        qry_img = qry_img.to(device)

        for step in range(adaptation_steps):
            support_output = learner(sup_img)
            support_error = criterion(support_output, sup_label)
            learner.adapt(support_error)
        pred = learner(qry_img)
        pred = pred.argmax(dim=1).view(sup_label.shape)
        result.extend(pred.tolist())
    result_data = {"Category": result}
    df_result = pd.DataFrame(result_data)
    df_result.to_csv(output_path, index_label = "Id")
    print(f"Saved file at {output_path}")

with open("./test.pkl", "rb") as f:
    test_data = pickle.load(f) # a dictionary

# make support dataset
sup_images = test_data["sup_images"]
sup_images = sup_images.reshape(600*25, 3, 84, 84)
sup_labels = test_data["sup_labels"]
sup_labels = sup_labels.reshape(600*25)
sup_dataset = TestDataset(sup_images, sup_labels) 
sup_loader = DataLoader(sup_dataset, batch_size= Nways*Kshots, shuffle=False)

# make query dataset
qry_images = test_data["qry_images"]
qry_images = qry_images.reshape(600*25, 3, 84, 84)
qry_dataset = TestDataset(qry_images)
qry_loader = DataLoader(qry_dataset, batch_size=Nways*Kshots, shuffle=False)

if __name__ == '__main__':
    
    file_path = './mini_0423_keep_maml_weights.pth'
    fast_lr = 1e-1
    adapt_steps = 5
    output_path = './pred_final.csv'
    metasgd_pred(file_path, fast_lr, adapt_steps, output_path)