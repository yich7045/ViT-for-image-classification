import torch
from glob import glob
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from skimage import io
from ViT_encoder import VisionTransformer
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
import csv
# prepare data
class Train_Datasets(Dataset):
    def __init__(self, csv_file):
        self.annotations =pd.read_csv(csv_file)
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize((1000, 1000)),
                                             transforms.ToTensor()])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = glob('trainval-20211121T195848Z-*/trainval/' + self.annotations.iloc[index, 0]+ '*_image.jpg')
        image = io.imread(img_path[0])
        image = self.transform(image)
        y_label = torch.zeros(3,)
        y_label[int(self.annotations.iloc[index, 1])] = 1.
        return image, y_label

class Test_Datasets(Dataset):
    def __init__(self, csv_file):
        self.annotations =pd.read_csv(csv_file)
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize((1000, 1000)),
                                             transforms.ToTensor()])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = glob('test-20211123T043833Z-*/test/' + self.annotations.iloc[index, 0]+ '*_image.jpg')
        image = io.imread(img_path[0])
        image = self.transform(image)
        position = self.annotations.iloc[index, 1]
        return image, position

len_data = len(pd.read_csv('labels.csv'))
train_set = Train_Datasets(csv_file='labels.csv')
len_test_data = len(pd.read_csv('submission.csv'))
test_set = Test_Datasets(csv_file='submission.csv')

# train_set, validation_set = torch.utils.data.random_split(train_set,
#                                                         [int(0.01*len_data), len_data-int(0.01*len_data)])

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

# model
device = 'cuda'
ViT = VisionTransformer().to(device)
lr = 5e-6
# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(ViT.parameters(), lr=lr)
# scheduler
epochs = 100

def one_hot_ce_loss(outputs, targets):
    criterion = nn.BCELoss()
    return criterion(outputs, targets)

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    for data, label in tqdm(train_loader, total=len(train_loader)):
        data = data.to(device)
        label = label.to(device)
        output = ViT(data)
        loss = one_hot_ce_loss(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (output.argmax(dim=1) == label.argmax(dim=1)).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)
    print(str(epoch)+': train_accuracy: ' + str(epoch_accuracy))

    with torch.no_grad():
        name = str(epoch) + '_result.csv'
        with open(name, 'w') as f:
            writer = csv.writer(f, delimiter=',', lineterminator='\n')
            for data, position in test_loader:
                data = data.to(device)
                position = position.to(device)
                test_output = ViT(data)
                test_label = test_output.argmax(dim=1)
                for i in range(len(position)):
                    writer.writerow([test_label[i].cpu().numpy(), position[i].cpu().numpy()])


