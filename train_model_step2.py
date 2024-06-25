import jittor as jt
import jclip as clip
from tqdm import tqdm
import argparse
from jittor import optim
from sklearn.linear_model import LogisticRegression
import numpy as np
from jittor.dataset import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, imgs_dir, features, labels, preprocess):
        super(MyDataset, self).__init__()
        self.imgs_dir = imgs_dir
        self.features = features
        self.labels = labels
        self.preprocess = preprocess

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        return feature, label

    def __len__(self):
        return len(self.features)

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='A')
args = parser.parse_args()

model, preprocess = clip.load("ViT-B-32.pkl")

imgs_dir = 'Dataset/'

train_features = np.load('train_features.npy')
train_labels = np.load('train_labels.npy')

train_features = jt.array(train_features)
train_labels = jt.array(train_labels)

dataset = MyDataset(imgs_dir, train_features, train_labels, preprocess)
train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 检查数据集大小
print("Total number of samples in dataset:", len(dataset))

# 检查数据迭代器
for i, batch in enumerate(train_dataloader):
    if i == 0:
        print("First batch size:", len(batch))
    print("Batch {}: features shape: {}, labels shape: {}".format(i, batch[0].shape, batch[1].shape))
    if i > 10:  # 只检查前 10 个批次
        break

# Define loss function and optimizer
loss_img = jt.nn.CrossEntropyLoss()
loss_txt = jt.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    print("-------第 {} 轮训练开始-------".format(epoch + 1))
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for batch in train_dataloader:
        features, labels = batch
        features = jt.array(features)
        labels = jt.array(labels)

        result_loss = loss_img(features, labels)
        optimizer.step(result_loss)

        pbar.set_description("loss: {:.4f}".format(result_loss.item()))

jt.save(model.state_dict(), 'fine_tuned_clip.pkl')
