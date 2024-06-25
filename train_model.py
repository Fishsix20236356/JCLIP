import jittor as jt
from PIL import Image
import jclip as clip
import os
from tqdm import tqdm
import argparse
from jittor import optim
from sklearn.linear_model import LogisticRegression
import numpy as np
from jittor.dataset import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, imgs_dir, features, labels, preprocess):
        super(MyDataset,self).__init__()
        self.imgs_dir = imgs_dir
        self.features = features
        self.labels = labels
        self.preprocess = preprocess

    def __getitem__(self, index):
        feature=self.features[index]
        label = self.labels[index]
        return feature, label

    def __len__(self):
        return len(self.features)

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='A')
args = parser.parse_args()

model, preprocess = clip.load("ViT-B-32.pkl")

classes = open('Dataset/classes.txt').read().splitlines()

# Process class names
new_classes = []
for c in classes:
    c = c.split(' ')[0]
    if c.startswith('Animal'):
        c = c[7:]
    if c.startswith('Thu-dog'):
        c = c[8:]
    if c.startswith('Caltech-101'):
        c = c[12:]
    if c.startswith('Food-101'):
        c = c[9:]
    c = 'a photo of ' + c
    new_classes.append(c)

text = clip.tokenize(new_classes)
text_features = model.encode_text(text)
text_features /= text_features.norm(dim=-1, keepdim=True)

# Load training data
imgs_dir = 'Dataset/'
train_labels = open('Dataset/train.txt').read().splitlines()
train_imgs = [l.split(' ')[0] for l in train_labels]
train_labels = [int(l.split(' ')[1]) for l in train_labels]

cnt = {}
new_train_imgs = []
new_train_labels = []
for i in range(len(train_imgs)):
    label = int(train_labels[i])
    if label not in cnt:
        cnt[label] = 0
    if cnt[label] < 4:
        new_train_imgs.append(train_imgs[i])
        new_train_labels.append(np.array(train_labels))
        cnt[label] += 1

# calculate image features of training data
train_features = []
print('Training data processing:')
with jt.no_grad():
    for img in tqdm(new_train_imgs):
        img = os.path.join(imgs_dir, img)
        image = Image.open(img)
        image = preprocess(image).unsqueeze(0)
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        train_features.append(image_features)

train_features = np.concatenate(train_features, axis=0)
train_labels = np.concatenate(new_train_labels, axis=0)

# Save train_features and train_labels to .npy files
np.save('train_features.npy', train_features)
np.save('train_labels.npy', train_labels)

