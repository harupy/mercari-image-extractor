import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
from config import IMAGE_DIR, TENSORBORAD_DIR
from utils import mkdir_if_not_exists


def get_data_loader(data_dir, transform):
  dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform)
  data_loader = data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
  return data_loader


def get_device():
  return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_vectors(img_dir):
  resnet = models.resnet50(pretrained=True)

  preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
  ])

  data_loader = get_data_loader(img_dir, preprocess)

  features = list(resnet.children())[:-1]
  model = nn.Sequential(*features)

  device = get_device()
  model.to(device)
  model.eval()

  vectors = []

  with torch.no_grad():
    for images, _ in tqdm(data_loader):
      images = images.to(device)
      vectors_batch = model(images).squeeze().cpu().numpy()
      vectors.extend(vectors_batch)

  return np.array(vectors)


def main():
  vectors = get_vectors(IMAGE_DIR)
  print('Writing to a tsv file...')
  mkdir_if_not_exists(TENSORBORAD_DIR)
  pd.DataFrame(vectors).to_csv(f'{TENSORBORAD_DIR}/vectors_all.tsv', index=False, sep='\t', header=False)


if __name__ == '__main__':
  main()
