from glob import glob
from tqdm import tqdm
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class ImagevDatasetForClassi(Dataset):
    def __init__(self, mode='train', transform=None, known_class=['Healthy', 'Drift', 'Mssing_by_blank', 'Outlier', 'Periodic'], unknown_class=['Randomly_Mssing', 'Square'], width=224, height=224):
        self.mode = mode
        self.transform = transform
        self.width = width
        self.height = height

        path = '../db/data_source/'
        if mode == 'train':
            path += 'train/'
        else:
            path += 'val/'

        class_num = 0
        self.x_data, self.y_data = [], []
        for known in tqdm(known_class):
            imgfiles = sorted(glob(path + known + '/*.jpg'))
            for imgfile in imgfiles:
                self.x_data.append(np.array(Image.open(imgfile)))
                self.y_data.append(class_num)

            class_num += 1

        '''
        if mode == 'val':
            for unknown in tqdm(unknown_class):
                imgfiles = sorted(glob(path + unknown + '/*.jpg'))
                for imgfile in imgfiles:
                    self.x_data.append(np.array(Image.open(imgfile)))
                    self.y_data.append(class_num)

                #class_num += 1
        '''

        self.x_data = np.array(self.x_data)
        self.y_data = np.array(self.y_data)
        print(self.x_data.shape, self.y_data.shape)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        sample = torch.as_tensor(self.x_data[idx]).view(3, self.height, self.width), torch.as_tensor(self.y_data[idx])
        if self.transform:
            sample = self.transform(sample)

        return sample



if __name__ == "__main__":
    train_dataset = ImagevDatasetForClassi(mode='train')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    for batch_idx, samples in enumerate(train_loader):
        data, target = samples
        print(data.shape, target.shape)
        print(torch.max(data), torch.min(data))
        print(torch.max(target), torch.min(target))

        break