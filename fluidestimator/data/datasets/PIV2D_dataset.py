import torch
import flowiz as fz
from torch.utils.data import TensorDataset
import numpy as np
from PIL import Image


class PIV2D(torch.utils.data.Dataset):
    def __init__(self,
                 data_index_list,
                 data,
                 targets_index_list=None,
                 targets=None,
                 transform=None,
                 train=True):
        """
        The dataset read the images on the fly
        Args:
            data_index_list: shuffled index list for training image path
            data (string list): A list of training image path 
            targets_index_list (string list): shuffled index list for test image path
            targets: A list of test image path
            train: True train mode, false eval mode (print data path)
        """
        self.data_index_list = data_index_list
        self.data = data
        self.targets_index_list = targets_index_list
        self.targets = targets
        self.count = 0
        self.train = train

    def eval(self):
        self.train = False

    def __len__(self):
        return len(self.data_index_list)

    def __getitem__(self, idx):
        sample_ind = self.data_index_list[idx]
        if self.targets_index_list is not None:
            label_ind = self.targets_index_list[idx]
        data_dict = {}
        img1_path = self.data[0][sample_ind]
        img2_path = self.data[1][sample_ind]
        img1_name = img1_path.split('/')[-1].replace('_img1.tif', '')
        img2_name = img2_path.split('/')[-1].replace('_img2.tif', '')

        sample_1 = np.asarray(Image.open(
            img1_path)) * 1.0 / 255.0
        sample_2 = np.asarray(Image.open(
            img2_path)) * 1.0 / 255.0

        if not self.train:
            print(' ')
            print('img1: ', img1_name)
            print('img2: ', img2_name)
            if self.targets is not None:
                print('label: ', self.targets[sample_ind])
            print(' ')

        sample = torch.FloatTensor(np.array([sample_1, sample_2]))
        data_dict['image'] = sample
        data_dict['gt'] = None
        data_dict['name'] = img1_name
        data_dict['type'] = '_'.join(img1_name.split('_')[:-1])
        # read_flow convert the RGB value in .flo to velocity components u and v
        if self.targets is not None:
            label = fz.read_flow(self.targets[label_ind])
            label = torch.FloatTensor(np.array([label[..., 0], label[..., 1]]))
            data_dict['gt'] = label
        return data_dict
