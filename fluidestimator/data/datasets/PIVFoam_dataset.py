import torch
import random
import flowiz as fz
from torch.utils.data import TensorDataset
import numpy as np
from PIL import Image


class PIVFoam(torch.utils.data.Dataset):
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

    def _preprocess(self, im, target_size=256):
        """resize image to target_size x target_size"""
        w, h = im.size
        b_s = min(h, w)
        scale = b_s / target_size
        c_p = (int(h / 2), int(w / 2))  # centre point
        im = im.crop((
                 int(c_p[1] - b_s/2),  # left
                 int(c_p[0] - b_s/2),  # top
                 int(c_p[1] + b_s/2),  # right
                 int(c_p[0] + b_s/2),  # bottom
                 ))
        return im.resize((int(b_s / scale), int(b_s / scale)),
                         resample=Image.BILINEAR)

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

        im1 = self._preprocess(Image.open(img1_path).convert('L'))
        im2 = self._preprocess(Image.open(img2_path).convert('L'))
        sample_1 = np.asarray(im1) * 1.0 / 255.0
        sample_2 = np.asarray(im2) * 1.0 / 255.0

        if not self.train:
            print(' ')
            print('img1: ', img1_name)
            print('img2: ', img2_name)
            if self.targets is not None:
                print('label: ', self.targets[sample_ind])
            print(' ')

        sample = torch.FloatTensor([sample_1, sample_2])
        data_dict['image'] = sample
        data_dict['gt'] = torch.zeros_like(sample)
        data_dict['name'] = img1_name
        data_dict['type'] = 'foam_cylinder'
        # read_flow convert the RGB value in .flo to velocity components u and v
        if self.targets is not None:
            label = fz.read_flow(self.targets[label_ind])
            label = torch.FloatTensor([label[..., 0], label[..., 1]])
            data_dict['gt'] = label
        return data_dict


class PIVFoamSequence(torch.utils.data.Dataset):
    def __init__(self,
                 data_index_list,
                 data,
                 sequence_length=2,
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
        # Sequence length of each training sample
        self.sequence_length = sequence_length
        # Number of data samples
        self.total_length = len(self.data_index_list) - 1


    def eval(self):
        self.train = False

    def _preprocess(self, im, target_size=256):
        """resize image to target_size x target_size"""
        w, h = im.size
        b_s = min(h, w)
        scale = b_s / target_size
        c_p = (int(h / 2), int(w / 2))  # centre point
        im = im.crop((
                 int(c_p[1] - b_s/2),  # left
                 int(c_p[0] - b_s/2),  # top
                 int(c_p[1] + b_s/2),  # right
                 int(c_p[0] + b_s/2),  # bottom
                 ))
        return im.resize((int(b_s / scale), int(b_s / scale)),
                         resample=Image.BILINEAR)

    def __len__(self):
        return len(self.data_index_list)

    def __getitem__(self, idx):
        # Handle idx which exceed the length limit
        if idx + self.sequence_length > self.total_length:
            idx = random.randint(0, self.total_length - self.sequence_length)
        data_dict = {}
        sample_list = []
        label_list = []
        name_list = []
        label = None

        for n in range(self.sequence_length):
            sample_ind = self.data_index_list[idx] + n
            label_ind = 0
            if self.targets_index_list is not None:
                label_ind = self.targets_index_list[idx] + n
            img1_path = self.data[0][sample_ind]
            img2_path = self.data[1][sample_ind]
            img1_name = img1_path.split('/')[-1].replace('_img1.tif', '')
            img2_name = img2_path.split('/')[-1].replace('_img2.tif', '')
            name_list.append(img1_name)

            base_name = img1_path.split('/')[-1].split('_')[0]
            order_number = img1_path.split('/')[-1].split('_')[1].replace('.jpg', '')

            im1 = self._preprocess(Image.open(img1_path).convert('L'))
            im2 = self._preprocess(Image.open(img2_path).convert('L'))
            sample_1 = np.asarray(im1) * 1.0 / 255.0
            sample_2 = np.asarray(im2) * 1.0 / 255.0

            curr_sample = torch.FloatTensor(np.array([sample_1, sample_2]))
            sample_list.append(torch.unsqueeze(curr_sample, dim=0))

            if self.targets is not None:
                # read_flow convert the RGB value in .flo to velocity components u and v
                curr_label = fz.read_flow(self.targets[label_ind])
                curr_label = torch.FloatTensor(np.array([curr_label[..., 0], curr_label[..., 1]]))
                label_list.append(torch.unsqueeze(curr_label, dim=0))
        if self.targets is not None:
            label = torch.cat(label_list, dim=0)
        sample = torch.cat(sample_list, dim=0)
        data_dict['image'] = sample
        data_dict['gt'] = torch.unsqueeze(torch.sum(sample, dim=1), dim=1)
        data_dict['name'] = order_number
        data_dict['type'] = 'foam_cylinder'
        return data_dict