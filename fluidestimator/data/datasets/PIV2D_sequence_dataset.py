import torch
import flowiz as fz
import numpy as np
from PIL import Image
import random


class PIV2DSequence(torch.utils.data.Dataset):
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

    def __len__(self):
        return len(self.data_index_list)

    def __getitem__(self, idx):
        # Handle idx which exceed the length limit
        if idx + self.sequence_length > self.total_length:
            # idx = random.randint(0, self.total_length - self.sequence_length)
            idx = (idx + self.sequence_length) % self.total_length
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

            sample_1 = np.asarray(Image.open(
                img1_path)) * 1.0 / 255.0
            sample_2 = np.asarray(Image.open(
                img2_path)) * 1.0 / 255.0

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
        data_dict['gt'] = None
        data_dict['name'] = name_list[0] + '-' + name_list[-1]
        data_dict['type'] = '_'.join(name_list[0].split('_')[:-1])
        data_dict['gt'] = label
        return data_dict


class PIV2DSequenceAllData(torch.utils.data.Dataset):
    def __init__(self,
                 data_index_list,
                 data,
                 img_buffer,
                 targets_index_list=None,
                 targets=None,
                 target_buffer=None,
                 transform=None,
                 train=True,
                 sequence_length=2):
        """
        The dataset read the images on the fly
        Args:
            data_index_list: shuffled index list for training image path
            data : all training images path
            img_buffer: all training images [number, channel, h, w]
            targets_index_list (string list): shuffled index list for test image path
            targets: all target image path
            target_buffer: all label images [number, channel, h, w]
            train: True train mode, false eval mode (print data path)
        """
        self.data_index_list = data_index_list
        self.data = data
        self.img_buffer = img_buffer
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

    def __len__(self):
        return len(self.data_index_list)

    def __getitem__(self, idx):
        # Handle idx which exceed the length limit
        if idx + self.sequence_length > self.total_length:
            # idx = random.randint(0, self.total_length - self.sequence_length)
            idx = (idx + self.sequence_length) % self.total_length
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

            curr_sample = torch.FloatTensor(np.array(self.img_buffer[idx]))
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
        data_dict['name'] = name_list[0] + '-' + name_list[-1]
        data_dict['type'] = '_'.join(name_list[0].split('_')[:-1])
        if label is None:
            label = torch.zeros_like(sample)
        data_dict['gt'] = label
        return data_dict


def read_img_to_buffer(img1_name_list, img2_name_list):
    sample_holder = [] 
    for img1_path, img2_path in zip(img1_name_list, img2_name_list):
        sample_1 = np.asarray(Image.open(
            img1_path)) * 1.0 / 255.0
        sample_2 = np.asarray(Image.open(
            img2_path)) * 1.0 / 255.0
        sample_holder.append(torch.FloatTensor(np.array([sample_1, sample_2])))
    return torch.stack(sample_holder, dim=0)


def read_label_to_buffer(img1_name_list, img2_name_list):
    sample_holder = [] 
    for img1_path, img2_path in zip(img1_name_list, img2_name_list):
        sample_1 = np.asarray(Image.open(
            img1_path)) * 1.0 / 255.0
        sample_2 = np.asarray(Image.open(
            img2_path)) * 1.0 / 255.0
        sample_holder.append(torch.FloatTensor(np.array([sample_1, sample_2])))
    return torch.stack(sample_holder, dim=0)
