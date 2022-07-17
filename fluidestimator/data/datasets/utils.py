import glob
import json
import logging
from tkinter.messagebox import NO
import numpy as np
from sklearn.model_selection import ShuffleSplit
from .PIVFoam_dataset import PIVFoam
from .PIV2D_dataset import PIV2D
from .PIV2D_sequence_dataset import PIV2DSequence, PIV2DSequenceAllData, read_img_to_buffer


def read_all(data_path):
    # Read the whole dataset
    try:
        img1_name_list = json.load(
            open(data_path + "/img1_name_list.json", 'r'))
        img2_name_list = json.load(
            open(data_path + "/img2_name_list.json", 'r'))
        gt_name_list = []
        try:
            gt_name_list = json.load(open(data_path + "/gt_name_list.json", 'r'))
        except:
            pass
    except:
        data_dir = glob.glob(data_path + "/*")
        print(data_dir)
        gt_name_list = []
        img1_name_list = []
        img2_name_list = []

        for dir in data_dir:
            try:
                gt_name_list.extend(glob.glob(dir + '/*flow.flo'))
            except:
                print('{}: No ground truth file'.format(dir))
            img1_name_list.extend(glob.glob(dir + '/*img1.*'))
            img2_name_list.extend(glob.glob(dir + '/*img2.*'))
        gt_name_list.sort()
        img1_name_list.sort()
        img2_name_list.sort()
        print(gt_name_list[0], img1_name_list[0], img2_name_list[0])
        print(len(gt_name_list), len(img1_name_list), len(img2_name_list))
        assert (len(gt_name_list) == len(img1_name_list))
        assert (len(img2_name_list) == len(img1_name_list))

        # Serialize data into file:
        json.dump(img1_name_list, open(data_path + "/img1_name_list.json",
                                       'w'))
        json.dump(img2_name_list, open(data_path + "/img2_name_list.json",
                                       'w'))
        json.dump(gt_name_list, open(data_path + "/gt_name_list.json", 'w'))
    return img1_name_list, img2_name_list, gt_name_list


def read_by_type(data_path):
    # Read the data by flow type
    data_dir = glob.glob(data_path + "/*[!json]")
    flow_img1_name_list = []
    flow_img2_name_list = []
    flow_gt_name_list = []

    try:
        flow_dir = [dir.split('/')[-1] for dir in data_dir]
        for f_dir in flow_dir:
            flow_img1_name_list.append(
                json.load(
                    open(data_path + "/" + f_dir + "_img1_name_list.json",
                         'r')))
            flow_img2_name_list.append(
                json.load(
                    open(data_path + "/" + f_dir + "_img2_name_list.json",
                         'r')))
            flow_gt_name_list.append(
                json.load(
                    open(data_path + "/" + f_dir + "_gt_name_list.json", 'r')))
    except:
        flow_dir = []
        flow_img1_name_list = []
        flow_img2_name_list = []
        flow_gt_name_list = []
        for dir in data_dir:
            # Initialize for different flow type
            sub_flow_img1_name_list = []
            sub_flow_img2_name_list = []
            sub_flow_gt_name_list = []
            sub_flow_gt_name_list.extend(glob.glob(dir + '/*flow.flo'))
            sub_flow_img1_name_list.extend(glob.glob(dir + '/*img1.tif'))
            sub_flow_img2_name_list.extend(glob.glob(dir + '/*img2.tif'))
            assert (len(sub_flow_gt_name_list) == len(sub_flow_img1_name_list))
            assert (
                    len(sub_flow_img2_name_list) == len(sub_flow_img1_name_list))
            sub_flow_gt_name_list.sort()
            sub_flow_img1_name_list.sort()
            sub_flow_img2_name_list.sort()

            # Serialize data into file:
            json.dump(
                sub_flow_img1_name_list,
                open(
                    data_path + "/" + dir.split('/')[-1] +
                    "_img1_name_list.json", 'w'))
            json.dump(
                sub_flow_img2_name_list,
                open(
                    data_path + "/" + dir.split('/')[-1] +
                    "_img2_name_list.json", 'w'))
            json.dump(
                sub_flow_gt_name_list,
                open(
                    data_path + "/" + dir.split('/')[-1] +
                    "_gt_name_list.json", 'w'))

            # Add to the total list
            flow_dir.append(dir.split('/')[-1])
            flow_img1_name_list.append(sub_flow_img1_name_list)
            flow_img2_name_list.append(sub_flow_img2_name_list)
            flow_gt_name_list.append(sub_flow_gt_name_list)
    img1_dict = dict()
    img2_dict = dict()
    gt_dict = dict()

    for i, name in enumerate(flow_dir):
        img1_dict[name] = flow_img1_name_list[i]
        img2_dict[name] = flow_img2_name_list[i]
        gt_dict[name] = flow_gt_name_list[i]
    return img1_dict, img2_dict, gt_dict


def construct_dataset(img1_name_list,
                      img2_name_list,
                      gt_name_list,
                      *,
                      shuffle=True,
                      cfg=None,
                      ratio=1.0,
                      test_size=0.1,
                      validate_size=0.1,
                      seed=42,
                      dataset_name='PIV2D'):
    """Construct dataset
    Args:
        img1_name_list: path list of the image1 in the pair
        img2_name_list: path list of the image2 in the pair
        gt_name_list: path list of the ground truth field
        ratio: Use how much of the data
        test_size: portion of test data (default 0.1)
    """
    logger = logging.getLogger(__name__)
    assert len(img1_name_list) == len(img2_name_list)

    amount = len(img1_name_list)
    total_data_index = np.arange(0, amount, 1)
    total_label_index = np.arange(0, amount, 1)

    train_data = [0]
    validate_data = [0]

    if test_size == 1.0:
        # No train, all test
        test_data = total_data_index
    else:
        if shuffle:
            # Divide train/validation and test data ( Default: 1:9)
            shuffler = ShuffleSplit(n_splits=1, test_size=test_size,
                                    random_state=seed).split(total_data_index,
                                                            total_label_index)
            indices = [(train_idx, test_idx) for train_idx, test_idx in shuffler][0]
            # Divide train and validation data ( Default: 1:9)
            shuffler_tv = ShuffleSplit(n_splits=1, test_size=validate_size,
                                    random_state=seed).split(indices[0], indices[0])
            indices_tv = [(train_idx, validation_idx)
                        for train_idx, validation_idx in shuffler_tv][0]

            train_data = indices_tv[0][:int(ratio * len(indices_tv[0]))]
            validate_data = indices_tv[1][:int(ratio * len(indices_tv[1]))]
            test_data = indices[1][:int(ratio * len(indices[1]))]
        else:
            total_num = len(total_data_index)
            chop_point = int(total_num*(1-test_size))
            train_data = total_data_index[0:chop_point]
            validate_data = []
            test_data = total_data_index[chop_point:]
    logger.info("Check training data: {}".format(len(train_data)))
    logger.info("Check validate data: {}".format(len(validate_data)))
    logger.info("Check test data: {}".format(len(test_data)))

    if dataset_name == 'PIV2D':
        train_dataset = PIV2D(train_data, [img1_name_list, img2_name_list],
                              targets_index_list=train_data,
                              targets=gt_name_list)
        validate_dataset = PIV2D(validate_data,
                                 [img1_name_list, img2_name_list],
                                 validate_data, gt_name_list)
        test_dataset = PIV2D(test_data, [img1_name_list, img2_name_list],
                             test_data, gt_name_list)
    elif dataset_name == 'PIVFoam':
        gt_name_list = None
        train_dataset = PIVFoam(train_data, [img1_name_list, img2_name_list],
                                targets_index_list=train_data,
                                targets=gt_name_list)
        validate_dataset = PIVFoam(validate_data,
                                   [img1_name_list, img2_name_list],
                                   validate_data, gt_name_list)
        test_dataset = PIVFoam(test_data, [img1_name_list, img2_name_list],
                               test_data, gt_name_list)
    elif dataset_name == 'PIV2DSequence':
        seq_len = 10
        if cfg:
            seq_len = cfg.DATASETS.SEQUENCE_LENGTH
        train_dataset = PIV2DSequence(train_data, [img1_name_list, img2_name_list],
                                      sequence_length=seq_len,
                                      targets_index_list=train_data,
                                      targets=gt_name_list)
        validate_dataset = PIV2DSequence(validate_data,
                                         [img1_name_list, img2_name_list], 
                                         sequence_length=seq_len, 
                                         targets_index_list=validate_data, 
                                         targets=gt_name_list)
        test_dataset = PIV2DSequence(test_data, 
                                     [img1_name_list, img2_name_list],
                                     sequence_length=seq_len, 
                                     targets_index_list=test_data, 
                                     targets=gt_name_list)
    elif dataset_name == 'PIV2DSequenceAllData':
        seq_len = 10
        img_buffer = read_img_to_buffer(img1_name_list, img2_name_list)
        train_dataset = PIV2DSequenceAllData(train_data, 
                                             [img1_name_list, img2_name_list],
                                             img_buffer=img_buffer,
                                             sequence_length=seq_len,
                                             targets_index_list=train_data,
                                             targets=None)
        validate_dataset = None
        test_dataset = PIV2DSequenceAllData(test_data, 
                                     [img1_name_list, img2_name_list],
                                     img_buffer=img_buffer,
                                     sequence_length=seq_len,
                                     targets_index_list=test_data, 
                                     targets=gt_name_list)
    else:
        raise NotImplementedError

    return train_dataset, validate_dataset, test_dataset


dataset_dict = {}


def fill_dataset_dict(cfg):
    # TODO: support only one dataset for train and test
    im1, im2, gt = read_all(cfg.DATA_DIR)

    start_index = []
    end_index = []
    im1_filtered = []
    im2_filtered = []
    gt_filtered = []
    for flow_type in cfg.DATASETS.FLOW_TYPE:
        if flow_type == "All":
            im1_filtered = im1
            im2_filtered = im2
            gt_filtered = gt
            break
        else:
            start_index.append(flow_types[flow_type][0])
            end_index.append(flow_types[flow_type][1])
    for s, e in zip(start_index, end_index):
        im1_filtered.extend(im1[s:e + 1])
        im2_filtered.extend(im2[s:e + 1])
        gt_filtered.extend(gt[s:e + 1])
    total_num = len(im1_filtered)
    if cfg.DATASETS.TOTAL_NUM != -1:
        total_num = cfg.DATASETS.TOTAL_NUM

    train_ds, validate_ds, test_ds = construct_dataset(im1_filtered[:total_num], im2_filtered[:total_num], gt_filtered[:total_num], shuffle=cfg.DATASETS.SHUFFLE,
                                                       test_size=cfg.DATASETS.TEST_RATIO,
                                                       dataset_name=cfg.DATASETS.TRAIN[0],
                                                       cfg=cfg)
    dataset_dict['train'] = train_ds
    dataset_dict['validate'] = validate_ds
    dataset_dict['test'] = test_ds


# Start and end index of different flow types
flow_types = {'DNS_turbulence': [0, 1999],
             'JHTDB_channel': [2000, 3899],
             'JHTDB_channel_hd': [3900, 4499],
             'JHTDB_isotropic1024_hd': [4500, 6499],
             'JHTDB_mhd1024_hd': [6500, 7299],
             'SQG': [7300, 8799],
             'backstep_Re1000': [8800, 9399],
             'backstep_Re1200': [9400, 10399],
             'backstep_Re1500': [10400, 11399],
             'backstep_Re800': [11400, 11999],
             'cylinder_Re150': [12000, 12499],
             'cylinder_Re200': [12500, 12999],
             'cylinder_Re300': [13000, 13499],
             'cylinder_Re40': [14000, 14049],
             'cylinder_Re400': [13500, 13999],
             'uniform': [14050, 15049]}
