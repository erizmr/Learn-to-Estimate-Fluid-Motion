import os
import torch
import pickle
import torch.nn as nn
from fluidestimator.modeling import PwcNet as UnPwcNet
from fluidestimator.modeling import Corrector
from fluidestimator.data.datasets import read_by_type, construct_dataset
from fluidestimator.config import get_cfg
from collections import defaultdict
from fluidestimator.engine import default_argument_parser, default_setup\



device = 'cuda'

def prediction(predictor, corrector, test_loader):
    criterion = nn.MSELoss()
    ret_list = []
    with torch.no_grad():
        for _, out in enumerate(test_loader):
            ret_dict = defaultdict(list)
            seq_len = out['image'].shape[1]
            prev = None
            corrected = None
            for num in range(seq_len):
                pred = predictor(out['image'][:,num,...])
                if prev is not None:
                    corrected, _, _, _ = corrector(prev, pred)
                if corrected is None:
                    prev = pred
                else:
                    prev = corrected
            ret_dict['corrected'].append(corrected)
            ret_list.append(ret_dict)
    return ret_list

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    
    predictor_model = UnPwcNet(cfg).to(device)
    predictor_model.load_state_dict(torch.load(cfg.MODEL.CORRECTOR.UPSTREAM_PREDICTOR)['model'])
    predictor_model.eval()

    data_path = "./data"
    flow_type = 'DNS_turbulence_small'
    img1_name_dict, img2_name_dict, gt_name_dict = read_by_type(data_path)
    assert flow_type in img1_name_dict
    img1_name_list, img2_name_list, gt_name_list = img1_name_dict[flow_type], img2_name_dict[flow_type], gt_name_dict[flow_type]

    _, _, test_dataset_PIV2DSeq = construct_dataset(img1_name_list, 
                                                    img2_name_list, 
                                                    gt_name_list, 
                                                    cfg=cfg,
                                                    shuffle=False,
                                                    test_size=0.5,
                                                    dataset_name='PIV2DSequence')

    test_sampler = torch.utils.data.SequentialSampler(test_dataset_PIV2DSeq)
    test_loader_PIV2DSeq = torch.utils.data.DataLoader(dataset=test_dataset_PIV2DSeq,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1,
                                            sampler = test_sampler)
    
    corrector_model_name = f"pretrained_model/corrector.pth"
    corrector_model = Corrector(cfg).to(device)
    corrector_model.load_state_dict(torch.load(f"{corrector_model_name}")['model'])
    corrector_model.eval()
    ret_list = prediction(predictor_model,
                          corrector_model,
                          test_loader_PIV2DSeq)
    os.makedirs("demo_output", exist_ok=True)
    with open("demo_output/demo_result.pkl", "wb") as f:
        pickle.dump(ret_list, f)
    
