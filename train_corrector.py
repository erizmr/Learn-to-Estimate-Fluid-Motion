import os
import time
import torch
from fluidestimator.config import get_cfg
from fluidestimator.modeling import PwcNet as UnPwcNet
from fluidestimator.engine import DefaultTrainer, SimpleTrainer, default_argument_parser, default_setup, launch


class SequenceTrainer(SimpleTrainer):
    """
    A trainer to handle the sequence data
    """
    def __init__(self, model, data_loader, optimizer, predictor):
        super().__init__(model, data_loader, optimizer)
        self.predictor = predictor
    
    def run_step(self):
        assert self.model.training, "[SimpleTrainer] modeling was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        images = torch.stack([x["image"] for x in data], dim=0)
        data_time = time.perf_counter() - start

        seq_len = images.shape[1]
        prev = None
        corrected = None
        loss_dict = {}
        for num in range(seq_len):
            pred = self.predictor(images[:,num,...])
            if prev is not None:
                sub_loss_dict, corrected = self.model(prev, pred)
                for name, v in sub_loss_dict.items():
                    if name in loss_dict:
                        loss_dict[name] += v
                    else:
                        loss_dict[name] = v
            if corrected is None:
                prev = pred
            else:
                prev = corrected

        losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        losses.backward()

        self._write_metrics(loss_dict, data_time)
        self.optimizer.step()


class Trainer(DefaultTrainer):
    """
    Replace the SimpleTrainer to SequenceTrainer
    """
    def __init__(self, cfg, predictor):
        super().__init__(cfg)
        self._trainer = SequenceTrainer(self.model, self.data_loader, self.optimizer, predictor)


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


def main(args):
    cfg = setup(args)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Setup predictor
    un_pwcnet = UnPwcNet(cfg).to(cfg.DEVICE)
    un_pwcnet.load_state_dict(torch.load(cfg.MODEL.CORRECTOR.UPSTREAM_PREDICTOR)['model'])
    un_pwcnet.eval()

    trainer = Trainer(cfg, un_pwcnet)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    # Corrector training
    args = default_argument_parser().parse_args()
    args.resume = True
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

