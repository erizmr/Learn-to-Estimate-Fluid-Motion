from .liteflownet import LiteFlowNet
from .pwcnet import PwcNet
from .corrector import Corrector


def build_model(cfg):
    if cfg.MODEL.PREDICTOR.NAME == 'liteflownet':
        return LiteFlowNet(cfg).to(cfg.DEVICE)
    elif cfg.MODEL.PREDICTOR.NAME == 'pwcnet':
        return PwcNet(cfg).to(cfg.DEVICE)
    elif cfg.MODEL.PREDICTOR.NAME == 'null':
        if cfg.MODEL.CORRECTOR.NAME == 'corrector':
            return Corrector(cfg).to(cfg.DEVICE)
    else:
        raise NotImplementedError
