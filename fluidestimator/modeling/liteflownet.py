# This file is adapted from https://github.com/sniklaus/pytorch-liteflownet

import math
import torch
import torch.nn.functional as F

from fluidestimator.modeling import correlation  # the custom cost volume layer
from typing import Dict, List, Tuple
from torch import Tensor, nn
from .loss_functions import multiscale_unsupervised_error, device, multiscale_supervised_error

Backward_tensorGrid = {}

def Backward(tensorInput, tensorFlow):
    if str(tensorFlow.size()) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(
            1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1,
                                                tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(
            1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1,
                                                tensorFlow.size(3))

        Backward_tensorGrid[str(tensorFlow.size())] = torch.cat(
            [tensorHorizontal, tensorVertical], 1).to(device)
    # end
    tensorFlow = torch.cat([
        tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
        tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)
    ], 1)
    return torch.nn.functional.grid_sample(
        input=tensorInput,
        grid=(Backward_tensorGrid[str(tensorFlow.size())] +
              tensorFlow).permute(0, 2, 3, 1),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True)


class LiteFlowNet(torch.nn.Module):
    def __init__(self, cfg):
        super(LiteFlowNet, self).__init__()
        self.mean = 1.0
        self.std = 1.0
        self.device = cfg.DEVICE
        self.cfg = cfg

        class Features(torch.nn.Module):
            def __init__(self):
                super(Features, self).__init__()

                self.moduleOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=1,
                                    out_channels=32,
                                    kernel_size=7,
                                    stride=1,
                                    padding=3),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                self.moduleTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                self.moduleThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                self.moduleFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64,
                                    out_channels=96,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96,
                                    out_channels=96,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                self.moduleFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=96,
                                    out_channels=128,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                self.moduleSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=128,
                                    out_channels=192,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

            # end

            def forward(self, tensorInput):
                tensorOne = self.moduleOne(tensorInput)
                tensorTwo = self.moduleTwo(tensorOne)
                tensorThr = self.moduleThr(tensorTwo)
                tensorFou = self.moduleFou(tensorThr)
                tensorFiv = self.moduleFiv(tensorFou)
                tensorSix = self.moduleSix(tensorFiv)

                return [
                    tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv,
                    tensorSix
                ]

            # end

        # end

        class Matching(torch.nn.Module):
            def __init__(self, intLevel):
                super(Matching, self).__init__()

                self.dblBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25,
                                    0.625][intLevel]

                if intLevel != 2:
                    self.moduleFeat = torch.nn.Sequential()

                elif intLevel == 2:
                    self.moduleFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32,
                                        out_channels=64,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                # end

                if intLevel == 6:
                    self.moduleUpflow = None

                elif intLevel != 6:
                    self.moduleUpflow = torch.nn.ConvTranspose2d(
                        in_channels=2,
                        out_channels=2,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                        groups=2)

                # end

                if intLevel >= 4:
                    self.moduleUpcorr = None

                elif intLevel < 4:
                    self.moduleUpcorr = torch.nn.ConvTranspose2d(
                        in_channels=49,
                        out_channels=49,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                        groups=49)

                # end

                self.moduleMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=49,
                                    out_channels=128,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32,
                                    out_channels=2,
                                    kernel_size=[0, 0, 7, 5, 5, 3,
                                                 3][intLevel],
                                    stride=1,
                                    padding=[0, 0, 3, 2, 2, 1, 1][intLevel]))

            # end

            def forward(self, tensorFirst, tensorSecond, tensorFeaturesFirst,
                        tensorFeaturesSecond, tensorFlow):
                tensorFeaturesFirst = self.moduleFeat(tensorFeaturesFirst)
                tensorFeaturesSecond = self.moduleFeat(tensorFeaturesSecond)

                if tensorFlow is not None:
                    tensorFlow = self.moduleUpflow(tensorFlow)
                # end

                if tensorFlow is not None:
                    tensorFeaturesSecond = Backward(
                        tensorInput=tensorFeaturesSecond,
                        tensorFlow=tensorFlow * self.dblBackward)
                # end

                if self.moduleUpcorr is None:
                    tensorCorrelation = torch.nn.functional.leaky_relu(
                        input=correlation.FunctionCorrelation(
                            tensorFirst=tensorFeaturesFirst,
                            tensorSecond=tensorFeaturesSecond,
                            intStride=1),
                        negative_slope=0.1,
                        inplace=False)

                elif self.moduleUpcorr is not None:
                    tensorCorrelation = self.moduleUpcorr(
                        torch.nn.functional.leaky_relu(
                            input=correlation.FunctionCorrelation(
                                tensorFirst=tensorFeaturesFirst,
                                tensorSecond=tensorFeaturesSecond,
                                intStride=2),
                            negative_slope=0.1,
                            inplace=False))

                # end

                return (tensorFlow if tensorFlow is not None else
                        0.0) + self.moduleMain(tensorCorrelation)

            # end

        # end

        class Subpixel(torch.nn.Module):
            def __init__(self, intLevel):
                super(Subpixel, self).__init__()

                self.dblBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25,
                                    0.625][intLevel]

                if intLevel != 2:
                    self.moduleFeat = torch.nn.Sequential()

                elif intLevel == 2:
                    self.moduleFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32,
                                        out_channels=64,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                # end

                self.moduleMain = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=[0, 0, 130, 130, 194, 258, 386][intLevel],
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32,
                                    out_channels=2,
                                    kernel_size=[0, 0, 7, 5, 5, 3,
                                                 3][intLevel],
                                    stride=1,
                                    padding=[0, 0, 3, 2, 2, 1, 1][intLevel]))

            # end

            def forward(self, tensorFirst, tensorSecond, tensorFeaturesFirst,
                        tensorFeaturesSecond, tensorFlow):
                tensorFeaturesFirst = self.moduleFeat(tensorFeaturesFirst)
                tensorFeaturesSecond = self.moduleFeat(tensorFeaturesSecond)

                if tensorFlow is not None:
                    tensorFeaturesSecond = Backward(
                        tensorInput=tensorFeaturesSecond,
                        tensorFlow=tensorFlow * self.dblBackward)
                # end

                return (tensorFlow
                        if tensorFlow is not None else 0.0) + self.moduleMain(
                            torch.cat([
                                tensorFeaturesFirst, tensorFeaturesSecond,
                                tensorFlow
                            ], 1))

            # end

        # end

        class Regularization(torch.nn.Module):
            def __init__(self, intLevel):
                super(Regularization, self).__init__()

                self.dblBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25,
                                    0.625][intLevel]

                self.intUnfold = [0, 0, 7, 5, 5, 3, 3][intLevel]

                if intLevel >= 5:
                    self.moduleFeat = torch.nn.Sequential()

                elif intLevel < 5:
                    self.moduleFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(
                            in_channels=[0, 0, 32, 64, 96, 128, 192][intLevel],
                            out_channels=128,
                            kernel_size=1,
                            stride=1,
                            padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                # end

                self.moduleMain = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=[0, 0, 131, 131, 131, 131, 195][intLevel],
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128,
                                    out_channels=128,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                if intLevel >= 5:
                    self.moduleDist = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32,
                                        out_channels=[0, 0, 49, 25, 25, 9,
                                                      9][intLevel],
                                        kernel_size=[0, 0, 7, 5, 5, 3,
                                                     3][intLevel],
                                        stride=1,
                                        padding=[0, 0, 3, 2, 2, 1,
                                                 1][intLevel]))

                elif intLevel < 5:
                    self.moduleDist = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32,
                                        out_channels=[0, 0, 49, 25, 25, 9,
                                                      9][intLevel],
                                        kernel_size=([0, 0, 7, 5, 5, 3,
                                                      3][intLevel], 1),
                                        stride=1,
                                        padding=([0, 0, 3, 2, 2, 1,
                                                  1][intLevel], 0)),
                        torch.nn.Conv2d(in_channels=[0, 0, 49, 25, 25, 9,
                                                     9][intLevel],
                                        out_channels=[0, 0, 49, 25, 25, 9,
                                                      9][intLevel],
                                        kernel_size=(1, [0, 0, 7, 5, 5, 3,
                                                         3][intLevel]),
                                        stride=1,
                                        padding=(0, [0, 0, 3, 2, 2, 1,
                                                     1][intLevel])))

                # end

                self.moduleScaleX = torch.nn.Conv2d(
                    in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel],
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0)
                self.moduleScaleY = torch.nn.Conv2d(
                    in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel],
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0)

            # eny

            def forward(self, tensorFirst, tensorSecond, tensorFeaturesFirst,
                        tensorFeaturesSecond, tensorFlow):
                tensorDifference = (tensorFirst - Backward(
                    tensorInput=tensorSecond,
                    tensorFlow=tensorFlow * self.dblBackward)).pow(2.0).sum(
                        1, True).sqrt().detach()

                tensorDist = self.moduleDist(
                    self.moduleMain(
                        torch.cat([
                            tensorDifference, tensorFlow -
                            tensorFlow.view(tensorFlow.size(0), 2, -1).mean(
                                2, True).view(tensorFlow.size(0), 2, 1, 1),
                            self.moduleFeat(tensorFeaturesFirst)
                        ], 1)))
                tensorDist = tensorDist.pow(2.0).neg()
                tensorDist = (tensorDist - tensorDist.max(1, True)[0]).exp()

                tensorDivisor = tensorDist.sum(1, True).reciprocal()

                tensorScaleX = self.moduleScaleX(
                    tensorDist * torch.nn.functional.unfold(
                        input=tensorFlow[:, 0:1, :, :],
                        kernel_size=self.intUnfold,
                        stride=1,
                        padding=int((self.intUnfold - 1) /
                                    2)).view_as(tensorDist)) * tensorDivisor
                tensorScaleY = self.moduleScaleY(
                    tensorDist * torch.nn.functional.unfold(
                        input=tensorFlow[:, 1:2, :, :],
                        kernel_size=self.intUnfold,
                        stride=1,
                        padding=int((self.intUnfold - 1) /
                                    2)).view_as(tensorDist)) * tensorDivisor

                return torch.cat([tensorScaleX, tensorScaleY], 1)

            # end

        # end

        self.moduleFeatures = Features()
        self.moduleMatching = torch.nn.ModuleList(
            [Matching(intLevel) for intLevel in [2, 3, 4, 5, 6]])
        self.moduleSubpixel = torch.nn.ModuleList(
            [Subpixel(intLevel) for intLevel in [2, 3, 4, 5, 6]])
        self.moduleRegularization = torch.nn.ModuleList(
            [Regularization(intLevel) for intLevel in [2, 3, 4, 5, 6]])

    # end

    def one_pass(self, first, second):
        tensorFeaturesFirst = self.moduleFeatures(first)
        tensorFeaturesSecond = self.moduleFeatures(second)

        tensorFirst = [first]
        tensorSecond = [second]

        for intLevel in [1, 2, 3, 4, 5]:
            tensorFirst.append(
                torch.nn.functional.interpolate(
                    input=tensorFirst[-1],
                    size=(tensorFeaturesFirst[intLevel].size(2),
                          tensorFeaturesFirst[intLevel].size(3)),
                    mode='bilinear',
                    align_corners=False))
            tensorSecond.append(
                torch.nn.functional.interpolate(
                    input=tensorSecond[-1],
                    size=(tensorFeaturesSecond[intLevel].size(2),
                          tensorFeaturesSecond[intLevel].size(3)),
                    mode='bilinear',
                    align_corners=False))
        # end

        tensorFlow = None
        flow_collection = []

        for intLevel in [-1, -2, -3, -4, -5]:
            tensorFlow = self.moduleMatching[intLevel](
                tensorFirst[intLevel], tensorSecond[intLevel],
                tensorFeaturesFirst[intLevel], tensorFeaturesSecond[intLevel],
                tensorFlow)
            tensorFlow = self.moduleSubpixel[intLevel](
                tensorFirst[intLevel], tensorSecond[intLevel],
                tensorFeaturesFirst[intLevel], tensorFeaturesSecond[intLevel],
                tensorFlow)
            tensorFlow = self.moduleRegularization[intLevel](
                tensorFirst[intLevel], tensorSecond[intLevel],
                tensorFeaturesFirst[intLevel], tensorFeaturesSecond[intLevel],
                tensorFlow)
            flow_collection.append(tensorFlow)

        flow_collection[-1] *= 4.0  # Final flow scale 20.0, others 5.0
        flow_collection = [flow * 5.0 for flow in flow_collection]
        return flow_collection

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        Spliting and batch the input images.
        """
        if not torch.is_tensor(batched_inputs):
            images = torch.stack([x["image"] for x in batched_inputs], dim=0)
            label = torch.stack([x["gt"] for x in batched_inputs], dim=0)
        else:
            images = batched_inputs
            # TODO: can not handle label data
            label = None

        img_1, img_2 = torch.split(images, split_size_or_sections=1, dim=1)

        assert (img_1.size(3) == img_2.size(3))
        assert (img_1.size(2) == img_2.size(2))

        int_width = img_1.size(3)
        int_height = img_1.size(2)
        img_1 = img_1.view(-1, 1, int_height, int_width)
        img_2 = img_2.view(-1, 1, int_height, int_width)

        int_preprocessed_width = int(math.floor(math.ceil(int_width / 32.0) * 32.0))
        int_preprocessed_height = int(math.floor(math.ceil(int_height / 32.0) * 32.0))

        first = torch.nn.functional.interpolate(
            input=img_1,
            size=(int_preprocessed_height, int_preprocessed_width),
            mode='bilinear',
            align_corners=False)
        second = torch.nn.functional.interpolate(
            input=img_2,
            size=(int_preprocessed_height, int_preprocessed_width),
            mode='bilinear',
            align_corners=False)
        return first, second, label, int_preprocessed_height, int_preprocessed_width

    def postprossing_flow(self, flow_collection, raw_shape, processed_shape):
        h, w = raw_shape
        p_h, p_w = processed_shape
        flow_full_reso = F.interpolate(input=flow_collection[-1],
                                                     size=(h, w),
                                                     mode='bilinear',
                                                     align_corners=False)

        flow_full_reso[:, 0, :, :] *= float(w) / float(p_w)
        flow_full_reso[:, 1, :, :] *= float(h) / float(p_h)
        return flow_full_reso

    def forward(self, batched_inputs: Tuple[Dict[str, Tensor]]):

        first, second, label, p_h, p_w = self.preprocess_image(batched_inputs)
        first[:, 0, :, :] = first[:, 0, :, :] - 0.411618
        second[:, 0, :, :] = second[:, 0, :, :] - 0.410782
        first = first.to(self.device)
        second = second.to(self.device)
        bs, c, h, w = first.shape
        forward_flow_collection = self.one_pass(first, second)
        # Bilinear interpolation & recover from padding for full resolution
        flow_full_reso = self.postprossing_flow(flow_collection=forward_flow_collection,
                                                raw_shape=(h, w),
                                                processed_shape=(p_h, p_w))
        forward_flow_collection.append(flow_full_reso)

        if self.training:
            if self.cfg.TRAIN.SUPERVISE:
                return multiscale_supervised_error(forward_flow_collection, label.to(self.device), cfg=self.cfg)
            backward_flow_collection = self.one_pass(second, first)
            flow_full_reso = self.postprossing_flow(flow_collection=backward_flow_collection,
                                                    raw_shape=(h, w),
                                                    processed_shape=(p_h, p_w))
            backward_flow_collection.append(flow_full_reso)

            return multiscale_unsupervised_error(forward_flow_collection,
                                                 backward_flow_collection,
                                                 None,
                                                 None,
                                                 first,
                                                 second,
                                                 cfg=self.cfg)

        else:
            return flow_full_reso

    # end


# end
