import logging
import torch
import json
import copy
import itertools
import os
import numpy as np
from .evaluator import DatasetEvaluator
from collections import OrderedDict

import fluidestimator.utils.comm as comm
from fluidestimator.config import CfgNode
from fluidestimator.modeling.loss_functions import error_rate, epe_loss
from fluidestimator.utils.file_io import PathManager


class PIV2DEvaluator(DatasetEvaluator):
    """
    PIV2D dataset
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:
                    "json_file": the path to the COCO format annotation
                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of flow type in PIV2D dataset
                such as 'JHTDB_isotropic1024_hd', 'DNS_turbulence', 'backstep', 'SQG' etc.
                By default, will infer this automatically from predictions
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:
                1. "instances_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
        """
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input['name'], "type": input['type'],
                          "flow": output.to(self._cpu_device).numpy(), "gt": input['gt'].to(self._cpu_device).numpy()}

            self._predictions.append(prediction)

    def evaluate(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[PIV2DEvaluator] Did not receive valid predictions.")
            return {}

        # Output file too large
        # if self._output_dir:
        #     PathManager.mkdirs(self._output_dir)
        #     file_path = os.path.join(self._output_dir, "flow_predictions.pth")
        #     with PathManager.open(file_path, "wb") as f:
        #         torch.save(predictions, f)

        self._results = OrderedDict()
        if "flow" in predictions[0]:
            self._eval_predictions(predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, predictions, img_ids=None):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for PIV2D format ...")
        flow_results = list(itertools.chain(*[x["flow"] for x in predictions]))

        # Output file too large
        # if self._output_dir:
        #     file_path = os.path.join(self._output_dir, "PIV2D_results.json")
        #     self._logger.info("Saving results to {}".format(file_path))
        #     with PathManager.open(file_path, "w") as f:
        #         f.write(json.dumps(flow_results))
        #         f.flush()

        self._logger.info(
            "Evaluating predictions with PIV2D ground truth..."
        )
        epe_all = []
        er_all = []
        for output in predictions:
            flow_gt = output['gt']
            flow = output['flow']
            f_type = output['type']
            f_u, f_v = np.split(flow, indices_or_sections=2, axis=1)
            f_ut, f_vt = np.split(flow_gt, indices_or_sections=2, axis=1)
            er = error_rate(f_u, f_v, f_ut, f_vt).astype(float)
            epe = epe_loss(flow, flow_gt).astype(float)
            epe_all.append(epe)
            er_all.append(er)
            epe_ftype, er_ftype = '_'.join(['epe', f_type]), '_'.join(['er', f_type])
            if epe_ftype not in self._results:
                self._results[epe_ftype] = [epe]
                self._results[er_ftype] = [er]
            else:
                self._results[epe_ftype].append(epe)
                self._results[er_ftype].append(er)
        self._results['epe_all'] = epe_all
        self._results['er_all'] = er_all

        # TODO: temporily solution for raw losses
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "PIV2D_raw_losses.json")
            self._logger.info("Saving raw losses to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._results))
                f.flush()

        # Compute mean for later process
        for k, v in self._results.items():
            self._results[k] = np.mean(v)
