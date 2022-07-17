from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

_C.VERSION = 1

_C.MODEL = CN()
_C.MODEL.WEIGHTS = ""
# ---------------------------------------------------------------------------- #
# Predictor options
# ---------------------------------------------------------------------------- #
_C.MODEL.PREDICTOR = CN()

_C.MODEL.PREDICTOR.NAME = "unpwc"
# Weights for each layer of flownet series modeling
_C.MODEL.PREDICTOR.FLOWNET_DEFAULT_WEIGHTS = [12.7, 5.5, 4.35, 3.9, 3.4, 1.1]


_C.MODEL.CORRECTOR = CN()

_C.MODEL.CORRECTOR.NAME = "corrector"
_C.MODEL.CORRECTOR.UPSTREAM_PREDICTOR = "unpwc"

_C.MODEL.CORRECTOR.COEF_CONVECTION = 0.1
_C.MODEL.CORRECTOR.COEF_DIFFUSION = 1.0


# Values to be used for image normalization (BGR order, since INPUT.FORMAT defaults to BGR).
# To train on images of different number of channels, just set different mean & std.
# Default values are the mean pixel value from ImageNet: [103.53, 116.28, 123.675]
# _C.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
# When using pre-trained models in Detectron1 or any MSRA models,
# std has been absorbed into its conv1 weights, so the std needs to be set 1.
# Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
# _C.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()

# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = (256,)
# Sample size of smallest side by choice or random selection from range give by
# INPUT.MIN_SIZE_TRAIN
_C.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 256
# Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
_C.INPUT.MIN_SIZE_TEST = 256
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 256
# Mode for flipping images used in data augmentation during training
# choose one of ["horizontal, "vertical", "none"]
# _C.INPUT.RANDOM_FLIP = "horizontal"

# `True` if cropping is used for data augmentation during training
_C.INPUT.CROP = CN({"ENABLED": False})
# Cropping type:
# - "relative" crop (H * CROP.SIZE[0], W * CROP.SIZE[1]) part of an input of size (H, W)
# - "relative_range" uniformly sample relative crop size from between [CROP.SIZE[0], [CROP.SIZE[1]].
#   and  [1, 1] and use it as in "relative" scenario.
# - "absolute" crop part of an input with absolute size: (CROP.SIZE[0], CROP.SIZE[1]).
# - "absolute_range", for an input of size (H, W), uniformly sample H_crop in
#   [CROP.SIZE[0], min(H, CROP.SIZE[1])] and W_crop in [CROP.SIZE[0], min(W, CROP.SIZE[1])]
_C.INPUT.CROP.TYPE = "relative_range"
# Size of crop in range (0, 1] if CROP.TYPE is "relative" or "relative_range" and in number of
# pixels if CROP.TYPE is "absolute"
_C.INPUT.CROP.SIZE = [0.9, 0.9]

# Whether the modeling needs RGB, YUV, HSV etc.
# Should be one of the modes defined here, as we use PIL to read the image:
# https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
# with BGR being the one exception. One can set image format to BGR, we will
# internally use RGB for conversion and flip the channels over
_C.INPUT.FORMAT = "RGB"


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training. Must be registered in DatasetCatalog
_C.DATASETS.TRAIN = ('PIV2D',)
_C.DATASETS.FLOW_TYPE = ('All',)
# List of the dataset names for testing. Must be registered in DatasetCatalog
_C.DATASETS.TEST = ('PIV2D',)
_C.DATASETS.TOTAL_NUM = -1
_C.DATASETS.SHUFFLE = True
# Default 10% percent data for test use
_C.DATASETS.TEST_RATIO = 0.1
_C.DATASETS.SEQUENCE_LENGTH = 10


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = False
# Options: TrainingSampler, RepeatFactorTrainingSampler
_C.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
# Repeat threshold for RepeatFactorTrainingSampler
_C.DATALOADER.REPEAT_THRESHOLD = 0.0

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

# See detectron2/solver/build.py for LR scheduler options
_C.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"

_C.SOLVER.MAX_ITER = 40000

_C.SOLVER.MAX_EPOCH = 50

_C.SOLVER.BASE_LR = 0.0001

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.NESTEROV = False

_C.SOLVER.WEIGHT_DECAY = 0.0001
# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0

_C.SOLVER.GAMMA = 0.1
# The iteration number to decrease learning rate by GAMMA.
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"

# Save a checkpoint after every this number of iterations
_C.SOLVER.CHECKPOINT_PERIOD = 5000

# Number of images per batch across all machines.
# If we have 16 GPUs and IMS_PER_BATCH = 32,
# each GPU will see 2 images per batch.
# May be adjusted automatically if REFERENCE_WORLD_SIZE is set.
_C.SOLVER.IMS_PER_BATCH = 16

# The reference number of workers (GPUs) this config is meant to train with.
# It takes no effect when set to 0.
# With a non-zero value, it will be used by DefaultTrainer to compute a desired
# per-worker batch size, and then scale the other related configs (total batch size,
# learning rate, etc) to match the per-worker batch size.
# See documentation of `DefaultTrainer.auto_scale_workers` for details:
_C.SOLVER.REFERENCE_WORLD_SIZE = 0

# Detectron v1 (and previous detection code) used a 2x higher LR and 0 WD for
# biases. This is not useful (at least for recent models). You should avoid
# changing these and they exist only to reproduce Detectron v1 training if
# desired.
_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.WEIGHT_DECAY_BIAS = _C.SOLVER.WEIGHT_DECAY

# Gradient clipping
_C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": False})
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
# Maximum absolute value used for clipping gradients
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

# Enable automatic mixed precision for training
# Note that this does not change model's inference behavior.
# To use AMP in inference, run inference under autocast()
_C.SOLVER.AMP = CN({"ENABLED": False})


# ---------------------------------------------------------------------------- #
# Specific train options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

# Period for saving checkpoints, -1 to disable
# _C.TRAIN.CHECKPOINT_PERIOD = 5
# Train batch size, set a moderate samll value for stochasticity
# _C.TRAIN.BATCH_SIZE = 8

# Supervised training
_C.TRAIN.SUPERVISE = False


# ---------------------------------------------------------------------------- #
# Loss functions settings
# ---------------------------------------------------------------------------- #
_C.TRAIN.LOSS = CN()

# Weights for including the specific loss
_C.TRAIN.LOSS.PHOTOMETRIC = 1.0

_C.TRAIN.LOSS.SMOOTHNESS_1ST = 0.0
_C.TRAIN.LOSS.SMOOTHNESS_2ND = 0.0

_C.TRAIN.LOSS.CONSISTENCY = 0.0

_C.TRAIN.LOSS.DIVERGENCE_1ST = 0.0
_C.TRAIN.LOSS.DIVERGENCE_2ND = 0.0

_C.TRAIN.LOSS.CURL_1ST = 0.0
_C.TRAIN.LOSS.CURL_2ND = 0.0


# weights for corrector training
_C.TRAIN.LOSS.VORTICITY_WARP = 0.0

_C.TRAIN.LOSS.REFINE_REGULARIZER = 0.0

_C.TRAIN.LOSS.CORRECTOR_DIVERGENCE = 0.0


# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# For end-to-end tests to verify the expected accuracy.
# Each item is [task, metric, value, tolerance]
# e.g.: [['bbox', 'AP', 38.5, 0.2]]
_C.TEST.EXPECTED_RESULTS = []
# The period (in terms of steps) to evaluate the model during training.
# Set to 0 to disable.
_C.TEST.EVAL_PERIOD = 0
# Test batch size, set a moderate large value for fast test.
_C.TEST.BATCH_SIZE = 128
_C.TEST.AUG = CN({"ENABLED": False})
_C.TEST.AUG.MIN_SIZES = (400, 500, 600, 700, 800, 900, 1000, 1100, 1200)
_C.TEST.AUG.MAX_SIZE = 4000
_C.TEST.AUG.FLIP = True

_C.TEST.PRECISE_BN = CN({"ENABLED": False})
_C.TEST.PRECISE_BN.NUM_ITER = 200


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.DEVICE = "cuda"
# Number of GPUS to use in the experiment
_C.NUM_GPUS = 1
# Number of workers for doing things
_C.NUM_WORKERS = 4

# Directory where output files are written
_C.OUTPUT_DIR = "./output"
_C.DATA_DIR = "./sample_data"

# Set seed to negative to fully randomize everything.
# Set seed to positive to use a fixed seed. Note that a fixed seed increases
# reproducibility but does not guarantee fully deterministic behavior.
# Disabling all parallelism further increases reproducibility.
_C.SEED = 42
# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_C.CUDNN_BENCHMARK = True
# The period (in terms of steps) for minibatch visualization at train time.
# Set to 0 to disable.
_C.VIS_PERIOD = 0

# global config is for quick hack purposes.
# You can set them in command line or config files,
# and access it with:
#
# from detectron2.config import global_cfg
# print(global_cfg.HACK)
#
# Do not commit any configs into it.
_C.GLOBAL = CN()
_C.GLOBAL.HACK = 1.0