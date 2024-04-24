from marllib.marl import JointQRNN as DRQNModel  # noqa
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as FCN # noqa
from .matrix_mlp import MatrixGameQMLP
from .matrix_split_mlp import MatrixGameSplitQMLP
from .dueling_q import DuelingQModel
