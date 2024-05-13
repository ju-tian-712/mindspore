# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""mint nn functional."""
from __future__ import absolute_import
from mindspore.ops.functional import (
    conv_transpose2d,
    grid_sample
)
# 1

# 2

# 3

# 4

# 5
from mindspore.ops.functional import pad_ext as pad
# 6
from mindspore.ops.function.array_func import unfold_ext as unfold
# 7
from mindspore.ops.auto_generate import fold_ext as fold
# 8
from mindspore.ops.functional import layer_norm
# 9
from mindspore.ops.function.nn_func import interpolate_ext as interpolate
# 10

# 11
from mindspore.ops.functional import relu
# 12

# 13

# 14
from mindspore.ops.function.nn_func import dropout_ext as dropout
# 15

# 16

# 17

# 18

# 19

# 20

# 21

# 22
from mindspore.ops.extend import max_pool2d as max_pool2d_ex

# 23

# 24

# 25

# 26

# 27

# 28

# 29

# 30

# 31
from mindspore.ops.functional import softmax as softmax_ex

# 32

# 33

# 34
from mindspore.ops.functional import batch_norm as batch_norm_ex

# 35

# 36
from mindspore.ops.functional import gelu
# 37

# 38
from mindspore.ops.auto_generate import dense

# 39
from mindspore.ops.functional import group_norm
# 40

# 41

# 42

# 43

# 44

# 45

# 46
from mindspore.ops.functional import silu
# 47

# 48

# 49
from mindspore.ops.functional import sigmoid
# 50

# 51

# 52
from mindspore.ops.functional import embedding
# 53

# 54
from mindspore.ops import normal_ext as normal
# 55

# 56

# 57

# 58

# 59

# 60

# 61

# 62

# 63

# 64
from mindspore.ops.extend import one_hot as one_hot_ext

# 65

# 66

# 67

# 68

# 69

# 70

# 71

# 72

# 73

# 74

# 75

# 76

# 77

# 78

# 79

# 80

# 81

# 82

# 83

# 84

# 85

# 86

# 87

# 88

# 89

# 90
from mindspore.ops.function.nn_func import avg_pool2d_ext as avg_pool2d
# 91

# 92
from mindspore.ops.extend import leaky_relu_ext as leaky_relu
# 93
from mindspore.ops.function.nn_func import softplus_ext as softplus
# 94
from mindspore.ops.function.math_func import tanh


# 95

# 96

# 97

# 98

# 99

# 100
from mindspore.ops.auto_generate import binary_cross_entropy_with_logits as bce_with_logits
def binary_cross_entropy_with_logits(input, target, weight, reduction, pos_weight):
    return bce_with_logits(input, target, weight, pos_weight, reduction)

def batch_norm(input_x, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    return batch_norm_ex(input_x, running_mean, running_var, weight, bias, training, momentum, eps)


def linear(input, weight, bias=None):
    return dense(input, weight, bias)


def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    return max_pool2d_ex(input, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         return_indices=return_indices, ceil_mode=ceil_mode)


def one_hot(tensor, num_classes=-1):
    return one_hot_ext(tensor, num_classes)


def softmax(input, dim=None, dtype=None):
    dim = -1 if dim is None else dim
    return softmax_ex(input, dim, dtype=dtype)


__all__ = [
    'conv_transpose2d',
    'max_pool2d',
    # 1
    'binary_cross_entropy_with_logits',
    # 2

    # 3

    # 4

    # 5
    'pad',
    # 6
    'unfold',
    # 7
    'fold',
    # 8
    'layer_norm',
    # 9
    'interpolate',
    # 10

    # 11
    'relu',
    # 12

    # 13

    # 14
    'dropout',
    # 15

    # 16

    # 17

    # 18

    # 19

    # 20

    # 21

    # 22

    # 23

    # 24

    # 25

    # 26

    # 27

    # 28

    # 29

    # 30

    # 31
    'softmax',
    # 32

    # 33

    # 34
    'batch_norm',
    # 35

    # 36
    'gelu',
    # 37

    # 38
    'linear',
    # 39
    'group_norm',
    # 40

    # 41

    # 42

    # 43

    # 44

    # 45

    # 46
    'silu',
    # 47

    # 48

    # 49
    'sigmoid',
    # 50

    # 51

    # 52
    'embedding',
    # 53

    # 54
    'normal',
    # 55

    # 56

    # 57

    # 58

    # 59

    # 60

    # 61

    # 62

    # 63

    # 64
    'one_hot',
    # 65

    # 66

    # 67

    # 68

    # 69

    # 70

    # 71

    # 72

    # 73

    # 74

    # 75

    # 76

    # 77

    # 78

    # 79

    # 80

    # 81

    # 82

    # 83

    # 84

    # 85

    # 86

    # 87

    # 88

    # 89

    # 90
    'avg_pool2d',
    # 91
    'grid_sample',
    # 92
    'leaky_relu',
    # 93
    'softplus',
    # 94
    'tanh',
    # 95

    # 96

    # 97

    # 98

    # 99

    # 100
]
