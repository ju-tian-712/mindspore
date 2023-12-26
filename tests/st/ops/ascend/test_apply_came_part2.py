# Copyright 2023 Huawei Technologies Co., Ltd
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
import numpy as np
import pytest


import mindspore as ms
import mindspore.nn as nn
from mindspore.ops.operations import _inner_ops as P
from mindspore import Tensor


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.apply_came_part2 = P.ApplyCamePart2()


    def construct(self, *inputs):
        return self.apply_came_part2(*inputs)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_net():
    """
    Feature: test apply_came_part2 tensor api.
    Description: test inputs given their dtype.
    Expectation: execute without error.
    """
    apply_came_part2 = Net()
    grad = Tensor(np.ones([1024, 64]), dtype=ms.float32)
    sum_grad_r = Tensor(np.ones([1024]), dtype=ms.float32)
    sum_grad_c = Tensor(np.ones([64]), dtype=ms.float32)
    sum_grad_rc = Tensor(np.array([64]), dtype=ms.float32)
    r = Tensor(np.ones([1024]), dtype=ms.float32)
    c = Tensor(np.ones([64]), dtype=ms.float32)
    beta2 = 0.5
    sum_r = Tensor(np.array([128]), dtype=ms.float32)
    global_shape = (1024, 64)
    inputs = [grad, sum_grad_r, sum_grad_c, sum_grad_rc, r, c, beta2, sum_r, global_shape]
    output = apply_came_part2(*inputs)
    print(output[0].asnumpy())