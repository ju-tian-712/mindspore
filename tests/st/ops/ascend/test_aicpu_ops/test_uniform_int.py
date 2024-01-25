# Copyright 2020 Huawei Technologies Co., Ltd
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

import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype


class Net(nn.Cell):
    def __init__(self, shape, seed=0, seed2=0):
        super(Net, self).__init__()
        self.uniformint = P.UniformInt(seed=seed)
        self.shape = shape

    def construct(self, minval, maxval):
        return self.uniformint(self.shape, minval, maxval)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_net_1D(mode):
    """
    Feature: test UniformInt op.
    Description: test UniformInt op.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="Ascend")
    seed = 10
    shape = (3, 2, 4)
    minval = 1
    maxval = 5
    net = Net(shape, seed=seed)
    tminval, tmaxval = Tensor(minval, mstype.int32), Tensor(maxval, mstype.int32)
    output = net(tminval, tmaxval)
    assert output.shape == (3, 2, 4)
