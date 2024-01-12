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
import glob
import os
import tempfile

import numpy as np
import pytest
from mindspore import Tensor
import mindspore.context as context
import mindspore.nn as nn
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = P.MatMul()
        self.gelu = P.GeLU()

    def construct(self, x, y, z):
        out = self.matmul(x, y, z)
        out = self.gelu(out)
        return out


def check_fusion_op_in_ir(ir_dir):
    fusion_op_name = 'FusionOp_MatMul_Gelu'
    for file in glob.glob(os.path.join(ir_dir, 'hwopt_d_ub_fusion_after_graph*.ir')):
        with open(file, 'r') as fr:
            if fusion_op_name in fr.read():
                return True
    return False


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_matmul_gelu_ub_fusion_success():
    """
    Feature: UB fusion feature on Ascend
    Description: Build a small net to test MatMul and GeLU ub fusion pass,
        check that fused op appears in the saved ir file
    Expectation: Fused op is in the saved ir file
    """
    with tempfile.TemporaryDirectory() as temp_dir_name:
        context.set_context(
            mode=context.GRAPH_MODE, device_target='Ascend', save_graphs=2,
            save_graphs_path=temp_dir_name
        )
        x = np.random.randn(2, 2).astype(np.float16)
        y = np.random.randn(2, 2).astype(np.float16)
        z = np.random.randn(2).astype(np.float16)
        net = Net()
        _ = net(Tensor(x), Tensor(y), Tensor(z))
        assert check_fusion_op_in_ir(temp_dir_name)