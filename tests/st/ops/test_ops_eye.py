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
import numpy as np
import pytest
import mindspore as ms
from mindspore import mint
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_expect_forward_output(n, m, dtype):
    return np.eye(n, m, dtype=dtype)


@test_utils.run_with_cell
def eye_forward_func(n, m, dtype=ms.float32):
    return mint.eye(n, m, dtype)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_eye_forward(context_mode):
    """
    Feature: pyboost function.
    Description: test function eye forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    cases = [
        [5, 5, ms.float32],
        [1, 1, ms.float32],
        [2, None, ms.float32]
    ]
    for n, m, dtype in cases:
        output = eye_forward_func(n, m, dtype)
        expect = generate_expect_forward_output(n, m, np.float32)
        np.testing.assert_allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_eye_dynamic(context_mode):
    """
    Feature: Test operator Eye by TEST_OP
    Description:  Test operator Eye with dynamic input
    Expectation: the result of Eye is correct.
    """

    TEST_OP(eye_forward_func, [[3, 3], [4, 5]], grad=False, mode=context_mode, jit_level='O0',
            nontensor_dynamic_type='None')
