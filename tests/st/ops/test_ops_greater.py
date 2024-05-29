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
from mindspore import Tensor, ops
from mindspore.mint import greater
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x, y):
    return np.greater(x, y)


@test_utils.run_with_cell
def greater_forward_func(x, y):
    return greater(x, y)


@test_utils.run_with_cell
def greater_backward_func(x, y):
    return ops.grad(greater_forward_func)(x, y)


@test_utils.run_with_cell
def greater_vmap_func(x, y):
    return ops.vmap(greater_forward_func, in_axes=0, out_axes=0)(x, y)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_greater_op_forward(mode):
    """
    Feature: test notequal op
    Description: test notequal run by pyboost
    Expectation: expect correct forward result.
    """
    ms.set_context(mode=mode)
    x = generate_random_input((1, 2, 3, 4), np.float32)
    y = generate_random_input((1, 2, 3, 4), np.float32)
    output = greater_forward_func(ms.Tensor(x), ms.Tensor(y))
    expect = generate_expect_forward_output(x, y)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_greater_op_forward_case01(mode):
    """
    Feature: test notequal op
    Description: test notequal run by pyboost
    Expectation: expect correct forward result.
    """
    ms.set_context(mode=mode)
    x = Tensor(np.array([24]).astype(np.float32))
    y = Tensor(np.array([1]).astype(np.float32))
    output = greater_forward_func(x, y)
    assert np.allclose(output.asnumpy(), [True])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_greater_op_backward(mode):
    """
    Feature: test notequal op
    Description: test notequal run by pyboost
    Expectation: expect correct forward result.
    """
    ms.set_context(mode=mode)
    x = Tensor(np.array([1, 2]).astype(np.float32))
    y = Tensor(np.array([2, 1]).astype(np.float32))
    output = greater_backward_func(ms.Tensor(x), ms.Tensor(y))
    expect_out = np.array([0., 0., 0.]).astype(np.float32)
    assert np.allclose(output[0].asnumpy(), expect_out, rtol=1e-4)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_greater_op_vmap(mode):
    """
    Feature: test notequal op
    Description: test notequal run by pyboost
    Expectation: expect correct forward result.
    """
    ms.set_context(mode=mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    y = generate_random_input((2, 3, 4, 5), np.float32)
    output = greater_vmap_func(ms.Tensor(x), ms.Tensor(y))
    expect_out = generate_expect_forward_output(x, y)
    np.testing.assert_array_equal(output.asnumpy(), expect_out)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
def test_greater_op_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function greater with dynamic shape in graph mode.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    y1 = generate_random_input((2, 3, 4, 5), np.float32)
    x2 = generate_random_input((3, 4, 5, 6, 7), np.float32)
    y2 = generate_random_input((3, 4, 5, 6, 7), np.float32)

    TEST_OP(greater_forward_func
            , [[ms.Tensor(x1), ms.Tensor(y1)], [ms.Tensor(x2), ms.Tensor(y2)]], 'greater')
