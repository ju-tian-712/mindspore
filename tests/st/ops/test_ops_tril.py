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
import pytest
import numpy as np
import mindspore as ms
from mindspore import context, Tensor
from mindspore.ops import tril
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def generate_expect_forward_output(x, k, dtype):
    return np.tril(x, k=k).astype(dtype)

def generate_expect_backward_output(x, k, dtype):
    return np.tril(x, k=k).astype(dtype)

@test_utils.run_with_cell
def tril_forward_func(x, k):
    return tril(x, k)

@test_utils.run_with_cell
def tril_backward_func(x, k):
    return ms.ops.grad(tril_forward_func, (0))(x, k)

@test_utils.run_with_cell
def tril_vmap_func(x, k):
    return ms.ops.vmap(tril_forward_func, in_axes=(0, None), out_axes=(0,))(x, k)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("k", [0, 1, -1])
def test_tril_normal(mode, k):
    """
    Feature: test tril operator
    Description: test tril run by pyboostf
    Expectation: success
    """
    context.set_context(mode=mode)

    ## forward
    np_array = np.random.rand(7, 4, 6, 2)
    x = Tensor(np_array, ms.float32)
    k = k
    output = tril_forward_func(x, k)
    expect = generate_expect_forward_output(np_array, k, np.float32)
    assert np.allclose(output.asnumpy(), expect)

    ## backward
    context.set_context(mode=mode)
    np_array = np.random.rand(2, 3, 4).astype(np.float32)
    x = Tensor(np_array, ms.float32)
    output = tril_backward_func(x, k)
    expect = generate_expect_backward_output(np_array, k, np.float32)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("k", [0, 1, -1])
def test_ops_tril_vmap(mode, k):
    """
    Feature: pyboost function.
    Description: test function tril vmap feature.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    k = k
    output = tril_vmap_func(Tensor(x), k)
    expect = generate_expect_forward_output(x, k, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tril_bfloat16(mode):
    """
    Feature: test tril operator
    Description: test tril run by pyboost
    Expectation: success
    """
    context.set_context(mode=mode)
    np_array = np.random.rand(2, 3, 4)
    x = Tensor(np_array, ms.bfloat16)
    k = 0
    output = tril_forward_func(x, k)
    expect = generate_expect_forward_output(np_array, k, np.float32)
    assert np.allclose(output.float().asnumpy(), expect, rtol=4e-3, atol=4e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_tril_dynamic(mode):
    """
    Feature: test dynamic by TEST_OP.
    Description: test ops.tril dynamic shape feature.
    Expectation: expect correct result.
    """
    input_case1 = Tensor(np.random.rand(3, 4, 5, 6).astype(np.float32))
    input_case2 = Tensor(np.random.rand(3, 4).astype(np.float32))
    TEST_OP(tril_forward_func, [[input_case1, 0], [input_case2, 1]], mode=mode, grad=True)
