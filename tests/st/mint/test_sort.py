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
from mindspore import ops, mint, jit, JitConfig
import tests.st.utils.test_utils as test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_expect_backward_output(values, indices, x, dim):
    ones = np.ones_like(values)
    grad_output = np.zeros_like(x)
    np.put_along_axis(grad_output, indices, ones, dim)
    return grad_output


def sort_forward_func(x, dim, descending, stable):
    return mint.sort(x, dim, descending, stable)


def sort_backward_func(x, dim, descending, stable):
    return ops.grad(sort_forward_func, (0, 1, 2, 3))(x, dim, descending, stable)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize("descending", [True, False])
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_sort_std(descending, mode):
    """
    Feature: Test sort with standard forward, backward feature.
    Description: call mint.sort with valid input and index.
    Expectation: return the correct value.
    """
    x_numpy = np.array([[[1, 2, 3, 4], [8, 7, 2, 0], [9, 4, 1, 8]],
                        [[5, 4, 1, 8], [2, 9, 0, 7], [6, 1, 7, 4]]]).astype(np.float32)
    x = ms.Tensor(x_numpy)

    expect_indices_list = [
        np.array([[[0, 1, 2, 3], [3, 2, 1, 0], [2, 1, 3, 0]],
                  [[2, 1, 0, 3], [2, 0, 3, 1], [1, 3, 0, 2]]]),
        np.array([[[0, 0, 2, 1], [1, 2, 1, 0], [2, 1, 0, 2]],
                  [[1, 2, 1, 2], [0, 0, 0, 1], [2, 1, 2, 0]]]),
        np.array([[[0, 0, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1]],
                  [[1, 1, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0]]])
    ]
    for dim, expected_indices in zip([-1, 1, -3], expect_indices_list):
        expected_output = np.sort(x_numpy, dim)

        if descending:
            if dim == -1:
                expected_output = expected_output[:, :, ::-1]
                expected_indices = expected_indices[:, :, ::-1]
            elif dim == 1:
                expected_output = expected_output[:, ::-1, :]
                expected_indices = expected_indices[:, ::-1, :]
            elif dim == -3:
                expected_output = expected_output[::-1, :, :]
                expected_indices = expected_indices[::-1, :, :]

        expected_grad = generate_expect_backward_output(expected_output, expected_indices, x, dim)

        if mode == 'pynative':
            ms.context.set_context(mode=ms.PYNATIVE_MODE)
            output, indices = sort_forward_func(x, dim, descending, False)
            ms_grad = sort_backward_func(x, dim, descending, False)
        else:
            output, indices = (jit(sort_forward_func, jit_config=JitConfig(jit_level="O0")))(x, dim, descending, False)
            ms_grad = (jit(sort_backward_func, jit_config=JitConfig(jit_level="O0")))(x, dim, descending, False)

        np.testing.assert_array_equal(output.asnumpy(), expected_output)
        np.testing.assert_array_equal(indices.asnumpy(), expected_indices)
        np.testing.assert_array_equal(ms_grad.asnumpy(), expected_grad)


def sort_forward_dyn_func(input_tensor):
    return mint.sort(input_tensor, -1, True, True)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_sort_dynamic_shape(mode):
    """
    Feature: Test sort with dynamic shape.
    Description: call mint.sort with valid input and index.
    Expectation: return the correct value.
    """
    x1 = np.array([[[1, 2, 3, 4], [8, 7, 2, 0], [9, 4, 1, 8]],
                   [[5, 4, 1, 8], [2, 9, 0, 7], [6, 1, 7, 4]]]).astype(np.float32)
    tensor_1 = ms.Tensor(x1)
    x2 = np.array([1, 0, 3, 4]).astype(np.float32)
    tensor_2 = ms.Tensor(x2)
    test_cell = test_utils.to_cell_obj(sort_forward_dyn_func)
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        TEST_OP(test_cell, [[tensor_1], [tensor_2]], grad=True, mode=ms.PYNATIVE_MODE)
    else:
        TEST_OP(test_cell, [[tensor_1], [tensor_2]], grad=True, jit_level='O0')
