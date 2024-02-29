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
import mindspore as ms
from mindspore import context, nn, Tensor, ops


class AddNet(nn.Cell):
    def construct(self, x, y):
        return x + y


class SubNet(nn.Cell):
    def construct(self, x, y):
        return x - y


class CastNet(nn.Cell):
    def __init__(self, out_dtype):
        super().__init__()
        self.out_dtype = out_dtype

    def construct(self, x):
        return ops.cast(x, self.out_dtype)


class GeluNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.gelu = ops.GeLU()

    def construct(self, x):
        return self.gelu(x)


def add_net(x_shape, y_shape, dtype):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    x = np.random.randn(*x_shape).astype(dtype)
    y = np.random.randn(*y_shape).astype(dtype)

    net = AddNet()
    output = net(Tensor(x), Tensor(y))
    expected = x + y

    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


def add_net_bf16(x_shape, y_shape, dtype):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    x = np.random.randn(*x_shape).astype(dtype)
    y = np.random.randn(*y_shape).astype(dtype)
    from bfloat16 import bfloat16
    expected = x.astype(bfloat16) + y.astype(bfloat16)

    x_t = Tensor(x, dtype=ms.bfloat16)
    y_t = Tensor(y, dtype=ms.bfloat16)
    net = AddNet()
    output = net(x_t, y_t)

    output = ops.cast(output, ms.float32)
    expected = expected.astype(np.float32)

    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


def sub_net(x_shape, y_shape, dtype):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    x = np.random.randn(*x_shape).astype(dtype)
    y = np.random.randn(*y_shape).astype(dtype)

    net = SubNet()
    output = net(Tensor(x), Tensor(y))
    expected = x - y

    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


def cast_net(x_shape, dtype, out_dtype):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    from bfloat16 import bfloat16
    dtype_map = {
        np.int32: ms.int32,
        np.int64: ms.int64,
        np.float16: ms.float16,
        bfloat16: ms.bfloat16
    }
    x = np.random.randn(*x_shape).astype(dtype)

    net = CastNet(dtype_map[out_dtype])
    output = net(Tensor(x))
    expected = x.astype(out_dtype)

    if out_dtype != bfloat16:
        np.testing.assert_array_almost_equal(output.asnumpy(), expected)


def gelu_net(x_shape, dtype):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    x = np.random.randn(*x_shape)

    net = GeluNet()
    output = net(Tensor(x, dtype=dtype))


def test_add_broadcast1(dtype=np.float16):
    """
    Feature: test add operator in graph mode
    Description: test add.
    Expectation: the result is correct
    """
    x_shape = (1024, 1664)
    y_shape = (1664,)
    add_net(x_shape, y_shape, dtype)


def test_add_nobroadcast(dtype=np.float16):
    """
    Feature: test add operator in graph mode
    Description: test add.
    Expectation: the result is correct
    """
    x_shape = (1024, 1664)
    y_shape = (1024, 1664)
    add_net(x_shape, y_shape, dtype)


def test_add_broadcast2(dtype=np.float16):
    """
    Feature: test add operator in graph mode
    Description: test add.
    Expectation: the result is correct
    """
    x_shape = (32, 1, 1664)
    y_shape = (1, 1024, 1664)
    add_net(x_shape, y_shape, dtype)


def test_add_bf16(dtype=np.float32):
    """
    Feature: test add operator in graph mode
    Description: test add.
    Expectation: the result is correct
    """
    x_shape = (1024, 1664)
    y_shape = (1, 1024, 1664)
    add_net(x_shape, y_shape, dtype)


def test_sub():
    """
    Feature: test sub operator in graph mode.
    Description: test sub.
    Expectation: the result is correct
    """
    x_shape = (1,)
    y_shape = (1,)
    dtype = np.int64
    sub_net(x_shape, y_shape, dtype)


def test_cast():
    """
    Feature: test cast operator in graph mode.
    Description: test cast.
    Expectation: the result is correct
    """
    x_shape = (4, 1)
    dtype = np.int64
    out_dtype = np.int32
    cast_net(x_shape, dtype, out_dtype)

    dtype = np.int32
    out_dtype = np.int64
    cast_net(x_shape, dtype, out_dtype)

    from bfloat16 import bfloat16
    dtype = np.float16
    out_dtype = bfloat16
    cast_net(x_shape, dtype, out_dtype)


def test_gelu():
    """
    Feature: test gelu operator in graph mode.
    Description: test gelu.
    Expectation: the result is correct
    """
    x_shape = (1,)
    dtype = ms.bfloat16
    gelu_net(x_shape, dtype)
