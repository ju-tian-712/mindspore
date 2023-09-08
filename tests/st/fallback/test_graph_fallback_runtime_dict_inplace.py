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
"""Test graph dict inplace operation"""
import pytest
import numpy as np

from mindspore import jit, jit_class, nn
from mindspore import context


context.set_context(mode=context.GRAPH_MODE)

global_dict_1 = {"1": 1}


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_global_dict_used_in_graph():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """
    @jit
    def foo():
        return global_dict_1

    res = foo()
    assert id(res) == id(global_dict_1)


global_dict_2 = {"1": [1, 2, 3, 4]}


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_global_dict_used_in_graph_2():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """
    @jit
    def foo():
        return global_dict_2["1"]

    res = foo()
    assert id(res) == id(global_dict_2["1"])


global_dict_3 = {"1": ([np.array([1, 2, 3]), np.array([4, 5, 6])], "test")}


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_global_dict_used_in_graph_3():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """
    @jit
    def foo():
        return global_dict_3["1"]

    res = foo()
    assert id(res[0]) == id(global_dict_3["1"][0])


global_input_dict_1 = {"1": 1}


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_global_dict_as_graph_input():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """
    @jit
    def foo(dict_input):
        return dict_input

    res = foo(global_input_dict_1)
    assert id(res) == id(global_input_dict_1)


global_input_dict_2 = {"1": [1, 2, 3, 4]}


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_global_dict_as_graph_input_2():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """
    @jit
    def foo(dict_input):
        return dict_input["1"]

    res = foo(global_input_dict_2)
    assert id(res) == id(global_input_dict_2["1"])


global_input_dict_3 = {"1": ([1, 2, 3, 4], 5, 6)}


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_global_dict_as_graph_input_3():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """
    @jit
    def foo(dict_input):
        return dict_input["1"]

    res = foo(global_input_dict_3)
    assert id(res[0]) == id(global_input_dict_3["1"][0])


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dict_inplace_with_attribute():
    """
    Feature: Enable dict do inplace operation.
    Description: support dict inplace ops.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def __init__(self, x):
            super(Net, self).__init__()
            self.x = x

        def construct(self):
            return self.x

    x = {"1": 1, "2": 2}
    net = Net(x)
    ret = net()
    assert id(x) == id(ret)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dict_inplace_with_attribute_2():
    """
    Feature: Enable dict do inplace operation.
    Description: support dict inplace ops.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def __init__(self, x):
            super(Net, self).__init__()
            self.x = x

        def construct(self):
            return self.x["2"]

    x = {"1": 1, "2": [1, 2, 3, 4]}
    net = Net(x)
    ret = net()
    assert id(x["2"]) == id(ret)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dict_inplace_setitem():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """
    @jit
    def foo(dict_input):
        dict_input["a"] = 3
        return dict_input

    x = {"a": 1, "b": 2}
    res = foo(x)
    assert res == {"a": 3, "b": 2}
    assert id(x) == id(res)


@pytest.mark.skip(reason="Dictionary with no return will be convert to tuple")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dict_inplace_setitem_2():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """
    @jit
    def foo(dict_input):
        dict_input["a"] = 3

    x = {"a": 1, "b": 2}
    foo(x)
    assert x == {"a": 3, "b": 2}


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dict_inplace_setitem_3():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """
    @jit
    def foo(dict_input, list_input):
        dict_input["b"] = list_input
        return dict_input

    x = {"a": 1, "b": 2}
    y = [1, 2, 3, 4]
    res = foo(x, y)
    assert res == {"a": 1, "b": [1, 2, 3, 4]}
    assert id(x) == id(res)
    assert id(y) == id(res["b"])


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dict_inplace_setitem_with_attribute():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, x):
            super(Net, self).__init__()
            self.x = x

        def construct(self):
            self.x["1"] = 10
            self.x["3"] = 3
            return self.x

    x = {"1": 1, "2": 2}
    net = Net(x)
    ret = net()
    assert ret == {"1": 10, "2": 2, "3": 3}
    assert net.x == {"1": 10, "2": 2, "3": 3}
    assert id(x) == id(ret)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dict_inplace_setitem_with_attribute_2():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """
    @jit_class
    class AttrClass():
        def __init__(self, x):
            self.attr = x

    class Net(nn.Cell):
        def __init__(self, x):
            super(Net, self).__init__()
            self.x = x

        def construct(self):
            self.x.attr["1"] = 10
            return self.x.attr

    x = {"1": 1, "2": 2}
    obj = AttrClass(x)
    net = Net(obj)
    ret = net()
    assert ret == {"1": 10, "2": 2}
    assert net.x.attr == {"1": 10, "2": 2}
    assert id(x) == id(ret)


@pytest.mark.skip(reason="setitem with abstract any do not convert to pyexecute yet")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dict_inplace_setitem_with_attribute_3():
    """
    Feature: Enable dict inplace operation
    Description: Dict after inplace operation should keep object not changed.
    Expectation: No exception.
    """
    class AttrClass():
        def __init__(self, x):
            self.attr = x

    class Net(nn.Cell):
        def __init__(self, x):
            super(Net, self).__init__()
            self.x = x

        def construct(self):
            self.x.attr["1"] = 10
            return self.x.attr

    x = {"1": 1, "2": 2}
    obj = AttrClass(x)
    net = Net(obj)
    ret = net()
    assert ret == {"1": 10, "2": 2}
    assert net.x.attr == {"1": 10, "2": 2}
    assert id(x) == id(ret)
