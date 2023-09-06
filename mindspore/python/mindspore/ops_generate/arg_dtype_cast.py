# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
# Copyright 2023-2024 Huawei Technologies Co., Ltd
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
"""Operator argument data type cast function."""

from numbers import Number
from mindspore import ops
from mindspore.common.tensor import Tensor


def int_to_tuple(data):
    return (data,)


def int_to_float(data):
    return float(data)


def float_to_tuple(data):
    return (data,)


def bool_to_tuple(data):
    return (data,)


def number_to_tuple(data):
    return (data,)


def list_to_tuple(data):
    return tuple(data)


def tensor_to_tuple(data):
    raise NotImplementedError


def int_to_tensor(data):
    return ops.scalar_to_tensor(data)


def float_to_tensor(data):
    return ops.scalar_to_tensor(data)


def bool_to_tensor(data):
    return ops.scalar_to_tensor(data)


def number_to_tensor(data):
    return ops.scalar_to_tensor(data)


def tuple_to_tensor(data):
    return ops.tuple_to_array(data)


def list_to_tensor(data):
    return ops.tuple_to_array(tuple(data))


def type_it(data, src_type, dst_type):
    """
    cast operator argument data type.
    """
    if isinstance(data, dst_type):
        return data
    if not isinstance(data, src_type):
        raise TypeError(f"For type_it, the {data} should be {src_type}, but get {type(data)}")
    if isinstance(data, int) and dst_type is float:
        return int_to_float(data)
    # to tuple
    if isinstance(data, int) and dst_type is tuple:
        return int_to_tuple(data)
    if isinstance(data, float) and dst_type is tuple:
        return float_to_tuple(data)
    if isinstance(data, Number) and dst_type is tuple:
        return number_to_tuple(data)
    if isinstance(data, bool) and dst_type is tuple:
        return bool_to_tuple(data)
    if isinstance(data, list) and dst_type is tuple:
        return list_to_tuple(data)
    if isinstance(data, Tensor) and dst_type is tuple:
        return tensor_to_tuple(data)
    # to tensor
    if isinstance(data, int) and dst_type is Tensor:
        return int_to_tensor(data)
    if isinstance(data, float) and dst_type is Tensor:
        return float_to_tensor(data)
    if isinstance(data, bool) and dst_type is Tensor:
        return bool_to_tensor(data)
    if isinstance(data, Number) and dst_type is Tensor:
        return number_to_tensor(data)
    if isinstance(data, tuple) and dst_type is Tensor:
        return tuple_to_tensor(data)
    if isinstance(data, list) and dst_type is Tensor:
        return list_to_tensor(data)

    raise TypeError("Unsupported type cast.")
