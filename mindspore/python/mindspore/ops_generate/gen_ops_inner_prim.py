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
"""Primitive defined for arg handler."""

from mindspore.ops.primitive import Primitive, prim_attr_register
from mindspore._c_expression import typing
from mindspore._c_expression import op_enum


class DtypeToEnum(Primitive):
    r"""
    Convert mindspore dtype to enum.

    Inputs:
        - **dtype** (mindspore.dtype) - The data type.

    Outputs:
        An integer.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    @prim_attr_register
    def __init__(self):
        """Initialize"""

    def __call__(self, dtype):
        """Run in PyNative mode"""
        if not isinstance(dtype, typing.Type):
            raise TypeError(f"For dtype_to_enum function, the input should be mindpsore dtype, but got {dtype}.")
        return typing.type_to_type_id(dtype)


class StringToEnum(Primitive):
    r"""
    Convert string to enum.

    Inputs:
        - **enum_str** (str) - The str data.

    Outputs:
        An integer.

    Supported Platforms:
        ``CPU``
    """

    @prim_attr_register
    def __init__(self):
        """Initialize"""

    def __call__(self, enum_str):
        """Run in PyNative mode"""
        if not isinstance(enum_str, str):
            raise TypeError(f"For StringToEnum op, the input should be a str, but got {type(enum_str)}.")
        return op_enum.str_to_enum(enum_str)