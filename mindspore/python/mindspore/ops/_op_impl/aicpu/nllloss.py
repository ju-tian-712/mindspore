# Copyright 2022 Huawei Technologies Co., Ltd
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

"""NLLLoss op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

nll_loss_op_info = AiCPURegOp("NLLLoss") \
    .fusion_type("OPAQUE") \
    .attr("reduction", "str") \
    .attr("ignore_index", "int") \
    .input(0, "x", "required") \
    .input(1, "target", "required") \
    .input(2, "weight", "optional") \
    .output(0, "y", "required") \
    .output(1, "total_weight", "required") \
    .dtype_format(DataType.F32_Default, DataType.I32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default) \
    .dtype_format(DataType.F32_Default, DataType.I64_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default) \
    .get_op_info()


@op_info_register(nll_loss_op_info)
def _nll_loss_aicpu():
    """NLLLoss aicpu register"""
    return
