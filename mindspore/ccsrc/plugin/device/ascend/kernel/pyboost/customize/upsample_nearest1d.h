/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CALL_UPSAMPLE_NEAREST1D_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CALL_UPSAMPLE_NEAREST1D_H_

#include "ir/tensor.h"
#include "ir/value.h"
#include "runtime/hardware/device_context_manager.h"
#include <vector>

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr UpsampleNearest1dAscendCall(const PrimitivePtr &primitive,
                                              const device::DeviceContext *device_context,
                                              const tensor::TensorPtr &input_tensor, const ValueTuplePtr &output_size,
                                              const ValueTuplePtr &scale_factors,
                                              const std::vector<tensor::TensorPtr> &outputs);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CALL_UPSAMPLE_NEAREST1D_H_
