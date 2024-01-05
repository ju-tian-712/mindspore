/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_ELEWISE_UNARY_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_ELEWISE_UNARY_H_
#include <string>
#include <vector>
#include <utility>

#include "plugin/device/ascend/kernel/internal/internal_kernel_mod.h"

namespace mindspore {
namespace kernel {
class ElewiseUnary : public InternalKernelMod {
 public:
  explicit ElewiseUnary(std::string &&op_type) : InternalKernelMod(std::move(op_type)) {}
  ~ElewiseUnary() = default;

 protected:
  virtual void SetComputeType(internal::OpParamPtr param_ptr) = 0;
  internal::OpParamPtr CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs);
  void SetInOutIdx();
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_ELEWISE_UNARY_H_
