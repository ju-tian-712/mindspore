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
#include "plugin/device/ascend/kernel/opapi/aclnn_kernel_mod.h"
#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include <functional>
#include "ir/tensor.h"
#include "runtime/stream.h"
#include "runtime/device/kernel_runtime.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "transform/acl_ir/acl_helper.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {

bool AclnnKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  MS_LOG(DEBUG) << "AclnnKernelMod Init";
  input_tensors_.resize(inputs.size());
  output_tensors_.resize(outputs.size());
  GetWorkSpaceInfo(inputs, outputs);
  return true;
}

int AclnnKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  GetWorkSpaceInfo(inputs, outputs);
  return KernelMod::Resize(inputs, outputs);
}

bool AclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                            const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  return true;
}

void AclnnKernelMod::ParseGenExecutor(const std::tuple<uint64_t, aclOpExecutor *, CallBackFunc> &args) {
  auto workspace_size = static_cast<size_t>(std::get<0>(args));
  if (workspace_size != 0) {
    std::vector<size_t> workspace_size_list = {workspace_size};
    SetWorkspaceSizeList(workspace_size_list);
  }
  executor_ = std::get<1>(args);
  if (executor_ == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Please check op api's generate!";
  }
  after_launch_func_ = std::get<2>(args);
  if (after_launch_func_ == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Please check op api's call back func!";
  }
}

void AclnnKernelMod::SetInputsInfo(const std::vector<TypeId> &type_ids, const ShapeArray &shapes) {
  input_size_list_.resize(type_ids.size());
  if (type_ids.size() != shapes.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Aclnn kernel's input type size is not equal with shape size:" << shapes.size()
                               << " and type's size:" << type_ids.size();
  }
  input_size_list_.resize(type_ids.size(), 0);
}

void AclnnKernelMod::SetOutputsInfo(const std::vector<TypeId> &type_ids, const ShapeArray &shapes) {
  if (type_ids.size() != shapes.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Aclnn kernel's output type size is not equal with shape size:" << shapes.size()
                               << " and type's size:" << type_ids.size();
  }
  output_size_list_.resize(type_ids.size());
  for (size_t i = 0; i < type_ids.size(); i++) {
    size_t type_size = GetTypeByte(TypeIdToType(type_ids[i]));
    size_t tensor_size = shapes[i].empty()
                           ? type_size
                           : std::accumulate(shapes[i].begin(), shapes[i].end(), type_size, std::multiplies<size_t>());
    tensor_size = std::max(tensor_size, type_size);
    output_size_list_[i] = tensor_size;
  }
}

}  // namespace kernel
}  // namespace mindspore
