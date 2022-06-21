/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/math/xdivy_gpu_kernel.h"

namespace mindspore {
namespace kernel {
bool XdivyGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                               const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (kernel_func_ == nullptr) {
    MS_LOG(EXCEPTION) << "Please call init before call Launch and make sure init success ...";
  }
  return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
}

bool XdivyGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), INPUT_NUM, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), OUTPUT_NUM, kernel_name_);
  auto x_type = inputs[0]->GetDtype();
  auto y_type = inputs[1]->GetDtype();
  auto out_type = outputs[0]->GetDtype();
  if (!(x_type == y_type && x_type == out_type)) {
    MS_LOG(ERROR) << "Xdivy need same input and output data type, but got X type:" << x_type << " Y type:" << y_type
                  << " out type:" << out_type;
    return false;
  }

  auto iter = func_map_.find(x_type);
  if (iter == func_map_.end()) {
    MS_LOG(ERROR) << "Xdivy only support tensor with data type float16, float32, "
                     "float64, Complex64, Complex128, but got typeid:"
                  << x_type;
    return false;
  }
  kernel_func_ = iter->second;
  return true;
}

int XdivyGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs,
                              const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), INPUT_NUM, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), OUTPUT_NUM, kernel_name_);
  ResetResource();
  int ret = NativeGpuKernelMod::Resize(base_operator, inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  std::vector<size_t> x_shape, y_shape, out_shape;
  auto in_shape_0 = inputs[0]->GetShapeVector();
  x_shape.assign(in_shape_0.begin(), in_shape_0.end());
  auto in_shape_1 = inputs[1]->GetShapeVector();
  y_shape.assign(in_shape_1.begin(), in_shape_1.end());
  auto out_shape_0 = outputs[0]->GetShapeVector();
  out_shape.assign(out_shape_0.begin(), out_shape_0.end());

  need_broadcast_ = common::AnfAlgo::IsTensorBroadcast(x_shape, y_shape);
  if (out_shape.size() > MAX_DIMS || out_shape.size() < x_shape.size() || out_shape.size() < y_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be greater than " << MAX_DIMS
                      << ", and output dimension can't less than input; but got x_shape dimension:" << x_shape.size()
                      << " ,y_shape dimension:" << y_shape.size() << " ,out_shape dimension:" << out_shape.size();
  }

  lhs_shape_.resize(MAX_DIMS, 1);
  rhs_shape_.resize(MAX_DIMS, 1);
  output_shape_.resize(MAX_DIMS, 1);
  for (size_t i = 0; i < out_shape.size(); i++) {
    if (need_broadcast_) {
      output_shape_[i] = out_shape[i];
    }
    out_ele_num_ *= out_shape[i];
  }
  int lhs_offset = out_shape.size() - x_shape.size();
  for (size_t j = 0; j < x_shape.size(); j++) {
    if (need_broadcast_) {
      lhs_shape_[j + lhs_offset] = x_shape[j];
    }
    x_ele_num_ *= x_shape[j];
  }
  int rhs_offset = out_shape.size() - y_shape.size();
  for (size_t k = 0; k < y_shape.size(); k++) {
    if (need_broadcast_) {
      rhs_shape_[k + rhs_offset] = y_shape[k];
    }
    y_ele_num_ *= y_shape[k];
  }
  return KRET_OK;
}

std::vector<KernelAttr> XdivyGpuKernelMod::support_ops_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16)},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64)},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex64)
     .AddInputAttr(kNumberTypeComplex64)
     .AddOutputAttr(kNumberTypeComplex64)},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex128)
     .AddInputAttr(kNumberTypeComplex128)
     .AddOutputAttr(kNumberTypeComplex128)},
};
std::vector<KernelAttr> XdivyGpuKernelMod::GetOpSupport() { return support_ops_; }

template <typename T>
using Complex = mindspore::utils::Complex<T>;
std::map<mindspore::TypeId, XdivyGpuKernelMod::XdivyFunc> XdivyGpuKernelMod::func_map_ = {
  {kNumberTypeFloat16, &XdivyGpuKernelMod::LaunchKernel<half>},
  {kNumberTypeFloat32, &XdivyGpuKernelMod::LaunchKernel<float>},
  {kNumberTypeFloat64, &XdivyGpuKernelMod::LaunchKernel<double>},
  {kNumberTypeComplex64, &XdivyGpuKernelMod::LaunchKernelComplex<Complex<float>>},
  {kNumberTypeComplex128, &XdivyGpuKernelMod::LaunchKernelComplex<Complex<double>>}};

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Xdivy, XdivyGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
