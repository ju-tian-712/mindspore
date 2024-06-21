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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAY_SEARCH_SORTED_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAY_SEARCH_SORTED_GPU_KERNEL_H_
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>
#include <numeric>
#include <map>
#include "abstract/utils.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/search_sorted_impl.cuh"

namespace mindspore {
namespace kernel {
class SearchSortedGpuKernelMod : public NativeGpuKernelMod {
 public:
  SearchSortedGpuKernelMod() { ResetResource(); }
  ~SearchSortedGpuKernelMod() override = default;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    if (is_null_input_) {
      return true;
    }
    return kernel_func_(this, inputs, workspace, outputs, cuda_stream);
  }
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  void ResetResource() noexcept {
    sequence_size_ = 0;
    value_size_ = 0;
    status1 = 0;
    count12 = 0;
    output_elements_ = 0;
    is_null_input_ = false;
    sequence_shape_.clear();
    value_shape_.clear();
    output_shape_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename S, typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);
  using SearchSortedFunc = std::function<bool(SearchSortedGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                                              const std::vector<kernel::KernelTensor *> &,
                                              const std::vector<kernel::KernelTensor *> &, void *)>;
  template <typename S, typename T>
  void CheckParam(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);

 private:
  bool right{false};
  size_t unit_output_size_{1};
  size_t sequence_per_size_{1};
  size_t value_per_size_{1};
  size_t sequence_size_ = 0;
  size_t value_size_ = 0;
  size_t output_elements_ = 0;
  int status1 = 0;
  int count12 = 0;
  bool should_last_repeat_ = True;
  std::vector<int64_t> sequence_shape_;
  std::vector<int64_t> value_shape_;
  std::vector<int64_t> output_shape_;
  bool is_null_input_{false};
  SearchSortedFunc kernel_func_{};
  cudaStream_t cuda_stream_;

  static std::vector<std::pair<KernelAttr, SearchSortedFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAY_SEARCH_SORTED_GPU_KERNEL_H_
