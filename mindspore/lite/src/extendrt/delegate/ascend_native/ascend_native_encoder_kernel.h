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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_NATIVE_ENCODER_KERNEL_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_NATIVE_ENCODER_KERNEL_H_

#include <string>
#include <iostream>
#include <vector>
#include <memory>
#include "ascend_native_impl/encoder_ps.h"
#include "extendrt/delegate/ascend_native/ascend_native_base_kernel.h"
#include "extendrt/utils/func_graph_utils.h"

namespace mindspore::kernel {
class AscendNativeEncoderKernel : public AscendNativeBaseKernel {
 public:
  AscendNativeEncoderKernel(const std::vector<InferTensor *> &inputs, const std::vector<InferTensor *> &outputs,
                            InferPrimitive prim, const InferContext *ctx, const void *stream, std::string name,
                            const void *acl_ctx)
      : AscendNativeBaseKernel(inputs, outputs, prim, ctx, stream, name, acl_ctx),
        driver_input_tensors_(ENCODER_LAST_IDX),
        driver_output_tensors_(ENCODER_OUTPUT_LAST_IDX) {}

  virtual ~AscendNativeEncoderKernel() {
    if (param_.expert_to_tokens_) free(param_.expert_to_tokens_);
    param_.expert_to_tokens_ = nullptr;
    if (prompt_mask_) ascend_native::FreeDevice(prompt_mask_);
    prompt_mask_ = nullptr;
    ascend_native::pangu_encoder_delete<aclFloat16>(encoder_executer_);
    encoder_executer_ = nullptr;
  }

  int Prepare() override;

  int Run() override;

  int InferShape() override;

  size_t get_workspace_size() const override { return ws_size_; }

  int ReSize() override;

 private:
  void PrintParam();
  void build_driver_input_const_tensors();
  int InitEncoderParam();
  int InitEncoderInputs();
  int CreateMask();
  int TransposeProjW();
  int ZeroOutBias();
  int prepareOfflineEncoderWeight();
  std::vector<int32_t> getOutputDimensions();
  ascend_native::EncoderParams param_;
  void *encoder_executer_ = nullptr;
  size_t ws_size_ = 0;
  int mask_tensor_idx_ = -1;
  int expert_tensor_idx_ = -1;
  std::vector<void *> driver_input_tensors_;
  std::vector<void *> driver_output_tensors_;
  static void *prompt_mask_;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_NATIVE_ENCODER_KERNEL_H_
