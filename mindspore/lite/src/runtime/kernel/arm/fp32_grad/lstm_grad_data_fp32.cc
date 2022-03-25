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
#include "src/runtime/kernel/arm/fp32_grad/lstm_grad_data_fp32.h"
#include <string>
#include <memory>
#include <algorithm>
#include "utils/ms_utils.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "nnacl/fp32/lstm_fp32.h"

namespace mindspore {
namespace kernel {
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_LSTMGradData;

int LSTMGradDataCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_9D);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int LSTMGradDataCPUKernel::ReSize() { return InitParam(); }

int LSTMGradDataCPUKernel::Run() {
  auto ret = MallocRunBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LSTMGradDataCPUKernel MallocRunBuffer error.";
    FreeRunBuffer();
    return RET_ERROR;
  }

  // get input tensors
  auto dC_tensor = in_tensors_.at(dC_index);
  MS_ASSERT(dC_tensor != nullptr);
  auto dH_tensor = in_tensors_.at(dH_index);
  MS_ASSERT(dH_tensor != nullptr);
  auto dy_tensor = in_tensors_.at(dy_index);
  MS_ASSERT(dy_tensor != nullptr);
  auto weights_tensor = in_tensors_.at(weights_index);
  MS_ASSERT(weights_tensor != nullptr);
  auto intermediate_tensor = in_tensors_.at(intermediate_data_index);
  MS_ASSERT(intermediate_tensor != nullptr);
  auto cell_input_tensor = in_tensors_.at(cell_input_index);
  MS_ASSERT(cell_input_tensor != nullptr);
  auto dX_tensor = out_tensors_.at(dX_out_index);
  MS_ASSERT(dX_tensor != nullptr);
  auto dH_out_tensor = out_tensors_.at(dH_out_index);
  MS_ASSERT(dH_out_tensor != nullptr);
  auto dC_out_tensor = out_tensors_.at(dC_out_index);
  MS_ASSERT(dC_out_tensor != nullptr);

  // Get Tensors Data
  int time_stamp_len = lstm_param_->batch_ * lstm_param_->hidden_size_;

  weights_ = reinterpret_cast<float *>(weights_tensor->data());
  ReorderLstmWeightGrad(weights_tmp_, weights_);

  dC_ = reinterpret_cast<float *>(dC_tensor->data());
  dH_ = reinterpret_cast<float *>(dH_tensor->data());
  dX_ = reinterpret_cast<float *>(dX_tensor->data());
  memset(dH_, 0, dH_tensor->Size());
  memset(dC_, 0, dC_tensor->Size());
  memset(dX_, 0, dX_tensor->Size());

  int w_size = lstm_param_->hidden_size_ * lstm_param_->input_size_;
  int h_size = lstm_param_->hidden_size_ * lstm_param_->hidden_size_;

  float *orig_da = dA_tmp_;
  if (lstm_param_->bidirectional_) {
    // Adjust pointer to backward cell
    cell_input_data_ = reinterpret_cast<float *>(cell_input_tensor->data()) + time_stamp_len;
    intermediate_data_ = reinterpret_cast<float *>(intermediate_tensor->data()) + time_stamp_len;
    dC_ = reinterpret_cast<float *>(dC_tensor->data()) + time_stamp_len;
    dH_ = reinterpret_cast<float *>(dH_tensor->data()) + time_stamp_len;
    dY_ = reinterpret_cast<float *>(dy_tensor->data()) + lstm_param_->hidden_size_;
    dA_tmp_ = orig_da + lstm_param_->seq_len_ * num_of_gates * time_stamp_len;
    int w_offset = num_of_gates * (w_size + h_size);
    int v_offset = weight_batch_ * w_size + num_of_gates * h_size;
    float *w = weights_tmp_ + w_offset;
    float *v = weights_tmp_ + v_offset;
    LstmBackpropUnidirectional(true, w, v);
  }
  // adjust to forward cell
  cell_input_data_ = reinterpret_cast<float *>(cell_input_tensor->data());
  intermediate_data_ = reinterpret_cast<float *>(intermediate_tensor->data());
  dC_ = reinterpret_cast<float *>(dC_tensor->data());
  dH_ = reinterpret_cast<float *>(dH_tensor->data());
  dY_ = reinterpret_cast<float *>(dy_tensor->data());

  int w_offset = 0;
  int v_offset = num_of_gates * w_size;
  float *w = weights_tmp_ + w_offset;
  float *v = weights_tmp_ + v_offset;
  dA_tmp_ = orig_da;
  LstmBackpropUnidirectional(false, w, v);

  // setup output tensors
  dh_out_ = reinterpret_cast<float *>(dH_out_tensor->data());
  dc_out_ = reinterpret_cast<float *>(dC_out_tensor->data());
  std::copy(&(dH_[0]), &(dH_[dH_tensor->ElementsNum()]), &(dh_out_[0]));
  std::copy(&(dC_[0]), &(dC_[dC_tensor->ElementsNum()]), &(dc_out_[0]));

  auto seq_stride = lstm_param_->seq_len_ * lstm_param_->output_step_;
  float *cell_state = intermediate_data_ + seq_stride * 1;
  std::copy(&(dA_tmp_[0]), &(dA_tmp_[num_of_gates * seq_stride]), &(cell_state[0]));
  FreeRunBuffer();
  return RET_OK;
}

int LSTMGradDataCPUKernel::LstmBackpropUnidirectional(bool is_backward, float *w, float *v) {
  auto seq_stride = lstm_param_->seq_len_ * lstm_param_->output_step_;
  int state_len = lstm_param_->batch_ * lstm_param_->hidden_size_;
  float *cell_state = intermediate_data_ + seq_stride * 1;
  float *input_gate = intermediate_data_ + seq_stride * 2;
  float *output_gate = intermediate_data_ + seq_stride * 3;
  float *forget_gate = intermediate_data_ + seq_stride * 4;
  float *cell_gate = intermediate_data_ + seq_stride * 5;

  int dir_mult = lstm_param_->bidirectional_ ? 2 : 1;
  int prev_time_stamp_offset = (is_backward) ? 1 : -1;
  int first_time_stamp = (is_backward) ? lstm_param_->seq_len_ - 1 : 0;
  for (int t = lstm_param_->seq_len_ - 1; t >= 0; t--) {
    int real_t = is_backward ? lstm_param_->seq_len_ - t - 1 : t;
    auto stride = real_t * lstm_param_->output_step_;
    float *curr_cell_state = cell_state + stride;
    float *prev_cell_state = (real_t == first_time_stamp)
                               ? cell_input_data_
                               : cell_state + (real_t + prev_time_stamp_offset) * lstm_param_->output_step_;
    float *curr_input_gate = input_gate + stride;
    float *curr_forget_gate = forget_gate + stride;
    float *curr_cell_gate = cell_gate + stride;
    float *curr_output_gate = output_gate + stride;
    float *curr_dx = dX_ + real_t * lstm_param_->batch_ * lstm_param_->input_size_;
    int seq_offset = real_t * lstm_param_->output_step_;
    for (int b = 0; b < lstm_param_->batch_; b++) {
      int batch_offset = b * dir_mult * lstm_param_->hidden_size_;
      float *dy = dY_ + seq_offset + batch_offset;
      memcpy(curr_dy_ + b * lstm_param_->hidden_size_, dy, lstm_param_->hidden_size_ * sizeof(float));
    }
    float *dA = nullptr;
    LstmGradDoInputStep(curr_output_gate, curr_cell_state, prev_cell_state, curr_cell_gate, curr_input_gate,
                        curr_forget_gate, curr_dy_, dC_, dH_, &dA, curr_dx, w, v, workspace_, lstm_param_);
    float *dA_t = dA_tmp_ + real_t * num_of_gates * state_len;
    std::copy(&(dA[0]), &(dA[num_of_gates * state_len]), &dA_t[0]);  // for w grad step
  }
  return RET_OK;
}

void LSTMGradDataCPUKernel::ReorderLstmWeightGrad(float *dst, float *src) {
  int uni_batch = lstm_param_->bidirectional_ ? weight_batch_ / 2 : weight_batch_;
  ReorderLstmWeights(dst, src, uni_batch, lstm_param_->hidden_size_, lstm_param_->input_size_, getLstmOrderIFGO());
  src += uni_batch * lstm_param_->hidden_size_ * lstm_param_->input_size_;
  dst += uni_batch * lstm_param_->hidden_size_ * lstm_param_->input_size_;
  ReorderLstmWeights(dst, src, uni_batch, lstm_param_->hidden_size_, lstm_param_->hidden_size_, getLstmOrderIFGO());
  src += uni_batch * lstm_param_->hidden_size_ * lstm_param_->hidden_size_;
  dst += uni_batch * lstm_param_->hidden_size_ * lstm_param_->hidden_size_;
  if (lstm_param_->bidirectional_) {
    ReorderLstmWeights(dst, src, uni_batch, lstm_param_->hidden_size_, lstm_param_->input_size_, getLstmOrderIFGO());
    src += uni_batch * lstm_param_->hidden_size_ * lstm_param_->input_size_;
    dst += uni_batch * lstm_param_->hidden_size_ * lstm_param_->input_size_;
    ReorderLstmWeights(dst, src, uni_batch, lstm_param_->hidden_size_, lstm_param_->hidden_size_, getLstmOrderIFGO());
    src += uni_batch * lstm_param_->hidden_size_ * lstm_param_->hidden_size_;
    dst += uni_batch * lstm_param_->hidden_size_ * lstm_param_->hidden_size_;
  }
}

int LSTMGradDataCPUKernel::DoGrad(int thread_id) { return RET_OK; }

int LSTMGradDataCPUKernel::InitParam() {
  auto input = in_tensors_.front();
  MS_ASSERT(input != nullptr);
  std::vector<int> in_shape = input->shape();
  lstm_param_->seq_len_ = in_shape.at(FIRST_INPUT);
  lstm_param_->batch_ = in_shape.at(SECOND_INPUT);

  int dir_multiplier = lstm_param_->bidirectional_ ? 2 : 1;
  lstm_param_->output_step_ = dir_multiplier * lstm_param_->batch_ * lstm_param_->hidden_size_;
  weight_batch_ = dir_multiplier * num_of_gates;
  state_is_vec_ = lstm_param_->batch_ == 1;

#ifdef ENABLE_AVX
  row_tile_ = C6NUM;
  col_tile_ = C16NUM;
#elif defined(ENABLE_ARM32)
  row_tile_ = C12NUM;
  col_tile_ = C4NUM;
#elif defined(ENABLE_SSE)
  row_tile_ = C4NUM;
  col_tile_ = C8NUM;
#else
  row_tile_ = C12NUM;
  col_tile_ = C8NUM;
#endif
  lstm_param_->input_row_align_ = UP_ROUND(lstm_param_->seq_len_ * lstm_param_->batch_, row_tile_);
  lstm_param_->input_col_align_ = UP_ROUND(lstm_param_->hidden_size_, col_tile_);
  input_size_align_ = UP_ROUND(lstm_param_->input_size_, row_tile_);
  input_thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(lstm_param_->input_col_align_, col_tile_));
  input_thread_stride_ = UP_DIV(UP_DIV(lstm_param_->input_col_align_, col_tile_), input_thread_count_);

  state_row_tile_ = row_tile_;
  state_col_tile_ = col_tile_;

  lstm_param_->state_row_align_ = state_is_vec_ ? 1 : UP_ROUND(lstm_param_->batch_, state_row_tile_);
  lstm_param_->state_col_align_ =
    state_is_vec_ ? lstm_param_->hidden_size_ : UP_ROUND(lstm_param_->hidden_size_, state_col_tile_);

  return RET_OK;
}

int LSTMGradDataCPUKernel::MallocRunBuffer() {
  int workspace_size = GetRunWorkspaceSize(lstm_param_);
  if (workspace_size == 0) {
    MS_LOG(ERROR) << "LstmGradDataCPUKernel malloc run workspace 0 error.";
    return RET_ERROR;
  }
  workspace_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(workspace_size * sizeof(float)));
  if (workspace_ == nullptr) {
    MS_LOG(ERROR) << "LstmGradDataCPUKernel malloc run workspace error.";
    return RET_ERROR;
  }
  auto dA_size = num_of_gates * lstm_param_->output_step_ * lstm_param_->seq_len_;
  if (dA_size == 0) {
    MS_LOG(ERROR) << "LstmGradDataCPUKernel malloc run dA_tmp size error.";
    return RET_ERROR;
  }
  dA_tmp_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(dA_size * sizeof(float)));
  if (dA_tmp_ == nullptr) {
    MS_LOG(ERROR) << "LstmGradDataCPUKernel malloc run dA_tmp alloc error.";
    return RET_ERROR;
  }
  int weights_size = weight_batch_ * lstm_param_->hidden_size_ * lstm_param_->input_size_ +  // IW matrics
                     weight_batch_ * lstm_param_->hidden_size_ * lstm_param_->hidden_size_;  // V matrics
  weights_tmp_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(weights_size * sizeof(float)));
  if (weights_tmp_ == nullptr) {
    MS_LOG(ERROR) << "LstmGradWeightCPUKernel malloc run weights_tmp_ alloc error.";
    return RET_ERROR;
  }
  int curr_dy_size = lstm_param_->hidden_size_ * lstm_param_->batch_;
  curr_dy_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(curr_dy_size * sizeof(float)));
  if (curr_dy_ == nullptr) {
    MS_LOG(ERROR) << "LstmCPUKernel malloc run curr_dy_ alloc error.";
    return RET_ERROR;
  }
  return RET_OK;
}

void LSTMGradDataCPUKernel::FreeRunBuffer() {
  if (workspace_ != nullptr) {
    ms_context_->allocator->Free(workspace_);
    workspace_ = nullptr;
  }
  if (dA_tmp_ != nullptr) {
    ms_context_->allocator->Free(dA_tmp_);
    dA_tmp_ = nullptr;
  }
  if (weights_tmp_ != nullptr) {
    ms_context_->allocator->Free(weights_tmp_);
    weights_tmp_ = nullptr;
  }
  if (curr_dy_ != nullptr) {
    ms_context_->allocator->Free(curr_dy_);
    curr_dy_ = nullptr;
  }
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LSTMGradData, LiteKernelCreator<LSTMGradDataCPUKernel>)
}  // namespace kernel
}  // namespace mindspore
