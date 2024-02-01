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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_ASCEND910B_DVPP_CROP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_ASCEND910B_DVPP_CROP_H_

#include <memory>
#include <vector>
#include <string>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/device_tensor_ascend910b.h"
#include "minddata/dataset/kernels/tensor_op.h"

namespace mindspore {
namespace dataset {
class DvppCropOp : public TensorOp {
 public:
  DvppCropOp(int32_t top, int32_t left, int32_t height, int32_t width)
      : top_(top), left_(left), height_(height), width_(width) {}

  ~DvppCropOp() override = default;

  Status Compute(const std::shared_ptr<DeviceTensorAscend910B> &input,
                 std::shared_ptr<DeviceTensorAscend910B> *output) override;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  std::string Name() const override { return kDvppCropOp; }

  bool IsDvppOp() override { return true; }

 private:
  int32_t top_;
  int32_t left_;
  int32_t height_;
  int32_t width_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_ASCEND910B_DVPP_CROP_H_
