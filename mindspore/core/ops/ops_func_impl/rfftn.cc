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

#include <set>
#include <memory>
#include <unordered_map>
#include "ops/op_utils.h"
#include "ops/ops_func_impl/fft_arithmetic.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/rfftn.h"

namespace mindspore {
namespace ops {
BaseShapePtr RFFTNFuncImpl::InferShape(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  return FFTNInferShape(primitive, input_args);
}

TypePtr RFFTNFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  return FFTInferType(primitive, input_args);
}

int32_t RFFTNFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  return FFTNCheckValidation(primitive, input_args);
}
}  // namespace ops
}  // namespace mindspore
