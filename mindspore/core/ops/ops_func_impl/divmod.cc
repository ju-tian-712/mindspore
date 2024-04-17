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

#include "ops/ops_func_impl/divmod.h"
#include <set>
#include <map>
#include <limits>
#include <string>
#include <utility>
#include "utils/check_convert_utils.h"
#include "ops/op_enum.h"
#include "abstract/dshape.h"
#include "ops/op_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
BaseShapePtr DivModFuncImpl::InferShape(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  return BroadCastInferShape(primitive->name(), input_args);
}

TypePtr DivModFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto x_dtype = input_args[kIndex0]->GetType();
  auto y_dtype = input_args[kIndex1]->GetType();

  auto mode = input_args[kIndex2]->GetValue();
  auto rounding_mode = GetScalarValue<int64_t>(mode);

  if (rounding_mode == RoundingMode::TRUNC || rounding_mode == RoundingMode::FLOOR) {
    return input_args[0]->GetType()->Clone();
  } else {
    static std::set<int> x_set = {kNumberTypeUInt8, kNumberTypeInt8, kNumberTypeInt16, kNumberTypeInt32,
                                  kNumberTypeInt64};
    static std::set<int> integral_set = {kNumberTypeUInt8, kNumberTypeInt8, kNumberTypeInt16, kNumberTypeInt32,
                                         kNumberTypeInt64};
    auto x_tensor_type = x_dtype->cast<TensorTypePtr>();
    auto y_tensor_type = y_dtype->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(x_tensor_type);
    MS_EXCEPTION_IF_NULL(y_tensor_type);
    auto x_type_id = x_tensor_type->element()->type_id();
    auto y_type_id = y_tensor_type->element()->type_id();
    if ((x_type_id == kNumberTypeFloat32 && integral_set.find(y_type_id) != integral_set.end()) ||
        (integral_set.find(x_type_id) != integral_set.end() && integral_set.find(y_type_id) != integral_set.end())) {
      return kFloat32;
    }
    std::map<std::string, TypePtr> types;
    (void)types.emplace("x", x_dtype);
    (void)types.emplace("y", y_dtype);
    return CheckAndConvertUtils::CheckMathBinaryOpTensorType(types, common_valid_types_with_complex, prim_name);
  }
}
}  // namespace ops
}  // namespace mindspore
