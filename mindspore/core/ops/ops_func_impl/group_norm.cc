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
#include "ops/ops_func_impl/group_norm.h"
#include <memory>
#include "abstract/dshape.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
constexpr int64_t kNumberTwo = 2;
constexpr int64_t kNumberEight = 8;
BaseShapePtr GroupNormFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  // Get input tensor shape.
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto x_shape_ptr = input_args[kInputIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(x_shape_ptr);
  const auto &x_shape = x_shape_ptr->GetShapeVector();
  const auto &weight_shape = input_args[kInputIndex2]->GetShape()->GetShapeVector();
  const auto &bias_shape = input_args[kInputIndex3]->GetShape()->GetShapeVector();
  auto num_groups_opt = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());
  std::vector<BaseShapePtr> shapes_list;
  if (!num_groups_opt.has_value() || IsDynamicRank(x_shape)) {
    auto any_shape =
      std::make_shared<abstract::TensorShape>(std::vector<int64_t>{abstract::TensorShape::kShapeRankAny});
    shapes_list = {any_shape, any_shape, any_shape};
    return std::make_shared<abstract::TupleShape>(shapes_list);
  }
  int64_t num_groups = num_groups_opt.value();
  const auto x_rank = x_shape.size();
  if (x_rank < kNumberTwo || x_rank > kNumberEight) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name()
                      << "', The dim of input must be between 2 and 8. But got: " << x_rank << ".";
  }
  if (weight_shape.size() == 0 || bias_shape.size() == 0) {
    MS_EXCEPTION(TypeError) << "For " << primitive->name()
                            << ", the weight and bias must be a tensor, but got a number.";
  }
  MS_CHECK_VALUE(weight_shape.size() == 1, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                             "rank of weight", SizeToLong(weight_shape.size()), kEqual, 1, primitive));
  MS_CHECK_VALUE(bias_shape.size() == 1, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                           "rank of bias", SizeToLong(bias_shape.size()), kEqual, 1, primitive));
  if (MS_LIKELY(!(IsDynamic(weight_shape) || IsDynamic(bias_shape)))) {
    MS_CHECK_VALUE(bias_shape == weight_shape, CheckAndConvertUtils::FormatCheckMsg("weight and bias", weight_shape,
                                                                                    kEqual, bias_shape, primitive));
  }
  const int64_t N = x_shape[0];
  const int64_t channel = x_shape[1];
  if (!IsDynamic(x_shape) && !IsDynamic(weight_shape) && MS_UNLIKELY(weight_shape[kInputIndex0] != channel)) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name()
                             << ", shape of weight and bias should be equal to input_x's channel dimension: " << channel
                             << ", bug got shape: " << weight_shape << ".";
  }
  ShapeVector out_shape{N, num_groups};
  (void)shapes_list.emplace_back(x_shape_ptr->Clone());
  (void)shapes_list.emplace_back(std::make_shared<abstract::TensorShape>(out_shape));
  (void)shapes_list.emplace_back(std::make_shared<abstract::TensorShape>(out_shape));
  return std::make_shared<abstract::TupleShape>(shapes_list);
}

TypePtr GroupNormFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto x_type = input_args[kInputIndex0]->GetType();
  MS_EXCEPTION_IF_NULL(x_type);
  std::vector<TypePtr> types_list;
  types_list = {x_type, x_type, x_type};
  return std::make_shared<Tuple>(types_list);
}

}  // namespace ops
}  // namespace mindspore
