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

#include "ops/ops_func_impl/repeat_interleave_grad.h"
#include <utility>
#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
namespace {
inline ShapeVector GetInferredShape(const ShapeVector &input_shape, const ShapeVector &repeats, const int64_t &dim) {
  ShapeVector result_shape;
  auto rank = SizeToLong(input_shape.size());
  int64_t real_dim = dim;
  real_dim = (real_dim < 0) ? (real_dim + rank) : real_dim;
  for (int64_t dim_index = 0; dim_index < rank; dim_index++) {
    if (dim_index != real_dim) {
      result_shape.emplace_back(input_shape[dim_index]);
    } else {
      auto item = SizeToLong(repeats.size());
      if (item == 1) {
        if (repeats[0] == 0) {
          MS_EXCEPTION(RuntimeError) << "repeats must be not zero";
        }
        item = input_shape[dim_index] / repeats[0];
      }
      result_shape.emplace_back(item);
    }
  }
  return result_shape;
}
}  // namespace

BaseShapePtr RepeatInterleaveGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                      const std::vector<AbstractBasePtr> &input_args) const {
  auto x_base_shape = input_args[kInputIndex0]->GetShape();
  auto x_shape = x_base_shape->GetShapeVector();
  if (MS_UNLIKELY(IsDynamicRank(x_shape))) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeRankAny});
  }

  auto repeats_opt = GetArrayValue<int64_t>(input_args[kInputIndex1]);
  auto dim_opt = GetScalarValue<int64_t>(input_args[kInputIndex2]->GetValue());
  if (!repeats_opt.has_value() || !dim_opt.has_value()) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeRankAny});
  }

  auto repeats_values = repeats_opt.value();
  auto dim = dim_opt.value();

  ShapeVector repeats;
  for (size_t i = 0; i < repeats_values.size(); i++) {
    if (repeats_values.IsValueUnknown(i)) {
      repeats.push_back(abstract::Shape::kShapeDimAny);
    } else {
      if (repeats_values[i] < 0) {
        MS_EXCEPTION(RuntimeError) << "For '" << primitive->name() << "', 'repeats' can not be negative.";
      }
      repeats.push_back(repeats_values[i]);
    }
  }

  auto inferred_shape = GetInferredShape(x_shape, repeats, dim);
  return std::make_shared<abstract::TensorShape>(inferred_shape);
}

TypePtr RepeatInterleaveGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetType();
}

ShapeArray RepeatInterleaveGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                    const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto x_shape = x_tensor->shape();

  auto repeats_opt = GetArrayValue<int64_t>(input_values[kInputIndex1]);
  auto dim_opt = GetScalarValue<int64_t>(input_values[kInputIndex2]);

  auto repeats_values = repeats_opt.value();
  auto dim = dim_opt.value();

  ShapeVector repeats;
  for (size_t i = 0; i < repeats_values.size(); i++) {
    if (repeats_values.IsValueUnknown(i)) {
      repeats.push_back(abstract::Shape::kShapeDimAny);
    } else {
      if (repeats_values[i] < 0) {
        MS_EXCEPTION(RuntimeError) << "For '" << primitive->name() << "', 'repeats' can not be negative.";
      }
      repeats.push_back(repeats_values[i]);
    }
  }

  auto inferred_shape = GetInferredShape(x_shape, repeats, dim);
  return ShapeArray{
    inferred_shape,
  };
}

TypePtrList RepeatInterleaveGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                                    const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &input_x_type = x_tensor->Dtype();
  TypePtrList type_ptr_list{input_x_type};
  return type_ptr_list;
}
REGISTER_SIMPLE_INFER(kNameRepeatInterleaveGrad, RepeatInterleaveGradFuncImpl)
}  // namespace ops
}  // namespace mindspore
