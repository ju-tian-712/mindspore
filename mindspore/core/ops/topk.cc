/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "ops/topk.h"
#include <set>
#include <utility>
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/array_ops.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(TopK, BaseOperator);
void TopK::Init(const bool sorted) { this->set_sorted(sorted); }
void TopK::set_sorted(const bool sorted) { (void)this->AddAttr(kSorted, api::MakeValue(sorted)); }

bool TopK::get_sorted() const {
  auto value_ptr = this->GetAttr(kSorted);
  return GetValue<bool>(value_ptr);
}

namespace {
abstract::TupleShapePtr TopKInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape());
  auto x_shape = shape_map[kShape];
  if (IsDynamicRank(x_shape)) {
    abstract::BaseShapePtr out_shape_ptr =
      std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{out_shape_ptr, out_shape_ptr});
  }
  int64_t k_v = 0;
  auto input1_value = input_args[kInputIndex1]->BuildValue();
  if ((IsDynamicRank(x_shape)) || !IsValueKnown(input1_value)) {
    auto unknown_shape_p = std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{unknown_shape_p, unknown_shape_p});
  }

  // 2rd input is a Tensor when TopK is a dynamic shape operator
  if (input_args[kInputIndex1]->isa<abstract::AbstractTensor>()) {
    MS_EXCEPTION_IF_NULL(input1_value);
    if (input1_value->isa<tensor::Tensor>()) {
      auto k_dim =
        CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape].size();
      if (k_dim > 1) {
        MS_LOG(EXCEPTION) << "For '" << prim_name
                          << "', the dimension of 'k' should only be 0 or 1 when 'k' is a Tensor, but got: " << k_dim
                          << ".";
      }
      auto k_tensor_ptr = input1_value->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(k_tensor_ptr);
      k_v = *static_cast<int32_t *>(k_tensor_ptr->data_c());
    }
  } else if (input_args[kInputIndex1]->isa<abstract::AbstractScalar>()) {
    k_v = GetValue<int64_t>(input1_value);
  } else {
    MS_LOG(EXCEPTION) << "Invalid abstract type:" << input_args[kInputIndex1]->type_name();
  }
  if (!x_shape.empty()) {
    auto ndims = x_shape.size() - 1;
    if (x_shape[ndims] != abstract::Shape::kShapeDimAny) {
      std::pair<int64_t, int64_t> k_range(0, x_shape[ndims]);
      CheckAndConvertUtils::CheckInRange<int64_t>("k", k_v, kIncludeRight, k_range, prim_name);
      x_shape[ndims] = k_v;
    }
  }

  auto out_shape_ptr = std::make_shared<abstract::Shape>(x_shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{out_shape_ptr, out_shape_ptr});
}

TuplePtr TopKInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto output0_type = input_args[kInputIndex0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_x", output0_type, common_valid_types, prim_name);
  auto k_type = input_args[kInputIndex1]->BuildType();
  const std::set<TypePtr> int_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTypeValid("k", k_type, int_types, prim_name);
  auto output1_type = kInt32;
  return std::make_shared<Tuple>(std::vector<TypePtr>{output0_type, output1_type});
}
}  // namespace

AbstractBasePtr TopKInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  auto infer_type = TopKInferType(primitive, input_args);
  auto infer_shape = TopKInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

bool TopK::get_attr(const char *attr) const {
  auto attr_ptr = GetAttr(attr);
  return GetValue<bool>(attr_ptr);
}

// AG means auto generated
class MIND_API AGTopKInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return TopKInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return TopKInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return TopKInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {1}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(TopK, prim::kPrimTopK, AGTopKInfer, false);
}  // namespace ops
}  // namespace mindspore
