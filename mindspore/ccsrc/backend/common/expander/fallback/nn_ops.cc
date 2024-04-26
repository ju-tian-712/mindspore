/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#include "backend/common/expander/fallback/fallback_irbuilder.h"
#include "include/common/utils/utils.h"
#include "utils/shape_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/base/types.h"
#include "ops/op_utils.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ops/renorm.h"
#include "ops/scatter_update.h"

namespace mindspore {
namespace expander {
namespace {
bool IsLastAxis(const ShapeVector &shape, int64_t axis) {
  if (axis == -1) {
    return true;
  }
  if (IsDynamicRank(shape)) {
    return false;
  }
  auto rank = SizeToLong(shape.size());
  if (axis < 0) {
    axis += rank;
  }
  return (axis == (rank - 1));
}

std::vector<int64_t> GetTransposeAxis(const std::vector<int64_t> &x_shape, int64_t axis) {
  std::vector<int64_t> reverse_axis;
  if (x_shape.empty()) {
    return reverse_axis;
  }
  auto rk = static_cast<int64_t>(x_shape.size());
  if (axis < 0) {
    axis += rk;
  }
  reverse_axis.reserve(x_shape.size());
  for (int64_t i = 0; i < rk; ++i) {
    (void)reverse_axis.emplace_back(i);
  }
  reverse_axis[LongToSize(axis)] = rk - 1;
  reverse_axis[LongToSize(rk - 1)] = axis;
  return reverse_axis;
}
}  // namespace

REG_FALLBACK_BUILDER("ArgMaxExt").SetBody(BODYFUNC(ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto dim = ib->GetInput(kIndex1);
  auto keepdim = ib->GetInput(kIndex2);
  if (input_x->dtype() == kBool) {
    input_x = ib->Cast(input_x, kInt32);
  }
  bool is_dim_none = True;
  auto dim_value = ib->Value(0);
  auto dim_value_ptr = dim->BuildValue();
  if (dim_value_ptr->isa<None>()) {
    input_x = ib->Reshape(input_x, {-1});
  } else {
    dim_value = dim;
    is_dim_none = False;
  }
  auto res = ib->Emit("Argmax", {input_x, dim_value, ib->Value<int64_t>(kInt64->type_id())});
  auto keepdim_value = ops::GetScalarValue<bool>(keepdim->BuildValue());
  if (!keepdim_value.has_value()) {
    auto true_case = [&res, &dim](Emitter *e) -> NodePtrList { return {e->Emit("ExpandDims", {res, dim})}; };
    auto false_case = [&res](Emitter *e) -> NodePtrList { return {res}; };
    if (!is_dim_none) {
      res = ib->Conditional(keepdim, true_case, false_case);
    }
  } else {
    if (keepdim_value.value() && !is_dim_none) {
      res = ib->Emit("ExpandDims", {res, dim});
    }
  }
  return {res};
});

REG_FALLBACK_BUILDER("SiLU").SetBody(BODYFUNC(ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto s = ib->Emit("Sigmoid", {ib->GetInput(kIndex0)});
  return {ib->Mul(input_x, s)};
});

REG_FALLBACK_BUILDER("SiLUGrad").SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex0);
  auto x = ib->GetInput(kIndex1);
  auto sigmoid_input = ib->Emit("Sigmoid", {x});
  auto bc_dx = ib->Mul(x, dout);
  auto bc_dy = ib->Mul(sigmoid_input, dout);
  auto dx = ib->Emit("SigmoidGrad", {sigmoid_input, bc_dx});
  return {ib->Add(dx, bc_dy)};
});

REG_FALLBACK_BUILDER("Baddbmm").SetBody(BODYFUNC(ib) {
  // baddbmm equation: output = beta * input + alpha * matmul(batch1, batch2)
  auto input = ib->GetInput(kIndex0);
  auto batch1 = ib->GetInput(kIndex1);
  auto batch2 = ib->GetInput(kIndex2);
  auto beta = ib->GetInput(kIndex3);
  auto alpha = ib->GetInput(kIndex4);

  auto mm_output = ib->BatchMatMul(batch1, batch2);
  auto alpha_output = ib->Mul(mm_output, alpha);
  auto beta_output = ib->Mul(input, beta);
  return {ib->Add(beta_output, alpha_output)};
});

DEF_PURE_SHAPE_CALC(g_dense_shapecalc)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto &x_shape = inputs.at(kIndex0);
    auto &w_shape = inputs.at(kIndex1);
    ShapeVector reshape_x_shape = {-1, x_shape.back()};
    ShapeVector reshape_w_shape = {-1, w_shape.back()};
    ShapeVector reshape_ret_shape;
    bool is_empty = std::any_of(x_shape.begin(), x_shape.end(), [](const int64_t shape) { return shape == 0; });
    if (is_empty) {
      reshape_x_shape[0] = 1;
      reshape_w_shape[0] = 1;
      return {reshape_x_shape, reshape_w_shape, reshape_ret_shape};
    }
    reshape_ret_shape = x_shape;
    if (w_shape.size() == 1) {
      reshape_ret_shape.erase(reshape_ret_shape.end() - 1);
    } else {
      reshape_ret_shape.back() = -1;
    }
    return {reshape_x_shape, reshape_w_shape, reshape_ret_shape};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> {
    constexpr const int64_t kRank2 = 2;
    int64_t ret_size = -1LL;
    if (!IsDynamicRank(inputs[0]) && !IsDynamicRank(inputs[1])) {
      if (inputs[0].size() == 1) {
        if (inputs[1].size() == 1) {
          ret_size = 0;
        } else {
          ret_size = 1;
        }
      } else {
        ret_size = inputs[0].size();
      }
    }
    return {kRank2, kRank2, ret_size};
  });

REG_FALLBACK_BUILDER("Dense").SetBody(BODYFUNC(ib) {
  constexpr const size_t kRank2 = 2;
  auto x = ib->GetInput(kIndex0);
  auto w = ib->GetInput(kIndex1);
  NodePtrList reshape_shapes;
  auto x_shape = x->shape();
  auto w_shape = w->shape();
  bool is_empty_tensor = x_shape.size() == 1 && w_shape.size() == 1 && x_shape[0] == 0 && w_shape[0] == 0;
  if (is_empty_tensor) {
    return {ib->GetInput(kIndex2)};
  }
  bool is_dynamic_rank = IsDynamicRank(x_shape) || IsDynamicRank(w_shape);
  bool need_reshape = (is_dynamic_rank || x_shape.size() != kRank2 || w_shape.size() != kRank2);
  if (need_reshape) {
    reshape_shapes = ib->ShapeCalc(g_dense_shapecalc, {x, w});
    x = ib->Reshape(x, reshape_shapes[kIndex0]);
    w = ib->Reshape(w, reshape_shapes[kIndex1]);
  }
  auto ret = ib->MatMul(x, w, false, true);
  ret = ib->Cast(ret, x->dtype());
  auto b = ib->GetInput(kIndex2);
  auto b_value = b->BuildValue();
  if (!b_value->isa<None>()) {
    ret = ib->Add(ret, b);
  }
  if (need_reshape) {
    ret = ib->Reshape(ret, reshape_shapes[kIndex2]);
  }
  return {ret};
});

class SoftmaxShapeCalc : public ShapeCalcFunctor {
 public:
  SoftmaxShapeCalc() : ShapeCalcFunctor("ShapeCalc_Softmax") {}
  ~SoftmaxShapeCalc() override = default;
  MS_DECLARE_PARENT(SoftmaxShapeCalc, ShapeCalcFunctor)

  ValuePtr ToValue() const override { return nullptr; }
  void FromValue(const ValuePtr &value) override {}
  ShapeArray Calc(const ShapeArray &inputs) const override {
    // inputs: {dout_shape, dim}
    auto dim = inputs.at(1)[0];
    return {GetTransposeAxis(inputs.at(0), dim)};
  }
  std::vector<int64_t> Infer(const ShapeArray &inputs, const HashSet<size_t> &) const override {
    int64_t dout_rank = IsDynamicRank(inputs.at(0)) ? -1 : SizeToLong(inputs.at(0).size());
    return {dout_rank};
  }
};
REG_FUNCTOR("ShapeCalc_Softmax", SoftmaxShapeCalc);

REG_FALLBACK_BUILDER("SoftmaxBackward").SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dim = ib->GetInput(kIndex2);

  auto shp = out->shape();
  auto dim_value_ptr = dim->BuildValue();
  int64_t dim_value{0};
  bool success = false;
  if (!(dim_value_ptr->isa<ValueAny>() || dim_value_ptr->isa<None>())) {
    dim_value = GetValue<int64_t>(dim_value_ptr);
    success = true;
  }
  if (success && IsLastAxis(shp, dim_value)) {
    auto dx = ib->Mul(out, ib->Sub(dout, ib->ReduceSum(ib->Mul(out, dout), ShapeVector{-1}, true)));
    return {dx};
  }
  auto reverse_axis = (IsDynamicRank(shp) || !success)
                        ? ib->ShapeCalc(std::make_shared<SoftmaxShapeCalc>(), {dout, dim}, {1})[0]
                        : ib->Value(GetTransposeAxis(shp, dim_value));
  out = ib->Transpose(out, reverse_axis);
  dout = ib->Transpose(dout, reverse_axis);
  auto dx = ib->Mul(out, ib->Sub(dout, ib->ReduceSum(ib->Mul(out, dout), ShapeVector{-1}, true)));
  dx = ib->Transpose(dx, reverse_axis);
  return {dx};
});

REG_FALLBACK_BUILDER("LayerNormGradExt").SetBody(BODYFUNC(ib) {
  auto dy = ib->GetInput(kIndex0);
  auto x = ib->GetInput(kIndex1);
  auto mean = ib->GetInput(kIndex3);
  auto rstd = ib->GetInput(kIndex4);
  auto gamma = ib->GetInput(kIndex5);
  return {ib->Emit("LayerNormGradV3", {dy, x, rstd, mean, gamma})};
});

class UpsampleNearest1DShapeCalc : public ShapeCalcFunctor {
 public:
  UpsampleNearest1DShapeCalc() : ShapeCalcFunctor("ShapeCalc_UpsampleNearest1D") {}
  ~UpsampleNearest1DShapeCalc() override = default;
  MS_DECLARE_PARENT(UpsampleNearest1DShapeCalc, ShapeCalcFunctor)

  ValuePtr ToValue() const override { return nullptr; }
  void FromValue(const ValuePtr &value) override {}
  ShapeArray Calc(const ShapeArray &inputs) const override {
    // inputs: {output_size}
    auto output_size = inputs.at(0);
    std::vector<int64_t> size{output_size.at(0), 1};
    return {size};
  }
  std::vector<int64_t> Infer(const ShapeArray &inputs, const HashSet<size_t> &) const override { return {2}; }
};
REG_FUNCTOR("ShapeCalc_UpsampleNearest1D", UpsampleNearest1DShapeCalc);

REG_FALLBACK_BUILDER("UpsampleNearest1D").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto output_size = ib->GetInput(kIndex1);
  auto scales = ib->GetInput(kIndex2);

  NodePtr size_node;
  if (output_size->abstract()->BuildType()->isa<TypeNone>()) {
    auto value_ptr = scales->BuildValue();
    auto x_shape = x->shape();
    if (!IsDynamicShape(x_shape) && ops::IsValueKnown(value_ptr)) {
      auto scales = GetValue<std::vector<pyfloat>>(value_ptr);
      std::vector<int64_t> size{static_cast<int64_t>(x_shape[2] * scales[0]), 1};
      size_node = ib->Value(size);
    } else {
      MS_LOG(ERROR) << "For UpsampleNearest1D, x should not be dynamic shape and scales should be const.";
      size_node = ib->Value(std::vector<int64_t>{1, 1});
    }
  } else {
    auto value_ptr = output_size->BuildValue();
    if (ops::IsValueKnown(value_ptr)) {
      auto output_size_val = GetValue<std::vector<int64_t>>(value_ptr);
      std::vector<int64_t> size{output_size_val.at(0), 1};
      size_node = ib->Value(size);
    } else {
      size_node = ib->ShapeCalc(std::make_shared<UpsampleNearest1DShapeCalc>(), {output_size}, {0})[0];
    }
  }

  auto new_x = ib->ExpandDims(x, -1);
  auto out = ib->Emit("ResizeNearestNeighborV2", {new_x, size_node, ib->Value(false), ib->Value(false)});

  auto real_out = ib->Squeeze(out, MakeValue(std::vector<int64_t>{-1}));

  return {real_out};
});

class UpsampleNearest2DShapeCalc : public ShapeCalcFunctor {
 public:
  UpsampleNearest2DShapeCalc() : ShapeCalcFunctor("ShapeCalc_UpsampleNearest2D") {}
  ~UpsampleNearest2DShapeCalc() override = default;
  MS_DECLARE_PARENT(UpsampleNearest2DShapeCalc, ShapeCalcFunctor)

  ValuePtr ToValue() const override { return nullptr; }
  void FromValue(const ValuePtr &value) override {}
  ShapeArray Calc(const ShapeArray &inputs) const override {
    // inputs: {output_size}
    auto output_size = inputs.at(0);
    std::vector<int64_t> size(2, 1);
    std::transform(output_size.begin(), std::min(output_size.begin() + kIndex2, output_size.end()), size.begin(),
                   [](int64_t v) { return v; });
    return {size};
  }
  std::vector<int64_t> Infer(const ShapeArray &inputs, const HashSet<size_t> &) const override { return {2}; }
};
REG_FUNCTOR("ShapeCalc_UpsampleNearest2D", UpsampleNearest2DShapeCalc);

REG_FALLBACK_BUILDER("UpsampleNearest2D").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto output_size = ib->GetInput(kIndex1);
  auto scales = ib->GetInput(kIndex2);

  NodePtr size_node;
  auto x_shape = x->shape();
  if (output_size->abstract()->BuildType()->isa<TypeNone>()) {
    auto value_ptr = scales->BuildValue();
    if (!IsDynamicShape(x_shape) && ops::IsValueKnown(value_ptr)) {
      auto scales = GetValue<std::vector<pyfloat>>(value_ptr);
      std::vector<int64_t> size{static_cast<int64_t>(x_shape[2] * scales[0]),
                                static_cast<int64_t>(x_shape[3] * scales[1])};
      size_node = ib->Value(size);
    } else {
      MS_LOG(ERROR) << "For UpsampleNearest2D, x should not be dynamic and scales should be const.";
      size_node = ib->Value(std::vector<int64_t>{1, 1});
    }
  } else {
    if (IsDynamicRank(x_shape)) {
      auto value_ptr = output_size->BuildValue();
      if (ops::IsValueKnown(value_ptr)) {
        auto output_size_val = GetValue<std::vector<int64_t>>(value_ptr);
        std::vector<int64_t> size(2, 1);
        std::transform(output_size_val.begin(), std::min(output_size_val.begin() + kIndex2, output_size_val.end()),
                       size.begin(), [](int64_t v) { return v; });
        size_node = ib->Value(size);
      } else {
        size_node = ib->ShapeCalc(std::make_shared<UpsampleNearest2DShapeCalc>(), {output_size}, {0})[0];
      }
    } else {
      size_node = output_size;
    }
  }

  auto out = ib->Emit("ResizeNearestNeighborV2", {x, size_node, ib->Value(false), ib->Value(false)});

  return {out};
});

class UpsampleNearest1DGradShapeCalc : public ShapeCalcFunctor {
 public:
  UpsampleNearest1DGradShapeCalc() : ShapeCalcFunctor("ShapeCalc_UpsampleNearest1DGrad") {}
  ~UpsampleNearest1DGradShapeCalc() override = default;
  MS_DECLARE_PARENT(UpsampleNearest1DGradShapeCalc, ShapeCalcFunctor)

  ValuePtr ToValue() const override { return nullptr; }
  void FromValue(const ValuePtr &value) override {}
  ShapeArray Calc(const ShapeArray &inputs) const override {
    // inputs: {input_size}
    auto input_size = inputs.at(0);
    std::vector<int64_t> size{input_size[input_size.size() - 1], 1};
    return {size};
  }
  std::vector<int64_t> Infer(const ShapeArray &inputs, const HashSet<size_t> &) const override { return {2}; }
};
REG_FUNCTOR("ShapeCalc_UpsampleNearest1DGrad", UpsampleNearest1DGradShapeCalc);

REG_FALLBACK_BUILDER("UpsampleNearest1DGrad").SetBody(BODYFUNC(ib) {
  auto grad = ib->GetInput(kIndex0);
  auto input_size = ib->GetInput(kIndex1);
  auto output_size = ib->GetInput(kIndex2);
  auto scales = ib->GetInput(kIndex3);

  auto value_ptr = input_size->BuildValue();
  NodePtr size_node;
  if (ops::IsValueKnown(value_ptr)) {
    auto input_size_val = GetValue<std::vector<int64_t>>(value_ptr);
    std::vector<int64_t> size{input_size_val[input_size_val.size() - 1], 1};
    size_node = ib->Value(size);
  } else {
    size_node = ib->ShapeCalc(std::make_shared<UpsampleNearest1DGradShapeCalc>(), {input_size}, {0})[0];
  }

  auto new_grad = ib->ExpandDims(grad, -1);
  auto out = ib->Emit("ResizeNearestNeighborV2Grad", {new_grad, size_node, ib->Value(false), ib->Value(false)});

  auto dx = ib->Squeeze(out, MakeValue(std::vector<int64_t>{-1}));

  return {dx};
});

class UpsampleNearest2DGradShapeCalc : public ShapeCalcFunctor {
 public:
  UpsampleNearest2DGradShapeCalc() : ShapeCalcFunctor("ShapeCalc_UpsampleNearest2DGrad") {}
  ~UpsampleNearest2DGradShapeCalc() override = default;
  MS_DECLARE_PARENT(UpsampleNearest2DGradShapeCalc, ShapeCalcFunctor)

  ValuePtr ToValue() const override { return nullptr; }
  void FromValue(const ValuePtr &value) override {}
  ShapeArray Calc(const ShapeArray &inputs) const override {
    // inputs: {input_size}
    auto input_size = inputs.at(0);
    std::vector<int64_t> size{input_size.begin() + (input_size.size() - kIndex2), input_size.end()};
    return {size};
  }
  std::vector<int64_t> Infer(const ShapeArray &inputs, const HashSet<size_t> &) const override { return {2}; }
};
REG_FUNCTOR("ShapeCalc_UpsampleNearest2DGrad", UpsampleNearest2DGradShapeCalc);

REG_FALLBACK_BUILDER("UpsampleNearest2DGrad").SetBody(BODYFUNC(ib) {
  auto grad = ib->GetInput(kIndex0);
  auto input_size = ib->GetInput(kIndex1);
  auto output_size = ib->GetInput(kIndex2);
  auto scales = ib->GetInput(kIndex3);

  auto value_ptr = input_size->BuildValue();
  NodePtr size_node;
  if (ops::IsValueKnown(value_ptr)) {
    auto input_size_val = GetValue<std::vector<int64_t>>(value_ptr);
    std::vector<int64_t> size{input_size_val.begin() + (input_size_val.size() - kIndex2), input_size_val.end()};
    size_node = ib->Value(size);
  } else {
    size_node = ib->ShapeCalc(std::make_shared<UpsampleNearest2DGradShapeCalc>(), {input_size}, {0})[0];
  }

  auto dx = ib->Emit("ResizeNearestNeighborV2Grad", {grad, size_node, ib->Value(false), ib->Value(false)});

  return {dx};
});

REG_FALLBACK_BUILDER("UpsampleBilinear2D").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto output_size = ib->GetInput(kIndex1);
  auto align_corners = ib->GetInput(kIndex3);
  if (output_size->abstract()->BuildType()->isa<TypeNone>()) {
    MS_LOG(EXCEPTION)
      << "For UpsampleBilinear2D, only output_size is supported in GE backend and it should not be None.";
  }
  auto out = ib->Emit("ResizeBilinearV2", {x, output_size, align_corners, ib->BoolNot(align_corners)});
  return {out};
});

REG_FALLBACK_BUILDER("UpsampleBilinear2DGrad").SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex0);
  auto input_size = ib->GetInput(kIndex1);
  auto output_size = ib->GetInput(kIndex2);
  auto align_corners = ib->GetInput(kIndex4);
  if (output_size->abstract()->BuildType()->isa<TypeNone>()) {
    MS_LOG(EXCEPTION)
      << "For UpsampleBilinear2DGrad, only output_size is supported in GE backend and it should not be None.";
  }
  // create original_image tensor
  auto value_ptr = input_size->BuildValue();
  if (!ops::IsValueKnown(value_ptr)) {
    MS_LOG(EXCEPTION) << "For UpsampleBilinear2DGrad, input_size should be const.";
  }
  auto x_type = dout->dtype()->type_id();
  auto x_shape = GetValue<std::vector<int64_t>>(value_ptr);
  auto x = ib->Fill(static_cast<double>(0.), x_shape, x_type);

  auto out = ib->Emit("ResizeBilinearGrad", {dout, x, align_corners, ib->BoolNot(align_corners)});
  return {out};
});

REG_FALLBACK_BUILDER("UpsampleLinear1D").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto output_size = ib->GetInput(kIndex1);
  auto scales = ib->GetInput(kIndex2);
  auto align_corners = ib->GetInput(kIndex3);

  NodePtr sizes_node{nullptr};
  NodePtr scales_node{nullptr};
  auto x_shape = x->shape();
  if (output_size->abstract()->isa<abstract::AbstractNone>()) {
    scales_node = scales;
    // fetch sizes
    auto scales_value_ptr = scales->BuildValue();
    MS_EXCEPTION_IF_NULL(scales_value_ptr);
    if (ops::IsValueKnown(scales_value_ptr)) {
      auto scales_value = GetValue<std::vector<pyfloat>>(scales_value_ptr);
      std::vector<int64_t> sizes_vec{static_cast<int64_t>(x_shape.at(kIndex2) * scales_value.at(kIndex0))};
      sizes_node = ib->Value(sizes_vec);
    } else {
      MS_LOG(EXCEPTION) << "For UpsampleLinear1D, scales should be const.";
    }
  } else {
    sizes_node = output_size;
    // fetch scales
    auto output_size_value_ptr = output_size->BuildValue();
    MS_EXCEPTION_IF_NULL(output_size_value_ptr);
    if (ops::IsValueKnown(output_size_value_ptr)) {
      auto output_size_value = GetValue<std::vector<int64_t>>(output_size_value_ptr);
      std::vector<float> scales_vec{static_cast<float>(output_size_value.at(kIndex0)) /
                                    static_cast<float>(x_shape.at(kIndex2))};
      scales_node = ib->Value(scales_vec);
    } else {
      MS_LOG(EXCEPTION) << "For UpsampleLinear1D, output_size should be const.";
    }
  }

  NodePtr coordinate_transformation_mode_node{nullptr};
  auto align_corners_ptr = align_corners->BuildValue();
  MS_EXCEPTION_IF_NULL(align_corners_ptr);
  if (ops::IsValueKnown(align_corners_ptr)) {
    auto align_corners_val = GetValue<bool>(align_corners_ptr);
    coordinate_transformation_mode_node = align_corners_val
                                            ? ib->Value(static_cast<int64_t>(CoordinateTransformMode::ALIGN_CORNERS))
                                            : ib->Value(static_cast<int64_t>(CoordinateTransformMode::HALF_PIXEL));
  } else {
    MS_LOG(EXCEPTION) << "For UpsampleLinear1D, align_corners should be const.";
  }

  auto new_x = ib->ExpandDims(x, -2);
  auto out = ib->Emit("ResizeD", {new_x, sizes_node, scales_node, coordinate_transformation_mode_node},
                      {{"mode", MakeValue("linear")}});
  auto real_out = ib->Squeeze(out, MakeValue(std::vector<int64_t>{-2}));

  return {real_out};
});

REG_FALLBACK_BUILDER("UpsampleLinear1DGrad").SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex0);
  auto input_size = ib->GetInput(kIndex1);
  auto output_size = ib->GetInput(kIndex2);
  auto scale_factor = ib->GetInput(kIndex3);
  auto align_corners = ib->GetInput(kIndex4);

  auto align_corners_ptr = align_corners->BuildValue();
  MS_EXCEPTION_IF_NULL(align_corners_ptr);
  if (ops::IsValueKnown(align_corners_ptr)) {
    auto align_corners_val = GetValue<bool>(align_corners_ptr);
    if (!align_corners_val && !scale_factor->abstract()->BuildType()->isa<TypeNone>()) {
      MS_LOG(EXCEPTION) << "For UpsampleLinear1DGrad with align_corners false, scales was not supported.";
    }
  } else {
    MS_LOG(EXCEPTION) << "For UpsampleLinear1DGrad, align_corners should be const.";
  }

  auto dout_type = dout->dtype()->type_id();
  NodePtr origin_image{nullptr};
  auto value_ptr = input_size->BuildValue();
  if (ops::IsValueKnown(value_ptr)) {
    auto input_size_val = GetValue<std::vector<int64_t>>(value_ptr);
    input_size_val.insert(input_size_val.begin() + kIndex2, 1);
    origin_image = ib->Fill(static_cast<double>(0.), input_size_val, dout_type);
  } else {
    MS_LOG(EXCEPTION) << "For UpsampleLinear1DGrad, input_size should be const.";
  }

  auto new_dout = ib->ExpandDims(ib->Cast(dout, TypeId::kNumberTypeFloat32), -2);
  auto dx = ib->Emit("ResizeBilinearGrad", {new_dout, origin_image, align_corners, ib->BoolNot(align_corners)});

  auto real_dx = ib->Squeeze(ib->Cast(dx, dout_type), MakeValue(std::vector<int64_t>{-2}));

  return {real_dx};
});

REG_FALLBACK_BUILDER("ResizeLinear1D").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto size = ib->GetInput(kIndex1);
  auto coordinate_transformation_mode = ib->GetInput(kIndex2);

  // fetch scales
  NodePtr scales_node{nullptr};
  auto x_shape = x->shape();
  auto size_value_ptr = size->BuildValue();
  MS_EXCEPTION_IF_NULL(size_value_ptr);
  if (ops::IsValueKnown(size_value_ptr)) {
    auto size_value = GetValue<std::vector<int64_t>>(size_value_ptr);
    std::vector<float> scales_vec{static_cast<float>(size_value.at(kIndex0)) / static_cast<float>(x_shape.at(kIndex2))};
    scales_node = ib->Value(scales_vec);
  } else {
    MS_LOG(EXCEPTION) << "For ResizeLinear1D, size should be const.";
  }

  auto new_x = ib->ExpandDims(x, -2);
  auto out =
    ib->Emit("ResizeD", {new_x, size, scales_node, coordinate_transformation_mode}, {{"mode", MakeValue("linear")}});
  auto real_out = ib->Squeeze(out, MakeValue(std::vector<int64_t>{-2}));

  return {real_out};
});

REG_FALLBACK_BUILDER("ResizeLinear1DGrad").SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex0);
  auto x = ib->GetInput(kIndex1);
  auto coordinate_transformation_mode = ib->GetInput(kIndex2);

  auto new_dout = ib->ExpandDims(ib->Cast(dout, TypeId::kNumberTypeFloat32), -2);
  NodePtr align_corners{nullptr};
  auto value_ptr = coordinate_transformation_mode->BuildValue();
  if (ops::IsValueKnown(value_ptr)) {
    auto mode = static_cast<CoordinateTransformMode>(GetValue<int64_t>(value_ptr));
    align_corners = mode == CoordinateTransformMode ::ALIGN_CORNERS ? ib->Value(true) : ib->Value(false);
  } else {
    MS_LOG(EXCEPTION) << "For ResizeLinear1DGrad, coordinate_transformation_mode should be const.";
  }

  auto dx = ib->Emit("ResizeBilinearGrad", {new_dout, x, align_corners, ib->BoolNot(align_corners)});

  auto dout_type = dout->dtype()->type_id();
  auto real_dx = ib->Squeeze(ib->Cast(dx, dout_type), MakeValue(std::vector<int64_t>{-2}));

  return {real_dx};
});

// It is just a temporary modification. If the attributes of the `TensorScatterElements`
// operator are changed to input, the `Scatter` operator can be directly replaced with `TensorScatterElements`.
REG_FALLBACK_BUILDER("Scatter").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto dim = ib->GetInput(kIndex1);
  auto index = ib->GetInput(kIndex2);
  auto src = ib->GetInput(kIndex3);
  auto reduce = ib->GetInput(kIndex4);
  auto dim_val = dim->BuildValue();
  auto reduce_val = reduce->BuildValue();
  if (!ops::IsValueKnown(dim_val) || !ops::IsValueKnown(reduce_val)) {
    MS_EXCEPTION(ValueError) << "For `TensorScatterElements` op, the `dim` and `reduce` must currently be a constant!";
  }
  std::unordered_map<int64_t, std::string> reduce_val_string{{0, "none"}, {1, "add"}};
  auto reduce_val_int = GetValue<int64_t>(reduce_val);
  const auto iter = reduce_val_string.find(reduce_val_int);
  if (iter == reduce_val_string.end()) {
    MS_EXCEPTION(ValueError) << "For `Scatter` op, fail to convert `reduce` val `" << reduce_val_int << "` to string!";
  }
  auto reduce_string = iter->second;
  auto out = ib->Emit("TensorScatterElements", {input, index, src},
                      {{"reduction", MakeValue<string>(reduce_string)}, {"axis", dim_val}});
  return {out};
});

REG_FALLBACK_BUILDER("ClampTensor").SetBody(BODYFUNC(ib) {
  // clamp equation: output = minimum(maximum(x, min), max)
  auto x = ib->GetInput(kIndex0);
  auto min = ib->GetInput(kIndex1);
  auto max = ib->GetInput(kIndex2);

  auto min_type_none = ib->GetDtype(min)->isa<TypeNone>();
  auto max_type_none = ib->GetDtype(max)->isa<TypeNone>();

  auto output = x;
  if (!min_type_none) {
    if (ib->GetDtype(x)->type_id() != ib->GetDtype(min)->type_id()) {
      min = ib->Cast(min, ib->GetDtype(x)->type_id());
    }
    output = ib->Maximum(output, min);
  }

  if (!max_type_none) {
    if (ib->GetDtype(x)->type_id() != ib->GetDtype(max)->type_id()) {
      max = ib->Cast(max, ib->GetDtype(x)->type_id());
    }
    output = ib->Minimum(output, max);
  }

  return {output};
});

REG_FALLBACK_BUILDER("ClampScalar").SetBody(BODYFUNC(ib) {
  // clamp equation: output = minimum(maximum(x, min), max)
  auto x = ib->GetInput(kIndex0);
  auto min = ib->GetInput(kIndex1);
  auto max = ib->GetInput(kIndex2);

  auto min_type_none = ib->GetDtype(min)->isa<TypeNone>();
  auto max_type_none = ib->GetDtype(max)->isa<TypeNone>();

  auto output = x;
  if (!min_type_none) {
    min = ib->ScalarToTensor(min, ib->GetDtype(min));
    if (ib->GetDtype(x)->type_id() != ib->GetDtype(min)->type_id()) {
      min = ib->Cast(min, ib->GetDtype(x)->type_id());
    }
    output = ib->Maximum(output, min);
  }

  if (!max_type_none) {
    max = ib->ScalarToTensor(max, ib->GetDtype(max));
    if (ib->GetDtype(x)->type_id() != ib->GetDtype(max)->type_id()) {
      max = ib->Cast(max, ib->GetDtype(x)->type_id());
    }
    output = ib->Minimum(output, max);
  }

  return {output};
});

NodePtr PaddingTupleToTensor(NodePtr paddings, FallbackIRBuilder *ib) {
  auto padding_value = paddings->BuildValue();
  auto padding_vec = CheckAndConvertUtils::CheckIntOrTupleInt("padding", padding_value, "pad");
  auto padding_tensor = ib->Tensor(padding_vec);
  return padding_tensor;
}

bool IsInputNeedExpand(NodePtr paddingsTensor, NodePtr inputTensor) {
  auto padding_shape = paddingsTensor->shape();
  auto input_x_shape = inputTensor->shape();
  return ((padding_shape[0] / 2) + 1) == SizeToLong(input_x_shape.size());
}

REG_FALLBACK_BUILDER("ConstantPadND").SetBody(BODYFUNC(ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto padding = ib->GetInput(kIndex1);
  auto value = ib->GetInput(kIndex2);
  auto value_tensor = ib->ScalarToTensor(value);
  if (value->dtype() != input_x->dtype()) {
    value_tensor = ib->Cast(value_tensor, input_x->dtype());
  }
  auto padding_tensor = PaddingTupleToTensor(padding, ib);
  bool is_expand = IsInputNeedExpand(padding_tensor, input_x);
  if (is_expand) {
    input_x = ib->Emit("ExpandDims", {input_x, ib->Value<int64_t>(0)});
  }
  auto out = ib->Emit("PadV3", {input_x, padding_tensor, value_tensor},
                      {{"mode", MakeValue<string>("constant")}, {"paddings_contiguous", MakeValue(true)}});
  if (is_expand) {
    out = ib->Squeeze(out, MakeValue(ShapeVector{0}));
  }
  return {out};
});

REG_FALLBACK_BUILDER("ReflectionPad1D").SetBody(BODYFUNC(ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto padding = ib->GetInput(kIndex1);

  auto padding_tensor = PaddingTupleToTensor(padding, ib);
  bool is_expand = IsInputNeedExpand(padding_tensor, input_x);
  if (is_expand) {
    input_x = ib->Emit("ExpandDims", {input_x, ib->Value<int64_t>(0)});
  }
  auto out = ib->Emit("PadV3", {input_x, padding_tensor, ib->EmitValue(kNone)},
                      {{"mode", MakeValue<string>("reflect")}, {"paddings_contiguous", MakeValue(true)}});
  if (is_expand) {
    out = ib->Squeeze(out, MakeValue(ShapeVector{0}));
  }
  return {out};
});

REG_FALLBACK_BUILDER("ReflectionPad2D").SetBody(BODYFUNC(ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto padding = ib->GetInput(kIndex1);

  auto padding_tensor = PaddingTupleToTensor(padding, ib);
  bool is_expand = IsInputNeedExpand(padding_tensor, input_x);
  if (is_expand) {
    input_x = ib->Emit("ExpandDims", {input_x, ib->Value<int64_t>(0)});
  }
  auto out = ib->Emit("PadV3", {input_x, padding_tensor, ib->EmitValue(kNone)},
                      {{"mode", MakeValue<string>("reflect")}, {"paddings_contiguous", MakeValue(true)}});
  if (is_expand) {
    out = ib->Squeeze(out, MakeValue(ShapeVector{0}));
  }
  return {out};
});

REG_FALLBACK_BUILDER("ReflectionPad3D").SetBody(BODYFUNC(ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto padding = ib->GetInput(kIndex1);

  auto padding_tensor = PaddingTupleToTensor(padding, ib);
  bool is_expand = IsInputNeedExpand(padding_tensor, input_x);
  if (is_expand) {
    input_x = ib->Emit("ExpandDims", {input_x, ib->Value<int64_t>(0)});
  }
  auto out = ib->Emit("PadV3", {input_x, padding_tensor, ib->EmitValue(kNone)},
                      {{"mode", MakeValue<string>("reflect")}, {"paddings_contiguous", MakeValue(true)}});
  if (is_expand) {
    out = ib->Squeeze(out, MakeValue(ShapeVector{0}));
  }
  return {out};
});

REG_FALLBACK_BUILDER("ReplicationPad1D").SetBody(BODYFUNC(ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto padding = ib->GetInput(kIndex1);

  auto padding_tensor = PaddingTupleToTensor(padding, ib);
  bool is_expand = IsInputNeedExpand(padding_tensor, input_x);
  if (is_expand) {
    input_x = ib->Emit("ExpandDims", {input_x, ib->Value<int64_t>(0)});
  }
  auto out = ib->Emit("PadV3", {input_x, padding_tensor, ib->EmitValue(kNone)},
                      {{"mode", MakeValue<string>("edge")}, {"paddings_contiguous", MakeValue(true)}});
  if (is_expand) {
    out = ib->Squeeze(out, MakeValue(ShapeVector{0}));
  }
  return {out};
});

REG_FALLBACK_BUILDER("ReplicationPad2D").SetBody(BODYFUNC(ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto padding = ib->GetInput(kIndex1);

  auto padding_tensor = PaddingTupleToTensor(padding, ib);
  bool is_expand = IsInputNeedExpand(padding_tensor, input_x);
  if (is_expand) {
    input_x = ib->Emit("ExpandDims", {input_x, ib->Value<int64_t>(0)});
  }
  auto out = ib->Emit("PadV3", {input_x, padding_tensor, ib->EmitValue(kNone)},
                      {{"mode", MakeValue<string>("edge")}, {"paddings_contiguous", MakeValue(true)}});
  if (is_expand) {
    out = ib->Squeeze(out, MakeValue(ShapeVector{0}));
  }
  return {out};
});

REG_FALLBACK_BUILDER("ReplicationPad3D").SetBody(BODYFUNC(ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto padding = ib->GetInput(kIndex1);

  auto padding_tensor = PaddingTupleToTensor(padding, ib);
  bool is_expand = IsInputNeedExpand(padding_tensor, input_x);
  if (is_expand) {
    input_x = ib->Emit("ExpandDims", {input_x, ib->Value<int64_t>(0)});
  }
  auto out = ib->Emit("PadV3", {input_x, padding_tensor, ib->EmitValue(kNone)},
                      {{"mode", MakeValue<string>("edge")}, {"paddings_contiguous", MakeValue(true)}});
  if (is_expand) {
    out = ib->Squeeze(out, MakeValue(ShapeVector{0}));
  }
  return {out};
});

REG_FALLBACK_BUILDER("ReflectionPad1DGrad").SetBody(BODYFUNC(ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto padding = ib->GetInput(kIndex2);

  auto padding_tensor = PaddingTupleToTensor(padding, ib);
  bool is_expand = IsInputNeedExpand(padding_tensor, input_x);
  if (is_expand) {
    input_x = ib->Emit("ExpandDims", {input_x, ib->Value<int64_t>(0)});
  }
  auto out = ib->Emit("PadV3Grad", {input_x, padding_tensor},
                      {{"mode", MakeValue<string>("reflect")}, {"paddings_contiguous", MakeValue(true)}});
  if (is_expand) {
    out = ib->Squeeze(out, MakeValue(ShapeVector{0}));
  }
  return {out};
});

REG_FALLBACK_BUILDER("ReflectionPad2DGrad").SetBody(BODYFUNC(ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto padding = ib->GetInput(kIndex2);

  auto padding_tensor = PaddingTupleToTensor(padding, ib);
  bool is_expand = IsInputNeedExpand(padding_tensor, input_x);
  if (is_expand) {
    input_x = ib->Emit("ExpandDims", {input_x, ib->Value<int64_t>(0)});
  }
  auto out = ib->Emit("PadV3Grad", {input_x, padding_tensor},
                      {{"mode", MakeValue<string>("reflect")}, {"paddings_contiguous", MakeValue(true)}});
  if (is_expand) {
    out = ib->Squeeze(out, MakeValue(ShapeVector{0}));
  }
  return {out};
});

REG_FALLBACK_BUILDER("ReflectionPad3DGrad").SetBody(BODYFUNC(ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto padding = ib->GetInput(kIndex2);

  auto padding_tensor = PaddingTupleToTensor(padding, ib);
  bool is_expand = IsInputNeedExpand(padding_tensor, input_x);
  if (is_expand) {
    input_x = ib->Emit("ExpandDims", {input_x, ib->Value<int64_t>(0)});
  }
  auto out = ib->Emit("PadV3Grad", {input_x, padding_tensor},
                      {{"mode", MakeValue<string>("reflect")}, {"paddings_contiguous", MakeValue(true)}});
  if (is_expand) {
    out = ib->Squeeze(out, MakeValue(ShapeVector{0}));
  }
  return {out};
});

REG_FALLBACK_BUILDER("ReplicationPad1DGrad").SetBody(BODYFUNC(ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto padding = ib->GetInput(kIndex2);

  auto padding_tensor = PaddingTupleToTensor(padding, ib);
  bool is_expand = IsInputNeedExpand(padding_tensor, input_x);
  if (is_expand) {
    input_x = ib->Emit("ExpandDims", {input_x, ib->Value<int64_t>(0)});
  }
  auto out = ib->Emit("PadV3Grad", {input_x, padding_tensor},
                      {{"mode", MakeValue<string>("edge")}, {"paddings_contiguous", MakeValue(true)}});
  if (is_expand) {
    out = ib->Squeeze(out, MakeValue(ShapeVector{0}));
  }
  return {out};
});

REG_FALLBACK_BUILDER("ReplicationPad2DGrad").SetBody(BODYFUNC(ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto padding = ib->GetInput(kIndex2);

  auto padding_tensor = PaddingTupleToTensor(padding, ib);
  bool is_expand = IsInputNeedExpand(padding_tensor, input_x);
  if (is_expand) {
    input_x = ib->Emit("ExpandDims", {input_x, ib->Value<int64_t>(0)});
  }
  auto out = ib->Emit("PadV3Grad", {input_x, padding_tensor},
                      {{"mode", MakeValue<string>("edge")}, {"paddings_contiguous", MakeValue(true)}});
  if (is_expand) {
    out = ib->Squeeze(out, MakeValue(ShapeVector{0}));
  }
  return {out};
});

REG_FALLBACK_BUILDER("ReplicationPad3DGrad").SetBody(BODYFUNC(ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto padding = ib->GetInput(kIndex2);

  auto padding_tensor = PaddingTupleToTensor(padding, ib);
  bool is_expand = IsInputNeedExpand(padding_tensor, input_x);
  if (is_expand) {
    input_x = ib->Emit("ExpandDims", {input_x, ib->Value<int64_t>(0)});
  }
  auto out = ib->Emit("PadV3Grad", {input_x, padding_tensor},
                      {{"mode", MakeValue<string>("edge")}, {"paddings_contiguous", MakeValue(true)}});
  if (is_expand) {
    out = ib->Squeeze(out, MakeValue(ShapeVector{0}));
  }
  return {out};
});

REG_FALLBACK_BUILDER("Embedding").SetBody(BODYFUNC(ib) {
  MS_LOG(ERROR) << "ABCCC";
  auto input = ib->GetInput(kIndex0);
  auto weight = ib->GetInput(kIndex1);
  auto padding_idx = ib->GetInput(kIndex2);
  auto max_norm = ib->GetInput(kIndex3);
  auto norm_type = ib->GetInput(kIndex4);

  auto max_norm_value = max_norm->BuildValue();

  if (max_norm_value != nullptr && !max_norm_value->isa<None>()) {
    auto norm_type_value = norm_type->BuildValue();
    if (!ops::IsValueKnown(max_norm_value) || !ops::IsValueKnown(norm_type_value)) {
      MS_INTERNAL_EXCEPTION(ValueError) << "For `Embedding` op, max_norm and norm_type must be constant!";
    }

    auto max_norm_double = static_cast<double>(GetValue<float>(max_norm_value));
    auto norm_type_double = static_cast<double>(GetValue<float>(norm_type_value));

    if (max_norm_double < 0) {
      MS_EXCEPTION(ValueError) << "For Embedding, the max_norm must be greater equal than 0, but got: "
                               << max_norm_double << ".";
    }

    // do EmbeddingRenorm
    auto new_input = ib->Emit(ops::kNameReshape, {input, ib->Value(std::vector<int64_t>{-1})});
    auto gather_out = ib->Emit(ops::kNameGather, {weight, new_input, ib->Value((int64_t)0), ib->Value((int64_t)0)});
    auto renorm_out = ib->Emit(ops::kNameRenorm, {gather_out},
                               {{"p", MakeValue<float>(norm_type_double)},
                                {"dim", MakeValue<int64_t>(0)},
                                {"maxnorm", MakeValue<float>(max_norm_double)}});

    if (IsDynamic(input->shape())) {
      MS_INTERNAL_EXCEPTION(ValueError)
        << "For `Embedding` op, dynamic_shape is not support on Fallback path, but got input shape: " << input->shape()
        << ".";
    }

    auto indices_size = SizeOf(input->shape());
    constexpr int64_t kMaxRangeSize = 1000000;
    auto indices = ib->Emit(ops::kNameRange, {ib->Value((int64_t)0), ib->Value(static_cast<int64_t>(indices_size)),
                                              ib->Value((int64_t)1), ib->Value(kMaxRangeSize)});
    auto gather_out2 = ib->Emit(ops::kNameGather, {renorm_out, indices, ib->Value((int64_t)0), ib->Value((int64_t)0)});
    auto mul_out = ib->Emit(ops::kNameMul, {gather_out, gather_out2});
    weight = ib->Emit(ops::kNameScatterUpdate, {weight, new_input, mul_out});
  }

  auto out = ib->Emit(ops::kNameGather, {weight, input, ib->Value((int64_t)0), ib->Value((int64_t)0)});
  return {out};
});
}  // namespace expander
}  // namespace mindspore
