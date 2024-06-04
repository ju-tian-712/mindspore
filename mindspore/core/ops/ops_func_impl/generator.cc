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

#include "ops/ops_func_impl/generator.h"
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
using namespace generator;
namespace {
int64_t GetCmd(AbstractBasePtr cmd) {
  auto cmd_opt = GetScalarValue<int64_t>(cmd->GetValue());
  if (MS_UNLIKELY(!cmd_opt.has_value())) {
    MS_LOG(EXCEPTION) << "Cmd value unavailable.";
  }
  auto cmd_value = cmd_opt.value();
  MS_CHECK_VALUE((cmd_value > _START && cmd_value < _END), "Unknown cmd: " + std::to_string(cmd_value));
  return cmd_value;
}

static std::unordered_map<int64_t, std::vector<TypePtr>> kGeneratorInputFormat{
  {STEP, {ParamType, ParamType, ParamType}},                  // step, seed, offset
  {SEED, {ParamType, ParamType}},                             // seed, offset
  {GET_STATE, {ParamType, ParamType}},                        // seed, offset
  {SET_STATE, {ParamType, ParamType, ParamType, ParamType}},  // seed, offset, new_seed, new_offset
  {UNPACK_STATE, {StateType}},                                // state
  {INITIAL_SEED, {ParamType}},                                // seed
};
}  // namespace

int32_t GeneratorFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  int64_t cmd = GetCmd(input_args[kCmdIndex]);
  const auto cmd_str = GeneratorEnumToString.at(cmd);
  const auto expected_types = kGeneratorInputFormat[cmd];
  auto input_types = input_args[kInputsIndex]->GetType()->cast<TuplePtr>();
  MS_EXCEPTION_IF_NULL(input_types);
  if (MS_UNLIKELY(input_types->dynamic_len())) {
    auto element_type = input_types->dynamic_element_type();
    CheckAndConvertUtils::CheckTensorTypeValid("inputs", element_type, {expected_types[0]}, primitive->name());
  } else {
    auto element_types = input_types->elements();
    MS_CHECK_VALUE(element_types.size() == expected_types.size(),
                   CheckAndConvertUtils::CheckInteger("input number for cmd " + cmd_str, input_args.size(), kEqual,
                                                      expected_types.size(), primitive->name()));
    for (size_t i = 0; i < element_types.size(); ++i) {
      CheckAndConvertUtils::CheckTensorTypeValid("inputs", element_types[i], {expected_types[i]}, primitive->name());
    }
  }
  return true;
}

BaseShapePtr GeneratorFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  int64_t cmd = GetCmd(input_args[kCmdIndex]);
  std::vector<ShapeVector> infer_shapes{};
  switch (cmd) {
    case STEP:
      infer_shapes = {{1}, {1}};
      break;
    case SEED:
      infer_shapes = {{1}};
      break;
    case GET_STATE:
      infer_shapes = {{sizeof(param_type) * 2}};
      break;
    case SET_STATE:
      infer_shapes = {{1}};
      break;
    case UNPACK_STATE:
      infer_shapes = {{1}, {1}};
      break;
    case INITIAL_SEED:
      infer_shapes = {{1}};
      break;
    default:
      MS_LOG(EXCEPTION) << "Unknown cmd: " << cmd;
  }
  std::vector<abstract::BaseShapePtr> infer_shape_ptrs{infer_shapes.size()};
  std::transform(infer_shapes.begin(), infer_shapes.end(), infer_shape_ptrs.begin(),
                 [](ShapeVector &v) { return std::make_shared<abstract::Shape>(v); });
  return std::make_shared<abstract::TupleShape>(infer_shape_ptrs);
}

TypePtr GeneratorFuncImpl::InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const {
  int64_t cmd = GetCmd(input_args[kCmdIndex]);
  std::vector<TypePtr> infer_types{};
  switch (cmd) {
    case STEP:
      infer_types = {ParamType, ParamType};
      break;
    case SEED:
      infer_types = {ParamType};
      break;
    case GET_STATE:
      infer_types = {StateType};
      break;
    case SET_STATE:
      infer_types = {kInt64};
      break;
    case UNPACK_STATE:
      infer_types = {ParamType, ParamType};
      break;
    case INITIAL_SEED:
      infer_types = {ParamType};
      break;
    default:
      MS_LOG(EXCEPTION) << "Unknown cmd: " << cmd;
  }
  return std::make_shared<Tuple>(infer_types);
}
}  // namespace mindspore::ops
