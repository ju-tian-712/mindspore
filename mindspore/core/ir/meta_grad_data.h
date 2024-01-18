/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_IR_META_GRAD_DATA_H_
#define MINDSPORE_CORE_IR_META_GRAD_DATA_H_

#include <memory>
#include <utility>
#include <map>
#include <vector>
#include <string>
#include "ir/anf.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace pynative::autograd {
class VariableAdjoint;
}  // namespace pynative::autograd

using VariableAdjointPtr = std::shared_ptr<pynative::autograd::VariableAdjoint>;
using VariableAdjointWeakPtr = std::weak_ptr<pynative::autograd::VariableAdjoint>;

class AutoGradMetaData {
 public:
  AutoGradMetaData() = default;
  AutoGradMetaData(const VariableAdjointPtr &variable, const ParameterPtr &parameter,
                   const InputType input_type = InputType::kConstant)
      : variable_(variable), parameter_(parameter), input_type_(input_type) {}
  VariableAdjointPtr variable() const { return variable_.lock(); }
  void set_variable(const VariableAdjointPtr &variable) { variable_ = variable; }
  ParameterPtr parameter() const { return parameter_.lock(); }
  void set_parameter(const ParameterPtr &parameter) { parameter_ = parameter; }
  void set_k_node(const AnfNodePtr &k_node) { k_node_ = k_node; }
  AnfNodePtr k_node() const { return k_node_.lock(); }
  InputType input_type() const { return input_type_; }
  void set_input_type(InputType input_type) { input_type_ = input_type; }
  size_t op_index() const { return op_index_; }
  void set_op_index(size_t op_index) { op_index_ = op_index; }

 private:
  // Weakptr for variable, to avoid circular reference
  VariableAdjointWeakPtr variable_;
  // Weakptr to hold ir parameter of input or parameter
  ParameterWeakPtr parameter_;
  // Weakptr to k_node for tensor
  AnfNodeWeakPtr k_node_;
  // Type of grad tensor
  InputType input_type_;
  // Optional for op output, represent index of op in execute order.
  size_t op_index_{0};
};
using AutoGradMetaDataPtr = std::shared_ptr<AutoGradMetaData>;
}  // namespace mindspore
#endif  // MINDSPORE_CORE_IR_META_GRAD_DATA_H_
