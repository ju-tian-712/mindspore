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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_GE_ADD_NOOP_TO_ADAM_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_GE_ADD_NOOP_TO_ADAM_H_
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/optimizer/pattern_engine.h"
#include "ir/anf.h"

namespace mindspore {
namespace opt {
class AddNoOpToAdam : public PatternProcessPass {
 public:
  explicit AddNoOpToAdam(bool multigraph = true) : PatternProcessPass("add_noop_to_adam", multigraph) {}
  ~AddNoOpToAdam() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;
};

class AddNoOpToAdamW : public PatternProcessPass {
 public:
  explicit AddNoOpToAdamW(bool multigraph = true) : PatternProcessPass("add_noop_to_adamw", multigraph) {}
  ~AddNoOpToAdamW() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;
};

class AddNoOpToAdaGrad : public PatternProcessPass {
 public:
  explicit AddNoOpToAdaGrad(bool multigraph = true) : PatternProcessPass("add_noop_to_ada_grad", multigraph) {}
  ~AddNoOpToAdaGrad() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;
};

class AddNoOpToFtrl : public PatternProcessPass {
 public:
  explicit AddNoOpToFtrl(bool multigraph = true) : PatternProcessPass("add_noop_to_ftrl", multigraph) {}
  ~AddNoOpToFtrl() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_GE_ADD_NOOP_TO_ADAM_H_
