/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_ADDN_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_ADDN_INFO_H_

#include <memory>
#include <string>
#include <vector>

#include "utils/hash_map.h"
#include "ir/value.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
class AddNInfo : public OperatorInfo {
 public:
  AddNInfo(const std::string &name, const Shapes &input_shape, const Shapes &output_shape, const PrimitiveAttrs &attrs)
      : OperatorInfo(name, input_shape, output_shape, attrs, std::make_shared<AddNCost>()) {}
  ~AddNInfo() = default;

  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override { return SetCostUnderStrategyBase(strategy); }
  void ReComputeBatchSplitFlagList() override;

 protected:
  Status GetAttrs() override {
    infer_strategy_mode_ = SAME_MODE;
    return SUCCESS;
  }
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferForwardCommunication() override { return SUCCESS; }
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_ADDN_INFO_H_
