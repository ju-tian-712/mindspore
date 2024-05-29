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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MULTI_WEIGHT_MATMULS_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MULTI_WEIGHT_MATMULS_FUSION_H_

#include <string>
#include <memory>
#include "include/backend/optimizer/pass.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/optimizer/optimizer.h"
#include "mindspore/core/ops/nn_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "plugin/device/ascend/optimizer/ir_fusion/inference_weight_preprocess_utils.h"

namespace mindspore {
namespace opt {
/**
 * Fuse WeightQuantBatchMatMul when a node is used by several matmuls.
 *
 * example:
 * x = WeightQuantBatchMatMul(A, w1, w4, ...)
 * y = WeightQuantBatchMatMul(A, w2, w5, ...)
 * z = WeightQuantBatchMatMul(A, w3, w6, ...)
 * ...
 * ------->
 * out = WeightQuantBatchMatMul(A, w1+w2+w3, w4+w5+w6, ...)
 * t = Split(out)
 * x = tuple_getitem(t, 0)
 * y = tuple_getitem(t, 1)
 * z = tuple_getitem(t, 2)
 * ...
 */
class MultiWeightMatmulsFusion : public Pass {
 public:
  MultiWeightMatmulsFusion() : Pass("multi_weight_matmuls_fusion") {}
  ~MultiWeightMatmulsFusion() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 protected:
  void Process(const std::string &name, const AnfNodePtr &node, const AnfNodePtrList &users,
               AnfNodePtrList *getitems) const;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MULTI_WEIGHT_MATMULS_FUSION_H_
