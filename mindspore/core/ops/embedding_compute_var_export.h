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

#ifndef MINDSPORE_CORE_OPS_EMBEDDING_COMPUTE_VAR_EXPORT_H
#define MINDSPORE_CORE_OPS_EMBEDDING_COMPUTE_VAR_EXPORT_H

#include <memory>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameEmbeddingComputeVarExport = "EmbeddingComputeVarExport";
class MIND_API EmbeddingComputeVarExport : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(EmbeddingComputeVarExport);
  EmbeddingComputeVarExport() : BaseOperator(kNameEmbeddingComputeVarExport) {
    InitIOName({"file_path", "ps_id", "table_id"}, {"output"});
  }
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_EMBEDDING_COMPUTE_VAR_EXPORT_H
