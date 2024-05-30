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

#include "ops/ops_func_impl/embedding_apply_ftrl.h"

#include <vector>
#include <set>
#include <map>
#include <string>
#include <memory>

#include "ops/ops_func_impl/embedding_utils.h"
#include "utils/ms_context.h"
#include "utils/check_convert_utils.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr EmbeddingApplyFtrlFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) const {
  return std::make_shared<abstract::TensorShape>(ShapeVector{});
}

TypePtr EmbeddingApplyFtrlFuncImpl::InferType(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) const {
  return std::make_shared<TensorType>(kInt32);
}
}  // namespace ops
}  // namespace mindspore
