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

#ifndef MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_UPSAMPLE_NEAREST3D_H_
#define MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_UPSAMPLE_NEAREST3D_H_

#include "ops/ops_func_impl/upsample_forward_base.h"

namespace mindspore {
namespace ops {
class MIND_API UpsampleNearest3DFuncImpl final : public UpsampleForwardBaseFuncImpl {
 protected:
  size_t GetImageRank() const noexcept override { return 5; }
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_UPSAMPLE_NEAREST3D_H_
