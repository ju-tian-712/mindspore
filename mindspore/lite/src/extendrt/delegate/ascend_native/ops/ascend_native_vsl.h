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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_OPS_ASCEND_NATIVE_VSL_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_OPS_ASCEND_NATIVE_VSL_H_
#include <string>
#include <vector>
#include <map>
#include <memory>
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAscendNativeVsl = "AscendNativeVsl";
/// \brief Custom defined user-defined operator prototype.
class MIND_API AscendNativeVsl : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(AscendNativeVsl);
  /// \brief Constructor.
  AscendNativeVsl() : BaseOperator(kNameAscendNativeVsl) {}

  void Init();

  /// \brief Method to set type attribute.
  ///
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_OPS_ASCEND_NATIVE_VSL_H_
