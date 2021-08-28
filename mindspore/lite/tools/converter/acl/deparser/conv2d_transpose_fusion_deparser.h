/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef ACL_DEPARSER_PRIMITIVE_CONV2DTRANSPOSEFUSION_DEPARSER_H
#define ACL_DEPARSER_PRIMITIVE_CONV2DTRANSPOSEFUSION_DEPARSER_H

#include "tools/converter/acl/deparser/primitive_deparser.h"
#include "ops/fusion/conv2d_transpose_fusion.h"

using mindspore::ops::kNameConv2dTransposeFusion;

namespace mindspore {
namespace lite {
class Conv2dTransposeFusionDeparser : public PrimitiveDeparser {
 public:
  Conv2dTransposeFusionDeparser() : PrimitiveDeparser(kNameConv2dTransposeFusion) {}
  ~Conv2dTransposeFusionDeparser() override = default;

  STATUS Deparser(const CNodePtr &cnode) override;
};
}  // namespace lite
}  // namespace mindspore
#endif  // ACL_DEPARSER_PRIMITIVE_CONV2DTRANSPOSEFUSION_DEPARSER_H
