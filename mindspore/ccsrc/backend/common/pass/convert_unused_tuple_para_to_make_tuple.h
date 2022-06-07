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

#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_PASS_CONVERT_UNUSED_TUPLE_PARA_TO_MAKE_TUPLE_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_PASS_CONVERT_UNUSED_TUPLE_PARA_TO_MAKE_TUPLE_H_

#include "ir/anf.h"
#include "backend/common/optimizer/pass.h"

namespace mindspore {
namespace opt {
class ConvertUnusedTupleParaToMakeTuple : public Pass {
 public:
  ConvertUnusedTupleParaToMakeTuple() : Pass("convert_unused_tuple_para_to_make_tuple") {}
  ~ConvertUnusedTupleParaToMakeTuple() override = default;
  bool Run(const FuncGraphPtr &graph) override;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_PASS_CONVERT_UNUSED_TUPLE_PARA_TO_MAKE_TUPLE_H_
