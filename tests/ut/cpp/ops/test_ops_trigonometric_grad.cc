/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (theGrad "License");
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
#include "ops/test_ops_cmp_utils.h"
#include "ops/ops_func_impl/acos_grad.h"
#include "ops/ops_func_impl/acosh_grad.h"
#include "ops/ops_func_impl/asin_grad.h"
#include "ops/ops_func_impl/asinh_grad.h"
#include "ops/ops_func_impl/atan_grad.h"

namespace mindspore {
namespace ops {
BINARY_SHAPE_EQUALS_TEST_WITH_DEFAULT_CASES(ACosGrad);
BINARY_SHAPE_EQUALS_TEST_WITH_DEFAULT_CASES(AcoshGrad);
BINARY_SHAPE_EQUALS_TEST_WITH_DEFAULT_CASES(AsinGrad);
BINARY_SHAPE_EQUALS_TEST_WITH_DEFAULT_CASES(AsinhGrad);
BINARY_SHAPE_EQUALS_TEST_WITH_DEFAULT_CASES(AtanGrad);
}  // namespace ops
}  // namespace mindspore
