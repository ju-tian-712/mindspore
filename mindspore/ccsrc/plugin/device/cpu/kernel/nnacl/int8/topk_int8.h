/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef NNACL_INT8_TOPK_INT8_H_
#define NNACL_INT8_TOPK_INT8_H_

#include "nnacl/op_base.h"
#include "nnacl/fp32/topk_fp32.h"

typedef struct TopkNodeInt8 {
  int8_t element;
  int32_t index;
} TopkNodeInt8;

#ifdef __cplusplus
extern "C" {
#endif
void TopkInt8(int8_t *input_data, int8_t *output_data, int32_t *output_index, TopkParameter *parameter);
#ifdef __cplusplus
}
#endif

#endif  // NNACL_INT8_TOPK_INT8_H_
