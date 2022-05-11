/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_NNACL_FP32_POWER_FP32_H_
#define MINDSPORE_NNACL_FP32_POWER_FP32_H_

#include <math.h>
#include "nnacl/op_base.h"
#include "nnacl/power_parameter.h"

typedef void (*PowerFun)(const float *, const float *, float *, int, float, float);
typedef float (*PowerScalarFun)(float x, const float *exponent);

#ifdef __cplusplus
extern "C" {
#endif
static inline bool CheckInteger(float f) { return fabsf(f - (int)(f)) < 0.000001; }

static inline float StdPowerScalar(float x, const float *exponent) { return powf(x, *exponent); }

int Power(const float *input, const float *exponent, float *output, int len, float scale, float shift, bool broadcast);
void PowerSingle(const float *input, const float *exponent, float *output, int len, float scale, float shift);
void PowerBroadCast(const float *input, const float *exponent, float *output, int len, float scale, float shift);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_POWER_FP32_H_
