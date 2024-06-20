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

#include "plugin/device/ascend/kernel/internal/acme/acme_helper.h"

#include <unordered_map>
#include "mindapi/base/type_id.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace kernel {

acme::DataType TransAcmeDataType(TypeId ms_type) {
  static const std::unordered_map<TypeId, acme::DataType> kMSTypeToAcmeType = {
    {kNumberTypeFloat16, acme::DataType::kTypeFloat16},     {kNumberTypeBFloat16, acme::DataType::kTypeBF16},
    {kNumberTypeFloat32, acme::DataType::kTypeFloat32},     {kNumberTypeDouble, acme::DataType::kTypeFloat64},
    {kNumberTypeInt32, acme::DataType::kTypeInt32},         {kNumberTypeUInt32, acme::DataType::kTypeUint32},
    {kNumberTypeInt16, acme::DataType::kTypeInt16},         {kNumberTypeUInt16, acme::DataType::kTypeUint16},
    {kNumberTypeInt8, acme::DataType::kTypeInt8},           {kNumberTypeUInt8, acme::DataType::kTypeUint8},
    {kNumberTypeInt64, acme::DataType::kTypeInt64},         {kNumberTypeUInt64, acme::DataType::kTypeUint64},
    {kNumberTypeComplex64, acme::DataType::kTypeComplex64}, {kNumberTypeComplex128, acme::DataType::kTypeComplex128},
    {kNumberTypeBool, acme::DataType::kTypeBool},
  };

  auto iter = kMSTypeToAcmeType.find(ms_type);
  if (iter == kMSTypeToAcmeType.end()) {
    MS_LOG(EXCEPTION) << "Type " << ms_type << " is not supported in Acme";
  }

  return iter->second;
}

acme::TensorFormat TransAcmeFormat(Format format) {
  static const std::unordered_map<Format, acme::TensorFormat> kMSFormatToAcmeFormat = {
    {DEFAULT_FORMAT, acme::TensorFormat::kFormatND},
    {NCHW, acme::TensorFormat::kFormatNCHW},
    {NHWC, acme::TensorFormat::kFormatNHWC},
    {ND, acme::TensorFormat::kFormatND},
  };

  auto iter = kMSFormatToAcmeFormat.find(format);
  if (iter == kMSFormatToAcmeFormat.end()) {
    MS_LOG(EXCEPTION) << "Format " << format << " is not supported in Acme";
  }

  return iter->second;
}

}  // namespace kernel
}  // namespace mindspore
