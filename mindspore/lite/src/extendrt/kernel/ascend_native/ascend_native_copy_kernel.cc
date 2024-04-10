/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "extendrt/kernel/ascend_native/ascend_native_copy_kernel.h"
#include <algorithm>
#include "extendrt/delegate/ascend_native/ascend_native_kernel_registry.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/vector_core/copy_cast.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/utils.h"
#include "extendrt/delegate/ops/copy.h"

namespace mindspore::kernel {
int AscendNativeCopyKernel::InferShape() {
  out_tensors_[0]->set_shape(in_tensors_[0]->shape());
  auto const &data_type = in_tensors_[0]->data_type();
  bool is_float =
    (data_type == kNumberTypeFloat32) || (data_type == kNumberTypeFloat16) || (data_type == kNumberTypeFloat);
  if (copy_type_ == ops::AscendNativeCopy::CopyFormatType::HOST_DEVICE) {
    if (is_float) {
      out_tensors_[0]->set_data_type(kNumberTypeFloat16);
    } else {
      out_tensors_[0]->set_data_type(data_type);
    }
  } else if (copy_type_ == ops::AscendNativeCopy::CopyFormatType::DEVICE_HOST) {
    out_tensors_[0]->set_data_type(data_type);
  }
  return lite::RET_OK;
}

int AscendNativeCopyKernel::Prepare() {
  ascend_native::SetContext(const_cast<void *>(acl_ctx_));
  auto prim = GetValueNode<PrimitivePtr>(primitive_.cnode->input(0));
  copy_type_ =
    static_cast<ops::AscendNativeCopy::CopyFormatType>(GetValue<int64_t>(prim->GetAttr(mindspore::ops::kCopyFormat)));
  auto ret = InferShape();
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Ascend native copy kernel inferShape failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

int AscendNativeCopyKernel::PreProcess() {
  ascend_native::SetContext(const_cast<void *>(acl_ctx_));
  out_tensors_[0]->ResetRefCount();
  if ((in_tensors_[0]->tensor_name().compare("attention_mask")) == 0) {
    return lite::RET_OK;
  }
  switch (copy_type_) {
    case ops::AscendNativeCopy::CopyFormatType::HOST_DEVICE: {
      if (out_tensors_[0]->device_data() == nullptr) {
        auto device_data = ascend_native::MallocDevice(out_tensors_[0]->Size());
        if (device_data == nullptr) {
          MS_LOG(ERROR) << "fail to allocate " << out_tensors_[0]->Size() << "Bytes for device";
          return lite::RET_NULL_PTR;
        }
        out_tensors_[0]->set_device_data(device_data);
      }
      break;
    }
    case ops::AscendNativeCopy::CopyFormatType::DEVICE_HOST: {
      if (out_tensors_[0]->data() == nullptr) {
        out_tensors_[0]->MallocData();
        if (out_tensors_[0]->data() == nullptr) {
          MS_LOG(ERROR) << "fail to allocate " << out_tensors_[0]->Size() << "Bytes for host";
          return lite::RET_ERROR;
        }
      }
      break;
    }
    case ops::AscendNativeCopy::CopyFormatType::NONE: {
      MS_LOG(WARNING) << "Ascend native copy kernel type is none. Kernel is redundant.";
      break;
    }
    default: {
      MS_LOG(ERROR) << "Ascend native copy kernel execute - copy type not supported.";
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}
// print device tensor
void PrintDeviceTensor(InferTensor *tensor, int max_len, void *stream, void *ctx) {
  int elem = std::min(static_cast<int>(tensor->ElementsNum()), max_len);
  std::cout << "device tensor " << tensor->tensor_name() << "elem " << elem << std::endl;
  switch (tensor->data_type()) {
    case kNumberTypeFloat16:
      ascend_native::PrintFp16(tensor->device_data(), elem, stream);
      break;
    case kNumberTypeFloat32:
      ascend_native::PrintFp32(tensor->device_data(), elem, stream);
      break;
    case kNumberTypeInt32:
      ascend_native::PrintInt32(tensor->device_data(), elem, stream);
      break;
    default:
      std::cout << "not supported type " << tensor->data_type() << std::endl;
  }
  std::cout << std::endl;
}

// print host tensor
void PrintHostTensor(InferTensor *tensor, int max_len, void *stream) {
  int elem = std::min(static_cast<int>(tensor->ElementsNum()), max_len);
  std::cout << "host tensor " << tensor->tensor_name() << "elem" << elem << std::endl;
  switch (tensor->data_type()) {
    case kNumberTypeFloat16: {
      ascend_native::PrintFp16Host(tensor->data(), elem, stream);
      break;
    }
    case kNumberTypeFloat32: {
      auto ptr = static_cast<float *>(tensor->data());
      for (int i = 0; i < elem; i++) {
        std::cout << *(ptr + i) << " ";
      }
      break;
    }
    case kNumberTypeInt32: {
      auto ptr = static_cast<int *>(tensor->data());
      for (int i = 0; i < elem; i++) {
        std::cout << *(ptr + i) << " ";
      }
      break;
    }
    default:
      std::cout << "not supported type " << tensor->data_type() << std::endl;
  }
  std::cout << std::endl;
}

int AscendNativeCopyKernel::Run() {
  MS_LOG(INFO) << "AscendNativeCopyKernel::Execute";
  ascend_native::SetContext(const_cast<void *>(acl_ctx_));
  auto elem = out_tensors_[0]->ElementsNum();
  // Execute copy
  if ((in_tensors_[0]->tensor_name().compare(mask_tensor_name_)) == 0) {
    return lite::RET_OK;
  }

  switch (copy_type_) {
    case ops::AscendNativeCopy::CopyFormatType::HOST_DEVICE: {
      if (in_tensors_[0]->data() == nullptr) {
        MS_LOG(ERROR) << "no host data to tensor " << in_tensors_[0]->tensor_name();
        return lite::RET_ERROR;
      }
      void *dst = out_tensors_[0]->device_data();
      if (dst == nullptr) {
        MS_LOG(ERROR) << "no output tensor allocated";
        return lite::RET_ERROR;
      }
      bool t_is_float =
        (in_tensors_[0]->data_type() == kNumberTypeFloat || in_tensors_[0]->data_type() == kNumberTypeFloat32);
      if (t_is_float) {
        ascend_native::CopyHostFp32ToDeviceFp16(in_tensors_[0]->data(), &dst, elem, const_cast<void *>(stream_));
      } else {
        int elem_size = mindspore::lite::DataTypeSize(in_tensors_[0]->data_type());
        switch (elem_size) {
          case Num4:
            ascend_native::CopyHostFp32ToDeviceFp32(in_tensors_[0]->data(), &dst, elem, const_cast<void *>(stream_));
            break;
          case Num2:
            ascend_native::CopyHostFp16ToDeviceFp16(in_tensors_[0]->data(), &dst, elem, const_cast<void *>(stream_));
            break;
          case Num1:
            ascend_native::CopyHostFp16ToDeviceFp16(in_tensors_[0]->data(), &dst, elem / 2,
                                                    const_cast<void *>(stream_));
            break;
          default:
            MS_LOG(ERROR) << "no supported size " << elem_size;
            return lite::RET_ERROR;
        }
      }
      break;
    }
    case ops::AscendNativeCopy::CopyFormatType::DEVICE_HOST: {
      if (in_tensors_[0]->device_data() == nullptr) {
        MS_LOG(ERROR) << "no device data to tensor " << in_tensors_[0]->tensor_name();
        return lite::RET_ERROR;
      }
      int elem_size = mindspore::lite::DataTypeSize(out_tensors_[0]->data_type());
      switch (elem_size) {
        case Num4:
          ascend_native::CopyDeviceFp16ToHostFp32(in_tensors_[0]->device_data(), out_tensors_[0]->data(), elem,
                                                  const_cast<void *>(stream_));
          break;
        case Num2:
          ascend_native::CopyDeviceFp16ToHostFp16(in_tensors_[0]->device_data(), out_tensors_[0]->data(), elem,
                                                  const_cast<void *>(stream_));
          break;
      }
      break;
    }
    case ops::AscendNativeCopy::CopyFormatType::NONE: {
      MS_LOG(WARNING) << "Ascend native copy kernel type is none. Kernel is redundant.";
      break;
    }
    default: {
      MS_LOG(ERROR) << "Ascend native copy kernel execute - copy type not supported. " << copy_type_;
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

int AscendNativeCopyKernel::PostProcess() {
  if ((in_tensors_[0]->tensor_name().compare("attention_mask")) == 0) {
    return lite::RET_OK;
  }
  switch (copy_type_) {
    case ops::AscendNativeCopy::CopyFormatType::HOST_DEVICE: {
      in_tensors_[0]->DecRefCount();
      break;
    }
    case ops::AscendNativeCopy::CopyFormatType::DEVICE_HOST: {
      auto ref = in_tensors_[0]->ref_count() - 1;
      in_tensors_[0]->set_ref_count(ref);
      if (ref < 0) {
        MS_LOG(ERROR) << "less than zero reference count";
        return lite::RET_ERROR;
      }
      break;
    }
    case ops::AscendNativeCopy::CopyFormatType::NONE: {
      MS_LOG(WARNING) << "Ascend native copy kernel type is none. Kernel is redundant.";
      break;
    }
    default: {
      MS_LOG(ERROR) << "Ascend native copy kernel execute - copy type not supported.";
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

int AscendNativeCopyKernel::ReSize() { return lite::RET_OK; }

REGISTER_ASCEND_NATIVE_CREATOR(ops::kNameAscendNativeCopy, AscendNativeCopyKernel)
}  // namespace mindspore::kernel
