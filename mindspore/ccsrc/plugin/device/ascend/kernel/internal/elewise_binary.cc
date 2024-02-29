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
#include "plugin/device/ascend/kernel/internal/elewise_binary.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include <memory>

namespace mindspore {
namespace kernel {
internal::OpParamPtr ElewiseBinary::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &outputs) {
  internal::OpParamPtr param_ptr = std::make_shared<internal::OpParam>();
  SetComputeType(param_ptr);
  return param_ptr;
}

void ElewiseBinary::SetInOutIdx() {
  inputsIdxMap_[0] = 0;
  inputsIdxMap_[1] = 1;
  outputsIdxMap_[0] = 0;
}

class InternalAdd : public ElewiseBinary {
 public:
  InternalAdd() : ElewiseBinary("Add") {}
  ~InternalAdd() = default;

 protected:
  void SetComputeType(internal::OpParamPtr param_ptr) override {
    param_ptr->opId = internal::OpId::Add;
    internal::ElewiseParam op_param;
    op_param.elewiseType = internal::ElewiseParam::ELEWISE_ADD;
    param_ptr->specificParam = op_param;
  }
  uint64_t GenTilingCacheKey(const std::vector<KernelTensor *> &inputs,
                             const std::vector<KernelTensor *> &outputs) override {
    return TilingCacheMgr::GetInstance().GenTilingCacheKey(kernel_name_, inputs[0]->GetShapeVector(),
                                                           inputs[0]->dtype_id(), inputs[1]->GetShapeVector(),
                                                           inputs[1]->dtype_id());
  }
};

class InternalSub : public ElewiseBinary {
 public:
  InternalSub() : ElewiseBinary("Sub") {}
  ~InternalSub() = default;

 protected:
  void SetComputeType(internal::OpParamPtr param_ptr) override {
    param_ptr->opId = internal::OpId::Sub;
    internal::ElewiseParam op_param;
    op_param.elewiseType = internal::ElewiseParam::ELEWISE_SUB;
    param_ptr->specificParam = op_param;
  }
};

class InternalEqual : public ElewiseBinary {
 public:
  InternalEqual() : ElewiseBinary("Equal") {}
  ~InternalEqual() = default;

 protected:
  void SetComputeType(internal::OpParamPtr param_ptr) override {
    param_ptr->opId = internal::OpId::Equal;
    internal::ElewiseParam op_param;
    op_param.elewiseType = internal::ElewiseParam::ELEWISE_EQUAL;
    param_ptr->specificParam = op_param;
  }
};

class InternalLess : public ElewiseBinary {
 public:
  InternalLess() : ElewiseBinary("Less") {}
  ~InternalLess() = default;

 protected:
  void SetComputeType(internal::OpParamPtr param_ptr) override {
    param_ptr->opId = internal::OpId::Less;
    internal::ElewiseParam op_param;
    op_param.elewiseType = internal::ElewiseParam::ELEWISE_LESS;
    param_ptr->specificParam = op_param;
  }
};

class InternalMul : public ElewiseBinary {
 public:
  InternalMul() : ElewiseBinary("Mul") {}
  ~InternalMul() = default;

 protected:
  void SetComputeType(internal::OpParamPtr param_ptr) override {
    param_ptr->opId = internal::OpId::Mul;
    internal::ElewiseParam op_param;
    op_param.elewiseType = internal::ElewiseParam::ELEWISE_MUL;
    param_ptr->specificParam = op_param;
  }
};

class InternalRealDiv : public ElewiseBinary {
 public:
  InternalRealDiv() : ElewiseBinary("RealDiv") {}
  ~InternalRealDiv() = default;

 protected:
  void SetComputeType(internal::OpParamPtr param_ptr) override {
    param_ptr->opId = internal::OpId::RealDiv;
    internal::ElewiseParam op_param;
    op_param.elewiseType = internal::ElewiseParam::ELEWISE_REALDIV;
    param_ptr->specificParam = op_param;
  }
};

MS_INTERNAL_KERNEL_FACTORY_REG(Add, InternalAdd);
MS_INTERNAL_KERNEL_FACTORY_REG(Sub, InternalSub);
MS_INTERNAL_KERNEL_FACTORY_REG(Equal, InternalEqual);
MS_INTERNAL_KERNEL_FACTORY_REG(Less, InternalLess);
MS_INTERNAL_KERNEL_FACTORY_REG(Mul, InternalMul);
MS_INTERNAL_KERNEL_FACTORY_REG(RealDiv, InternalRealDiv);
}  // namespace kernel
}  // namespace mindspore
