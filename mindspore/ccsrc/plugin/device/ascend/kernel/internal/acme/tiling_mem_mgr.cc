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

#include "plugin/device/ascend/kernel/internal/acme/tiling_mem_mgr.h"

#include "mindspore/core/utils/ms_context.h"
#include "mindspore/ccsrc/runtime/hardware/device_context_manager.h"
#include "plugin/device/ascend/hal/device/ascend_memory_pool.h"
#include "acl/acl.h"

namespace mindspore {
namespace kernel {
size_t TilingMemPool::GetAlignedSize(size_t size) { return (size + block_size_ - 1) & ~(block_size_ - 1); }

int TilingMemPool::Init() { return 0; }

std::pair<MemoryType, void *> TilingMemPool::Malloc(size_t size) {
  size = GetAlignedSize(size);
  bool old_created_flag = pool_mem_created_;
  if (mem_base_ptr_ == nullptr) {
    if (pool_mem_created_.compare_exchange_strong(old_created_flag, true)) {
      mem_base_ptr_ = static_cast<int8_t *>(MallocInner(block_size_ * block_num_));
      MS_EXCEPTION_IF_NULL(mem_base_ptr_);
    } else {
      // wait other thread to init mem_base_ptr_
      while (mem_base_ptr_ == nullptr) {
      }
    }
  }

  if (offset_ + size <= total_size_) {
    size_t old_offset = offset_;
    const size_t kMaxTrailCount = 100000;
    size_t loop_count = 0;
    while (!offset_.compare_exchange_strong(old_offset, old_offset + size)) {
      loop_count++;
      if (loop_count == kMaxTrailCount) {
        MS_LOG(EXCEPTION) << "Trying too many times while allocing memory from TilingMemPool.";
      }
    }

    if (old_offset + size > total_size_) {
      // memory in pool is not enough
      return MallocOneOffMem(size);
    }

    return std::make_pair(kMemoryCached, mem_base_ptr_ + old_offset);
  }

  // memory in pool is not enough
  return MallocOneOffMem(size);
}

void TilingMemPool::Free(MemoryType mem_type, void *addr) {
  if (mem_type == kMemoryOneOff) {
    FreeInner(addr);
  }
}

void TilingMemPool::Free(void *addr) {
  if (addr < mem_base_ptr_ || addr >= mem_base_ptr_ + total_size_) {
    // the addr is not in pool
    FreeInner(addr);
  }
}

TilingMemPool::~TilingMemPool() {
  if (mem_base_ptr_ != nullptr) {
    FreeInner(mem_base_ptr_);
  }
}

int TilingMemPoolDevice::Init() { return 0; }

void *TilingMemPoolDevice::MallocInner(size_t size) {
  void *device_addr;
  (void)device::ascend::AscendMemoryPool::GetInstance().AllocDeviceMem(size, &device_addr);
  return device_addr;
}

void TilingMemPoolDevice::FreeInner(void *addr) {
  (void)device::ascend::AscendMemoryPool::GetInstance().FreeDeviceMem(&addr);
}

int TilingMemPoolHost::Init() { return 0; }

void *TilingMemPoolHost::MallocInner(size_t size) { return malloc(size); }

void TilingMemPoolHost::FreeInner(void *addr) { free(addr); }

TilingMemMgr::TilingMemMgr() {
  auto context_ptr = mindspore::MsContext::GetInstance();
  uint32_t device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  std::string device_name = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device_context_ = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_name, device_id});
}

void TilingMemMgr::CopySync(void *host_ptr, void *device_ptr, size_t size) {
  device_context_->device_res_manager_->BindDeviceToCurrentThread(false);

  auto ret = aclrtMemcpy(device_ptr, size, host_ptr, size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "Copy tiling data from host to device failed!";
  }
}

void TilingMemMgr::CopySyncD2H(void *host_ptr, void *device_ptr, size_t size) {
  device_context_->device_res_manager_->BindDeviceToCurrentThread(false);

  auto ret = aclrtMemcpy(host_ptr, size, device_ptr, size, ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "Copy tiling data from host to device failed!";
  }
}

}  // namespace kernel
}  // namespace mindspore
