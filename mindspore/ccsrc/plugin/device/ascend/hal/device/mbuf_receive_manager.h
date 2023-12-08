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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_MBUF_RECEIVE_MANAGER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_MBUF_RECEIVE_MANAGER_H_

#include <atomic>
#include <string>
#include <vector>
#include <queue>
#include <thread>
#include <functional>
#include <mutex>
#include <memory>
#include <utility>
#include <future>
#include <condition_variable>
#include "acl/acl_tdt.h"
#include "ir/tensor.h"

namespace mindspore::device::ascend {

using MbufFuncType = std::function<void(acltdtDataset *)>;

enum class MbufReceiveError : int {
  Success = 0,
  Timeout = 1,
  AclError = 2,
};

class ScopeAclTdtDataset {
 public:
  ScopeAclTdtDataset() { acl_dataset_ = acltdtCreateDataset(); }
  acltdtDataset *Get() const { return acl_dataset_; }
  ~ScopeAclTdtDataset() {
    if (acl_dataset_ != nullptr && acltdtDestroyDataset(acl_dataset_) != ACL_SUCCESS) {
      MS_LOG(ERROR) << "AcltdtDestroyDataset failed.";
    } else {
      MS_LOG(INFO) << "AcltdtDestroyDataset succeed.";
    }
  }

 private:
  acltdtDataset *acl_dataset_;
};

class MbufDataHandler {
 public:
  MbufDataHandler(MbufFuncType func, uint32_t device_id, string channel_name, size_t capacity = 128,
                  int32_t timeout = 800);
  ~MbufDataHandler();
  string GetChannelName() { return channel_name_; }
  uint32_t GetDeviceId() { return device_id_; }
  size_t GetCapacity() { return capacity_; }
  void StopReceive() { stop_receive_.store(true, std::memory_order_acq_rel); }

 private:
  MbufFuncType func_;
  uint32_t device_id_;
  std::string channel_name_;
  size_t capacity_;
  int32_t timeout_;
  std::mutex mutex_;
  std::atomic_bool stop_receive_{false};
  std::thread thread_;
  std::promise<MbufReceiveError> promise_;
  std::future<MbufReceiveError> future_;
  acltdtChannelHandle *acl_handle_;

  void HandleData();
  bool ReceiveAndProcessData(const ScopeAclTdtDataset &scope_acl_dataset);
  bool QueryChannelSize(size_t *queue_size);
};

class MbufDataHandlerManager {
 public:
  static MbufDataHandlerManager &GetInstance() {
    static MbufDataHandlerManager instance;
    return instance;
  }
  ~MbufDataHandlerManager() = default;
  MbufDataHandlerManager(const MbufDataHandlerManager &) = delete;
  MbufDataHandlerManager &operator=(const MbufDataHandlerManager &) = delete;

  void AddHandler(std::unique_ptr<MbufDataHandler> handler) { handles_.push_back(std::move(handler)); }
  void DestoryHandler() {
    for (auto &handle : handles_) {
      handle->StopReceive();
    }
    while (!handles_.empty()) {
      MS_LOG(INFO) << "The thread of " << handles_.back()->GetChannelName() << " channel is being destroyed.";
      handles_.pop_back();
    }
  }

 private:
  MbufDataHandlerManager() = default;
  std::vector<std::unique_ptr<MbufDataHandler>> handles_;
};

}  // namespace mindspore::device::ascend
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_TENSORDUMP_UTILS_H_