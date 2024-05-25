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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_KERNEL_PACKET_ENGINE_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_KERNEL_PACKET_ENGINE_H_

#include <string>
#include <memory>
#include "include/common/symbol_engine/symbol_engine_impl.h"

namespace mindspore {
namespace graphkernel {
namespace symshape {
using mindspore::symshape::SymbolEngineImpl;

/// \brief SymbolEngine for kernel packet graph.
class KernelPacketEngine : public SymbolEngineImpl {
 public:
  using SymbolEngineImpl::SymbolEngineImpl;
  ~KernelPacketEngine() = default;
  MS_DECLARE_PARENT(KernelPacketEngine, SymbolEngineImpl)

  std::string ToString() const override { return "KernelPacketEngine_" + name_; }
  static std::shared_ptr<symshape::KernelPacketEngine> Build(const FuncGraphPtr &func_graph);
};
}  // namespace symshape
using KernelPacketEnginePtr = std::shared_ptr<symshape::KernelPacketEngine>;
}  // namespace graphkernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_KERNEL_PACKET_ENGINE_H_
