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
#include "plugin/device/ascend/hal/hardware/ge_kernel_executor.h"
#include <utility>
#include <algorithm>
#include "include/common/utils/parallel_context.h"
#include "include/common/profiler.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "backend/common/session/kernel_graph_mgr.h"
#include "ops/auto_generate/gen_ops_primitive.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "plugin/device/ascend/hal/hardware/ge_graph_optimization.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "plugin/device/ascend/hal/hardware/acl_somas.h"
#include "plugin/device/ascend/hal/hardware/acl_stream_assign.h"
#include "plugin/device/ascend/kernel/rts/rt_kernel_build.h"
#include "plugin/device/ascend/kernel/hccl/hccl_kernel_metadata.h"
#include "plugin/device/ascend/kernel/hccl/hccl_kernel_build.h"
#include "plugin/device/ascend/kernel/pyboost/customize/customize_copy.h"

#ifdef ENABLE_DVM
#include "plugin/device/ascend/kernel/dvm/dvm_kernel_build.h"
#endif

#ifndef ENABLE_SECURITY
#include "include/backend/debug/data_dump/dump_json_parser.h"
#include "include/backend/optimizer/helper.h"
#include "plugin/device/ascend/hal/device/kernel_select_ascend.h"
#include "plugin/device/ascend/kernel/opapi/aclnn_kernel_build.h"
#include "plugin/device/ascend/kernel/acl/acl_kernel_build.h"
#include "plugin/device/ascend/kernel/host/host_kernel_build.h"
#include "plugin/device/ascend/kernel/host/host_kernel_metadata.h"
#include "kernel/kernel_build_info.h"
#include "transform/acl_ir/acl_helper.h"
#include "transform/acl_ir/op_api_util.h"
#include "transform/acl_ir/ge_adapter_info.h"
#include "transform/symbol/acl_compiler_symbol.h"
#include "transform/symbol/acl_rt_symbol.h"
#include "transform/symbol/symbol_utils.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/backend/debug/data_dump/overflow_dumper.h"
#include "include/backend/debug/profiler/profiling.h"
#include "plugin/device/ascend/hal/profiler/profiling_framework_data.h"
#include "utils/anf_utils.h"
#endif

namespace mindspore::device::ascend {
namespace {
bool GenerateKernelMod(const std::vector<CNodePtr> &kernels) {
  for (const auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    if (AnfAlgo::GetKernelMod(kernel)) {
      continue;
    }
    if (AnfAlgo::IsKernelSelectBackoffOp(kernel)) {
      continue;
    }
    kernel::KernelModPtr kernel_mod_ptr = nullptr;
    auto kernel_type = AnfAlgo::GetKernelType(kernel);
    if (kernel_type == KernelType::ACL_KERNEL) {
      kernel_mod_ptr = kernel::AclOpBuild(kernel);
    } else if (kernel_type == KernelType::HOST_KERNEL) {
      kernel_mod_ptr = kernel::HostOpBuild(kernel);
    } else if (kernel_type == KernelType::HCCL_KERNEL) {
      kernel_mod_ptr = kernel::HcclOpBuild(kernel);
    } else if (kernel_type == KernelType::OPAPI_KERNEL) {
      kernel_mod_ptr = kernel::AclnnOpBuild(kernel);
#ifdef ENABLE_DVM
    } else if (kernel_type == KernelType::AKG_KERNEL) {
      kernel_mod_ptr = kernel::DvmOpBuild(kernel);
#endif
    } else {
      MS_LOG(EXCEPTION) << "The kernel: " << kernel->fullname_with_scope() << " kernel build failed, kernel type: "
                        << kernel::KernelTypeLabel(AnfAlgo::GetKernelType(kernel));
    }
    MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
    AnfAlgo::SetKernelMod(kernel_mod_ptr, kernel.get());
  }
  return true;
}

bool GraphWithNoRealKernel(const KernelGraphPtr &kernel_graph) {
  const auto &nodes = kernel_graph->execution_order();
  for (auto &node : nodes) {
    if (AnfUtils::IsRealKernel(node)) {
      return false;
    }
  }
  return true;
}

void SetAclOpPrecisionMode() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);

  auto precision_mode = ms_context->get_param<std::string>(MS_CTX_PRECISION_MODE);
  if (precision_mode.empty()) {
    precision_mode = (transform::AclUtil::KeepOriginDType() == 1) ? "must_keep_origin_dtype" : "allow_fp32_to_fp16";
  }
  MS_LOG(INFO) << "Set aclop PRECISION_MODE: " << precision_mode;
  auto ret = CALL_ASCEND_API(aclSetCompileopt, aclCompileOpt::ACL_PRECISION_MODE, precision_mode.c_str());
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Acl set precision mode failed! Error flag is " << ret;
  }

  auto op_precision_mode = ms_context->get_param<std::string>(MS_CTX_OP_PRECISION_MODE);
  if (op_precision_mode.empty()) {
    return;
  }
  MS_LOG(INFO) << "Set aclop OP_PRECISION_MODE: " << op_precision_mode;
  ret = CALL_ASCEND_API(aclSetCompileopt, aclCompileOpt::ACL_OP_PRECISION_MODE, op_precision_mode.c_str());
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Acl set op precision mode failed! Error flag is " << ret;
  }
}

void SelectKernel(const KernelGraphPtr &kernel_graph, std::set<KernelGraphPtr> *const memo) {
  // select kernel
  MS_EXCEPTION_IF_NULL(memo);
  if (memo->find(kernel_graph) != memo->end()) {
    return;
  }
  memo->insert(kernel_graph);
  const auto &kernels = kernel_graph->execution_order();
  for (const auto &kernel : kernels) {
    auto [select_res, msg, etype] = device::ascend::SelectKernelInfoWithMsg(kernel_graph, kernel);
    if (!select_res) {
      MS_LOG(INFO) << "node is " << kernel->fullname_with_scope() << " should backoff";
      std::pair<std::string, ExceptionType> failure_info = std::make_pair(msg, etype);
      device::ascend::HandleKernelSelectFailure(kernel_graph, kernel, failure_info);
    }
  }
  if (!kernel_graph->is_from_single_op()) {
    kernel_graph->SetKernelObjectTypesForUnrealNodes();
  }
  for (auto &child_graph : kernel_graph->child_graph_order()) {
    SelectKernel(child_graph.lock(), memo);
  }
}

void InlineSubGraph(const KernelGraphPtr &graph) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
#ifdef ENABLE_DUMP_IR
  bool save_graphs = context_ptr->CanDump(kIntroductory);
  if (save_graphs) {
    std::string file_name = "hwopt_d_before_inline_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph, true, kWholeStack);
  }
#endif
  auto kernel_cnodes = graph->execution_order();
  AnfNodePtr last_call = nullptr;
  for (auto &kernel_cnode : kernel_cnodes) {
    MS_EXCEPTION_IF_NULL(kernel_cnode);
    if (common::AnfAlgo::CheckPrimitiveType(kernel_cnode, prim::kPrimCallInline)) {
      auto sub_graph = common::AnfAlgo::GetNodeAttr<KernelGraphPtr>(kernel_cnode, kAttrKernelGraph);
      MS_EXCEPTION_IF_NULL(sub_graph);
      MS_LOG(INFO) << "InlineSubGraph: " << kernel_cnode->fullname_with_scope()
                   << ", sub graph: " << sub_graph->graph_id() << ", need inline: " << sub_graph->need_inline();
      auto main_graph = kernel_cnode->func_graph();
      MS_EXCEPTION_IF_NULL(main_graph);
      auto mng = main_graph->manager();
      auto kernel_info = dynamic_cast<device::KernelInfo *>(kernel_cnode->kernel_info());
      MS_EXCEPTION_IF_NULL(kernel_info);
      AnfNodePtrList inp;
      auto &call_input = kernel_cnode->inputs();
      // let operators on different subgraphs will not be executed interleavedly
      for (size_t i = 1; i < call_input.size(); i++) {
        if (last_call != nullptr) {
          auto depend = graph->NewCNode({NewValueNode(prim::kPrimDepend), call_input[i], last_call});
          MS_EXCEPTION_IF_NULL(depend);
          depend->set_abstract(call_input[i]->abstract());
          inp.push_back(depend);
        } else {
          inp.push_back(call_input[i]);
        }
      }
      const auto &ref_map = sub_graph->GetRefMap();
      auto out = session::KernelGraphMgr::DoInline(sub_graph, main_graph, inp, kernel_cnode->input(0)->scope(),
                                                   kernel_info->graph_id(), ref_map, graph);
      (void)mng->Replace(kernel_cnode, out);
      // Inline graph boundary: MakeTuple---->Depend---->Tensormove
      // Avoid long link times at runtime
      auto value_node = graph->NewValueNode(MakeValue(std::make_shared<tensor::Tensor>(1)));
      MS_EXCEPTION_IF_NULL(value_node);
      auto depend = graph->NewCNode({NewValueNode(prim::kPrimDepend), value_node, out});
      MS_EXCEPTION_IF_NULL(depend);
      depend->set_abstract(value_node->abstract());
      auto tensor_move = graph->NewCNode({NewValueNode(prim::kPrimTensorMove), depend});
      MS_EXCEPTION_IF_NULL(tensor_move);
      tensor_move->set_abstract(value_node->abstract());
      common::AnfAlgo::SetNodeAttr(kAttrKernelGraphBoundary, MakeValue(sub_graph), tensor_move);

      // select kernel
      auto [select_res, msg, etype] = device::ascend::SelectKernelInfoWithMsg(graph, tensor_move);
      if (!select_res) {
        MS_LOG(INFO) << "node is " << tensor_move->fullname_with_scope() << " should backoff";
        std::pair<std::string, ExceptionType> failure_info = std::make_pair(msg, etype);
        device::ascend::HandleKernelSelectFailure(graph, tensor_move, failure_info);
      }

      last_call = tensor_move;
    }
  }
  GEGraphOptimization::GetInstance().OptimizeACLGraphAfterInline(graph);
#ifdef ENABLE_DUMP_IR
  if (save_graphs) {
    std::string file_name = "hwopt_d_after_inline_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph, true, kWholeStack);
  }
#endif
}
}  // namespace

void GeKernelExecutor::Initialize() {
  if (initialized_) {
    return;
  }
  InitializeAcl();
  MS_EXCEPTION_IF_NULL(device_context_);
  res_manager_ = device_context_->device_res_manager_.get();
  MS_EXCEPTION_IF_NULL(res_manager_);
  graph_executor_ = dynamic_cast<GeGraphExecutor *>(device_context_->graph_executor_.get());
  // not check graph executor, may use in ascend device context
  SetAclOpPrecisionMode();
  transform::AclUtil::SetDeterministic();
  initialized_ = true;
}

void GeKernelExecutor::Destroy() {
  if (!initialized_) {
    return;
  }
  res_manager_ = nullptr;
  graph_executor_ = nullptr;
  initialized_ = false;
}

void GeKernelExecutor::UnifyMindIR(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  GEGraphOptimization::GetInstance().UnifyMindIR(graph);
}

void GeKernelExecutor::AddMindIRPass(const KernelGraphPtr &graph) const {
  GEGraphOptimization::GetInstance().GEMindIRPass(graph);
}

void GeKernelExecutor::OptimizeGraph(const FuncGraphPtr &graph) const {
  // will be cached by OpCompileInfo
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  // GE graph run mode do optimize in ProcessBeforeRun
  if (kernel_graph->is_graph_run_mode() && IsEnableRefMode()) {
    return;
  }
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "GeOptimizeGraph", 1, 0, 0);
  std::set<KernelGraphPtr> memo;
  GEGraphOptimization::GetInstance().OptimizeACLGraph(kernel_graph, &memo);
  memo.clear();
  SelectKernel(kernel_graph, &memo);
  memo.clear();
  GEGraphOptimization::GetInstance().OptimizeACLGraphAfterKernelSelect(kernel_graph, &memo);
  memo.clear();
  InlineSubGraph(kernel_graph);
  OptimizeExecutionOrder(NOT_NULL(graph));
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "GeOptimizeGraph", 1, 0, 1);
}

void GeKernelExecutor::CreateKernel(const std::vector<CNodePtr> &nodes) const {
  if (!nodes.empty() && IsEnableRefMode()) {
    auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(nodes[0]->func_graph());
    MS_EXCEPTION_IF_NULL(kernel_graph);
    // Not create kernel when use GE
    if (!kernel_graph->is_from_single_op() && kernel_graph->is_graph_run_mode()) {
      return;
    }
  }
  // build kernel mod
  MS_LOG(DEBUG) << "Status record: start create kernel.";
  profiler::CollectHostInfo("Ascend", "CreateKernel", "CreateGeKernel", 1, 0, 0);
  PROF_START(create_kernel);
  device::ascend::SetKernelInfoBeforeCreateKernel(nodes);
  auto ret = GenerateKernelMod(nodes);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Kernel build error.";
  }
  PROF_END(create_kernel);
  profiler::CollectHostInfo("Ascend", "CreateKernel", "CreateGeKernel", 1, 0, 1);
  MS_LOG(DEBUG) << "Status record: end create kernel.";
}

namespace {
void CreateEventKernelMod(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto nodes = kernel_graph->execution_order();
  for (auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!IsOneOfPrimitiveCNode(node, {prim::kPrimStreamSend, prim::kPrimStreamRecv})) {
      continue;
    }
    device::ascend::GenerateKernelBuildInfo(node, RT_KERNEL);
    auto kernel_mod_ptr = kernel::RtOpBuild(node);
    MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
    AnfAlgo::SetKernelMod(kernel_mod_ptr, node.get());
  }
}
}  // namespace

void GeKernelExecutor::DoStreamAssign(const KernelGraphPtr &kernel_graph) {
  MS_LOG(DEBUG) << "Status record: start stream assign.";
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  // stream assign
  AclStreamAssign::GetInstance().AssignStream(NOT_NULL(kernel_graph));
  CreateEventKernelMod(kernel_graph);
#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->CanDump(kIntroductory);
  if (save_graphs) {
    std::string file_name = "hwopt_d_after_stream_assign_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  kernel_graph->PrintGraphExecuteOrder();
  MS_LOG(DEBUG) << "Status record: end stream assign.";
}

void GeKernelExecutor::DoSomas(const FuncGraphPtr &graph) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  DoStreamAssign(kernel_graph);
  // somas
  MS_LOG(DEBUG) << "Status record: start do somas.";
  if (ms_context->get_param<int>(MS_CTX_MEMORY_OPTIMIZE_LEVEL) != kOptimizeO0) {
    auto somas = std::make_shared<AclSomas>();
    bool ret = somas->Assign(kernel_graph);
    if (ret) {
      MS_LOG(INFO) << "Somas allocate success for graph " << kernel_graph->graph_id()
                   << " somas size: " << kernel_graph->somas_whole_block_size();
    } else if (somas->IsSupportSomas(*kernel_graph)) {
      MS_LOG(WARNING) << "Somas allocate failed for graph " << kernel_graph->graph_id();
    }
  }
  MS_LOG(DEBUG) << "Status record: end do somas.";
}

void GeKernelExecutor::OptimizeExecutionOrder(const FuncGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(DEBUG) << "Status record: start optimize execution order. graph id: " << kernel_graph->graph_id();
  auto execution_order = kernel_graph->execution_order();
  kernel_graph->EnableRuntimeCache();
  common::AnfAlgo::ReorderExecList(NOT_NULL(&execution_order));
  kernel_graph->DisableRuntimeCache();
  kernel_graph->set_execution_order(execution_order);
  MS_LOG(DEBUG) << "Status record: end optimize execution order. graph id: " << kernel_graph->graph_id();
}

void GeKernelExecutor::PreprocessBeforeRun(const FuncGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  profiler::CollectHostInfo("Ascend", "PreprocessBeforeRun", "GePreprocess", 1, 0, 0);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);

  // use GE
  if (kernel_graph->is_graph_run_mode() && IsEnableRefMode()) {
    if (GraphWithNoRealKernel(kernel_graph)) {
      return;
    }
    MS_EXCEPTION_IF_NULL(graph_executor_);
    graph_executor_->PreprocessBeforeRun(kernel_graph);
    profiler::CollectHostInfo("Ascend", "PreprocessBeforeRun", "GePreprocess", 1, 0, 1);
    return;
  }

  // nop op -> memcpy
  const auto &nodes = kernel_graph->execution_order();
  for (const auto &node : nodes) {
    auto op_name = common::AnfAlgo::GetCNodeName(node);
    // If the 2nd input of reshape is not a value node, then there are two inputs to select the host reshape operator
    bool is_host_reshape_op = false;
    if (op_name == prim::kPrimReshape->name()) {
      auto kernel_mod = AnfAlgo::GetKernelMod(node);
      MS_EXCEPTION_IF_NULL(kernel_mod);
      is_host_reshape_op = kernel_mod->GetKernelModType() == kernel::KernelModType::HostKernelMod;
    }
    bool is_nop_op = transform::AclHelper::IsNopNode(node);
    bool is_transpose_nop = (op_name == prim::kPrimTranspose->name() || op_name == prim::kPrimTransposeD->name()) &&
                            common::AnfAlgo::HasNodeAttr(kAttrNopOp, node);
    if (is_transpose_nop || (is_nop_op && !is_host_reshape_op)) {
      nop_op_to_memcpy_.insert(node);
    }
  }

  DoSomas(NOT_NULL(graph));

  profiler::CollectHostInfo("Ascend", "PreprocessBeforeRun", "GePreprocess", 1, 0, 1);
}

bool GeKernelExecutor::PySyncRuning(void *stream) const {
  MS_EXCEPTION_IF_NULL(res_manager_);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE) &&
      !AscendStreamMng::GetInstance().SyncStream(stream)) {
    return false;
  }
  return true;
}

bool GeKernelExecutor::MemoryCopyAsync(const CNodePtr &node, const vector<KernelTensor *> &inputs,
                                       const vector<KernelTensor *> &outputs) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(DEBUG) << "Launch MemoryCopyAsync instead for kernel " << node->fullname_with_scope();
  if (inputs.size() != 1 || outputs.size() != 1) {
    MS_LOG(DEBUG) << "Kernel " << node->fullname_with_scope() << " input output size should be 1 but"
                  << " input size is:" << inputs.size() << " output size is:" << outputs.size();
  }

  const auto stream = AscendStreamMng::GetInstance().GetStream(kDefaultStreamIndex);
  MS_EXCEPTION_IF_NULL(stream);
  aclError status = CALL_ASCEND_API(aclrtMemcpyAsync, outputs[0]->device_ptr(), outputs[0]->size(),
                                    inputs[0]->device_ptr(), inputs[0]->size(), ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
  if (status != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "MemCpyAsync op aclrtMemcpyAsync failed, ret:" << status << " destMax:" << outputs[0]->size()
                  << " count:" << inputs[0]->size();
    return false;
  }
  return true;
}

bool GeKernelExecutor::LaunchKernel(const CNodePtr &kernel, const vector<KernelTensor *> &inputs,
                                    const vector<KernelTensor *> &workspace, const vector<KernelTensor *> &outputs,
                                    KernelMod *kernel_mod, void *stream) const {
  profiler::ascend::ProfilingFrameworkData::RecordLaunchGETaskBegin(kernel);
  // launch kernel
  uint64_t start_time = 0;
  PROFILER_START(start_time);
  if (nop_op_to_memcpy_.find(kernel) != nop_op_to_memcpy_.end()) {
    if (!MemoryCopyAsync(kernel, inputs, outputs)) {
      MS_LOG(ERROR) << "Memory copy failed for kernel " << kernel->fullname_with_scope();
      return false;
    }
  } else {
    MS_LOG(DEBUG) << "Begin launch kernel: " << kernel->fullname_with_scope();
    MS_EXCEPTION_IF_NULL(kernel_mod);
    MS_EXCEPTION_IF_NULL(stream);
    bool ret = kernel_mod->Launch(inputs, workspace, outputs, stream);
    MS_LOG(DEBUG) << "End launch kernel: " << kernel->fullname_with_scope();
    if (!ret) {
      MS_LOG(ERROR) << "Launch kernel failed, kernel full name: " << kernel->fullname_with_scope();
      res_manager_->ResetStreamAndCtx();
      return false;
    }
  }
  profiler::ascend::ProfilingFrameworkData::RecordGETask(kernel);
  // for PyNative Sync Run mode
  auto ret = PySyncRuning(stream);
  PROFILER_END(start_time, runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kKernelLaunch,
               kernel->fullname_with_scope(), false);
  return ret;
}

void AclrtLaunchCallback(void *user_data) {
  CallbackFunc *callback_func = reinterpret_cast<CallbackFunc *>(user_data);
  (*callback_func)();
  delete callback_func;
}

bool GeKernelExecutor::LaunchCallback(CallbackFunc callback_func, size_t stream_id) const {
  auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  if (stream == nullptr) {
    stream_id = kDefaultStreamIndex;
    stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  }
  MS_EXCEPTION_IF_NULL(stream);
  auto callback_func_ptr = new CallbackFunc(callback_func);
  aclError ret = CALL_ASCEND_API(aclrtLaunchCallback, AclrtLaunchCallback, callback_func_ptr,
                                 aclrtCallbackBlockType::ACL_CALLBACK_NO_BLOCK, stream);
  MS_LOG(DEBUG) << "Launch callback for stream_id : " << stream_id << ", ret : " << ret << ".";
  if (ret) {
    delete callback_func_ptr;
    MS_LOG(WARNING) << "Launch callback for stream_id : " << stream_id << " failed, ret : " << ret << ".";
    if (res_manager_->SyncStream(stream_id)) {
      callback_func();
      return true;
    }
    res_manager_->ResetStreamAndCtx();
    return false;
  }
  return true;
}

bool GeKernelExecutor::ExecuteKernelTask(const runtime::KernelTaskType &task_type,
                                         const device::DeviceAddressPtrList &input_addr_list,
                                         const device::DeviceAddressPtrList &output_addr_list,
                                         const size_t &stream_id) const {
  MS_LOG(DEBUG) << "task_type:" << task_type;
  if (runtime::KernelTaskType::kCOPY_TASK == task_type) {
    constexpr size_t kCopyTaskInputsNum = 2;
    // Copy task is a in-place op, the output is the first input.
    // To reuse the aclnnInplaceCopy, the first input of Copy is used as the operator output,
    // and the second input is used as the operator input.
    if (input_addr_list.size() != kCopyTaskInputsNum) {
      MS_LOG(EXCEPTION) << "input_addr_list.size() is invalid, input_addr_list.size():" << input_addr_list.size();
    }
    kernel::pyboost::CustomizeCopyAscend(device_context_, input_addr_list[1], input_addr_list[0], stream_id);
  } else {
    // For contiguous task, there must be at least one input and one output.
    if (input_addr_list.empty() || output_addr_list.empty()) {
      MS_LOG(EXCEPTION) << "input_addr_list.size() or output_addr_list.size() is invalid, input_addr_list.size():"
                        << input_addr_list.size() << ", output_addr_list.size():" << output_addr_list.size();
    }
    kernel::pyboost::CustomizeCopyAscend(device_context_, input_addr_list[0], output_addr_list[0], stream_id);
  }

  return true;
}
}  // namespace mindspore::device::ascend
