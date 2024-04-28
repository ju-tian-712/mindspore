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

#include "frontend/parallel/pass/bias_add_comm_swap.h"
#include <memory>
#include <list>
#include <vector>
#include <string>
#include <utility>
#include "include/common/utils/utils.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/other_ops.h"

namespace mindspore {
namespace parallel {
namespace {
constexpr const char BIAS_ADD_COMM_SWAP[] = "bias_add_comm_swap";

bool IsSubRankList(const RankList &child_list, const RankList &parent_list) {
  for (auto &child : child_list) {
    if (std::find(parent_list.begin(), parent_list.end(), child) == parent_list.end()) {
      return false;
    }
  }
  return true;
}
bool IsAddNodeValid(const CNodePtr &add_node, const AnfNodePtr &comm_node) {
  OperatorInfoPtr add_distribute_operator = add_node->user_data<OperatorInfo>();
  if (add_distribute_operator == nullptr) {
    return false;
  }
  TensorInfo node_add_tensor_in = add_distribute_operator->inputs_tensor_info()[LongToSize(1)];
  TensorLayout node_add_tensor_layout = node_add_tensor_in.tensor_layout();
  auto node_add_rank_list = node_add_tensor_layout.InferRepeatedGroup();

  auto comm_prim = GetCNodePrimitive(comm_node);
  if (!comm_prim->HasAttr(GROUP)) {
    return false;
  }
  auto comm_group = GetValue<std::string>(comm_prim->GetAttr(GROUP));
  MS_EXCEPTION_IF_NULL(g_device_manager);
  auto comm_rank_list = g_device_manager->FindRankListByHashName(comm_group);
  return IsSubRankList(comm_rank_list, node_add_rank_list);
}

// find matmul node
AnfNodePtr FindMatMulNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto matmul_node = GetInputNodeWithFilter(node, [&](const CNodePtr &cnode) {
    bool filter = !IsPrimitiveCNode(cnode, prim::kPrimMatMul);
    return std::make_pair(filter, 1);
  });
  return matmul_node;
}

// find allreduce/reduce_scatter node
AnfNodePtr FindValidCommNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto comm_node = GetInputNodeWithFilter(node, [&](const AnfNodePtr &anode) {
    bool filter = !IsPrimitiveCNode(anode, prim::kPrimAllReduce) && !IsPrimitiveCNode(anode, prim::kPrimReduceScatter);
    return std::make_pair(filter, 1);
  });
  if (comm_node == nullptr ||
      (!IsPrimitiveCNode(comm_node, prim::kPrimAllReduce) && !IsPrimitiveCNode(comm_node, prim::kPrimReduceScatter))) {
    return nullptr;
  }
  auto matmul_node = FindMatMulNode(comm_node);
  if (matmul_node == nullptr || !IsPrimitiveCNode(matmul_node, prim::kPrimMatMul)) {
    return nullptr;
  }
  return comm_node;
}

void FindAllValidAddNode(const FuncGraphPtr &graph, HashMap<CNodePtr, AnfNodePtr> *add_node_map) {
  std::list<CNodePtr> graph_orders = graph->GetOrderedCnodes();
  std::vector<CNodePtr> origin_nodes_topological(graph_orders.cbegin(), graph_orders.cend());
  for (const auto &node : origin_nodes_topological) {
    if (!IsPrimitiveCNode(node, prim::kPrimAdd)) {
      MS_LOG(INFO) << "For cur node, it must be node add and its strategy must be all ones, but got "
                   << node->DebugString();
      continue;
    }

    auto comm_node = FindValidCommNode(node->cast<AnfNodePtr>());
    if (comm_node == nullptr) {
      MS_LOG(INFO) << "For cur node, cannot find valid comm node, cur node is " << node->DebugString();
      continue;
    }
    if (!IsAddNodeValid(node, comm_node)) {
      MS_LOG(INFO) << "For cur node, its strategy not equal to comm node, cur node is " << node->DebugString()
                   << " comm node is " << comm_node->DebugString();
      continue;
    }
    (*add_node_map)[node] = comm_node;
  }
}

void HandleNodePullUp(const AnfNodePtr &comm_node, const CNodePtr &add_node) {
  auto graph = comm_node->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  // handle matmul node, connect it to next node of reduce_scatter/allreduce
  auto comm_node_input = comm_node->cast<CNodePtr>()->input(1);
  (void)manager->Replace(comm_node, comm_node_input);
}

void HandleNodeBiasAdd(const AnfNodePtr &comm_node, const CNodePtr &add_node) {
  auto comm_prim = GetCNodePrimitive(comm_node);
  if (!comm_prim->HasAttr(kAttrRankSize)) {
    MS_LOG(ERROR) << "cur prim has not attr " << kAttrRankSize << ", cur node is " << comm_node->DebugString();
    return;
  }
  auto rank_size = GetValue<int64_t>(comm_prim->GetAttr(kAttrRankSize));
  auto bias_node = add_node->input(2);
  const auto bias_dtype = bias_node->abstract()->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(bias_dtype);
  mindspore::tensor::TensorPtr tensor_ptr =
    std::make_shared<mindspore::tensor::Tensor>(rank_size, bias_dtype->element()->GetType());
  auto const_node = NewValueNode(MakeValue(tensor_ptr));
  const_node->set_abstract(bias_node->abstract());
  AnfNodePtrList div_node_inputs = {NewValueNode(prim::kPrimRealDiv), bias_node, const_node};

  auto fg = comm_node->func_graph();
  auto div_node = fg->NewCNode(div_node_inputs);
  div_node->set_abstract(bias_node->abstract()->Clone());
  auto graph = comm_node->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  (void)manager->Replace(bias_node, div_node);
}

void HandleNodePullDown(const AnfNodePtr &comm_node, const CNodePtr &add_node) {
  auto graph = comm_node->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  AnfNodePtrList new_comm_node_inputs = {comm_node->cast<CNodePtr>()->input(0), add_node};
  auto new_comm_node = graph->NewCNode(new_comm_node_inputs);
  new_comm_node->set_abstract(comm_node->abstract());
  auto prim = GetCNodePrimitive(new_comm_node);
  (void)prim->AddAttr(BIAS_ADD_COMM_SWAP, MakeValue(true));
  (void)manager->Replace(add_node, new_comm_node);
}

void HandleAddNode(HashMap<CNodePtr, AnfNodePtr> *add_node_map) {
  for (auto node_pair : (*add_node_map)) {
    auto add_node = node_pair.first;
    auto comm_node = node_pair.second;
    HandleNodePullUp(comm_node, add_node);
    HandleNodeBiasAdd(comm_node, add_node);
    // pull down comm node, change add node user's input to allreduce
    HandleNodePullDown(comm_node, add_node);
  }
}
}  // namespace

void BiasAddCommSwap(const FuncGraphPtr &graph) {
  if (parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kSemiAutoParallel &&
      parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kAutoParallel) {
    MS_LOG(INFO) << "BiasAddCommSwap is only support under [semi_]auto_parallel, skip it.";
    return;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->get_param<bool>(MS_CTX_BIAS_ADD_COMM_SWAP)) {
    return;
  }

  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  HashMap<CNodePtr, AnfNodePtr> add_node_map;
  for (auto &each_graph : manager->func_graphs()) {
    FindAllValidAddNode(each_graph, &add_node_map);
  }
  // pull up add node, pull down allreduce/reduce_scatter node
  HandleAddNode(&add_node_map);
}
}  // namespace parallel
}  // namespace mindspore
