/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/auto_parallel/rec_core/rec_partition.h"

#include <algorithm>
#include <memory>
#include <functional>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include "frontend/parallel/status.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "include/common/utils/parallel_context.h"

namespace mindspore {
namespace parallel {

double GetMatMulMaxCostIn(const Graph::NodeType &op) {
  auto cost_ptr = std::make_shared<CostMatMul>();
  return cost_ptr->GetMaxCostIn(op.apply);
}

double GetBatchMatMulMaxCostIn(const Graph::NodeType &op) {
  auto cost_ptr = std::make_shared<CostBatchMatMul>();
  return cost_ptr->GetMaxCostIn(op);
}

double GetConvolutionMinCostIn(const Graph::NodeType &op) {
  auto cost_ptr = std::make_shared<CostConvolution>();
  return cost_ptr->GetMinCostIn(op);
}

template <class T>
double GetMaxCostIn(const Graph::NodeType &) {
  auto cost_ptr = std::make_shared<T>();
  return cost_ptr->GetMaxCostIn();
}

template <class T>
double GetMinCostIn(const Graph::NodeType &) {
  auto cost_ptr = std::make_shared<T>();
  return cost_ptr->GetMinCostIn();
}

double GetZeroCostIn(const Graph::NodeType &) { return 0.0; }

using CostFunc = std::function<double(const Graph::NodeType &)>;

// Get the target node's weight for sorting.
double GetWeights(const Graph::NodeType &node) {
  const std::map<OperatorType, CostFunc> cost_func_map = {
    {kRecMatMul, GetMatMulMaxCostIn},
    {kRecBatchMatMul, GetBatchMatMulMaxCostIn},
    {kRecConvolution, GetConvolutionMinCostIn},
    {kRecPooling, GetMinCostIn<CostPooling>},
    {kRecElmWiseOp, GetMinCostIn<CostTensorAdd>},
    {kRecReLU, GetMinCostIn<CostCommon>},
    {kRecReshape, GetMinCostIn<CostReshape>},
    {kRecBiasAdd, GetMinCostIn<CostBiasAdd>},
    {kRecLog, GetMinCostIn<CostCommon>},
    {kRecExp, GetMinCostIn<CostCommon>},
    {kRecAdd, GetMinCostIn<CostCommon>},
    {kRecSub, GetMinCostIn<CostCommon>},
    {kRecMul, GetMinCostIn<CostCommon>},
    {kRecDiv, GetMinCostIn<CostCommon>},
    {kRecSqueeze, GetMinCostIn<CostCommon>},
    {kRecCast, GetMinCostIn<CostCommon>},
    {kRecBatchNorm, GetMaxCostIn<CostBatchParallel>},
    {kRecOneHot, GetMaxCostIn<CostBatchParallel>},
    {kRecPReLU, GetMaxCostIn<CostBatchParallel>},
    {kRecUnsortedSegmentOp, GetMaxCostIn<CostBatchParallel>},
    {kRecSoftmax, GetMaxCostIn<CostBatchParallel>},
    {kRecBatchParallel, GetMaxCostIn<CostBatchParallel>},
    {kRecSparseSoftmaxCrossEntropyWithLogits, GetMaxCostIn<CostBatchParallel>},
    {kRecSoftmaxCrossEntropyWithLogits, GetMaxCostIn<CostBatchParallel>},
    {kRecUnknownType, GetZeroCostIn},
    {kRecVirtual, GetZeroCostIn},
    {kRecStandAlone, GetZeroCostIn}};
  const OperatorRec &op = node.apply;
  const auto &cost_func_iter = cost_func_map.find(op.op_type);
  if (cost_func_iter == cost_func_map.end()) {
    MS_LOG(EXCEPTION) << "Failure: GetOperatorWeight failed.";
  }
  return cost_func_iter->second(node);
}

// Sort all the nodes by their weights
std::vector<size_t> SortByWeight(const std::shared_ptr<Graph> &graph) {
  MS_EXCEPTION_IF_NULL(graph);

  std::vector<std::pair<double, size_t>> weight_to_node_index;
  std::vector<size_t> node_index_by_weights;

  // Get node's weight.
  for (size_t pos = 0; pos < graph->nodes.size(); pos++) {
    if (graph->nodes[pos].info == kApplication) {
      const Graph::NodeType &node_ptr = graph->nodes[pos];
      double weight;
      if (PARTITION_ORDER == PartitionOrder::TopologyOrder) {
        weight = (node_ptr.apply.op_type == OperatorType::kRecUnknownType) ? DOUBLE_LOWEST : pos;
      } else {
        weight = GetWeights(node_ptr);
      }
      size_t index = pos;
      weight_to_node_index.push_back(std::make_pair(weight, index));
    }
  }

  // Ordering ops aka nodes of the graph
  std::sort(weight_to_node_index.begin(), weight_to_node_index.end());

  // Store the result in node_index_by_weights.
  uint64_t size = weight_to_node_index.size();
  for (uint64_t i = 1; i <= size; i++) {
    node_index_by_weights.push_back(weight_to_node_index[size - i].second);
  }

  return node_index_by_weights;
}

using StrategyRecFunc = std::function<StrategyRec(
  Graph::NodeType node, const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
  const std::shared_ptr<Graph> &graph, const bool isTraining)>;

StrategyRec MatMulStrategyRec(Graph::NodeType node,
                              const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                              const std::shared_ptr<Graph> &graph, const bool isTraining) {
  if (graph->dyn_shape_tmp_fix) {
    if (node.param_name.find(".projection.weight") != std::string::npos) {
      node.apply.str.inputTensor[0].str_w /= 2.0;
      node.apply.str.inputTensor[1].str_h /= 2.0;
      return node.apply.str;
    }
    if (node.param_name.find(".mapping.weight") != std::string::npos) {
      node.apply.str.inputTensor[1].str_w /= 2.0;
      node.apply.str.outputTensor.str_w /= 2.0;
      return node.apply.str;
    }
    if (node.param_name.find(".attention.dense2.weight") != std::string::npos) {
      node.apply.str.inputTensor[1].str_w /= 2.0;
      node.apply.str.outputTensor.str_w /= 2.0;
      return node.apply.str;
    }
  }
  // For MatMul
  auto cost_ptr = std::make_shared<CostMatMul>();
  return cost_ptr->GetOptimalStr(node, node_name_to_strategy, *graph, isTraining);
}

StrategyRec BatchMatMulStrategyRec(Graph::NodeType node,
                                   const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                                   const std::shared_ptr<Graph> &graph, const bool isTraining) {
  if (graph->dyn_shape_tmp_fix) {
    if (node.param_name.find(".projection.weight") != std::string::npos) {
      node.apply.str.inputTensor[0].str_w /= 2.0;
      node.apply.str.inputTensor[1].str_h /= 2.0;
      return node.apply.str;
    }
    if (node.param_name.find(".mapping.weight") != std::string::npos) {
      node.apply.str.inputTensor[1].str_w /= 2.0;
      node.apply.str.outputTensor.str_w /= 2.0;
      return node.apply.str;
    }

    bool same_inputs = false;
    bool projection_bias_bmm = false;
    bool mapping_bias_bmm = false;
    for (size_t idx = 0; idx < node.node_in.size(); idx++) {
      if (idx == node.node_in.size() - 1) {
        break;
      }
      for (size_t idx_bis = idx + 1; idx_bis < node.node_in.size(); idx_bis++) {
        if (node.node_in[idx] == node.node_in[idx_bis]) {
          same_inputs = true;
          break;
        }
      }
      if (same_inputs) {
        break;
      }
    }
    if (same_inputs) {
      return node.apply.str;
    }

    for (size_t idx = 0; idx < node.node_in.size(); idx++) {
      auto incoming_node_idx = node.node_in[idx];
      if (graph->nodes[incoming_node_idx].param_name.find(".projection.bias") != std::string::npos) {
        projection_bias_bmm = true;
        break;
      }
      if (graph->nodes[incoming_node_idx].param_name.find(".mapping.bias") != std::string::npos) {
        mapping_bias_bmm = true;
        break;
      }
    }
    if (projection_bias_bmm) {
      node.apply.str.inputTensor[0].str_w /= 2.0;
      node.apply.str.inputTensor[1].str_h /= 2.0;
      return node.apply.str;
    }
    if (mapping_bias_bmm) {
      node.apply.str.inputTensor[1].str_w /= 2.0;
      node.apply.str.outputTensor.str_w /= 2.0;
      return node.apply.str;
    }
  }
  auto cost_ptr = std::make_shared<CostBatchMatMul>();
  return cost_ptr->GetOptimalStr(node, node_name_to_strategy, *graph, isTraining);
}

StrategyRec ConvolutionStrategyRec(Graph::NodeType node,
                                   const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                                   const std::shared_ptr<Graph> &graph, const bool isTraining) {
  const bool enable_conv_chw_partition = false;
  auto cost_ptr = std::make_shared<CostConvolution>();
  return cost_ptr->GetOptimalStr(node, node_name_to_strategy, *graph, enable_conv_chw_partition);
}

StrategyRec PoolingStrategyRec(Graph::NodeType node,
                               const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                               const std::shared_ptr<Graph> &graph, const bool) {
  auto cost_ptr = std::make_shared<CostPooling>();
  return cost_ptr->GetOptimalStr(node, node_name_to_strategy, *graph);
}

StrategyRec ElmWiseStrategyRec(Graph::NodeType node,
                               const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                               const std::shared_ptr<Graph> &graph, const bool) {
  auto cost_ptr = std::make_shared<CostTensorAdd>();
  return cost_ptr->GetOptimalStr(node, node_name_to_strategy, *graph);
}

StrategyRec CommonStrategyRec(Graph::NodeType node,
                              const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                              const std::shared_ptr<Graph> &graph, const bool) {
  auto cost_ptr = std::make_shared<CostCommon>();
  return cost_ptr->GetOptimalStr(node, node_name_to_strategy, *graph);
}

StrategyRec ReshapeStrategyRec(Graph::NodeType node, const std::vector<std::pair<std::string, StrategyRec>> &,
                               const std::shared_ptr<Graph> &, const bool) {
  auto cost_ptr = std::make_shared<CostReshape>();
  return cost_ptr->GetOptimalStr(node);
}

StrategyRec BiasAddStrategyRec(Graph::NodeType node,
                               const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                               const std::shared_ptr<Graph> &graph, const bool) {
  auto cost_ptr = std::make_shared<CostBiasAdd>();
  return cost_ptr->GetOptimalStr(node, node_name_to_strategy, *graph);
}

StrategyRec BatchParallelStrategyRec(Graph::NodeType node, const std::vector<std::pair<std::string, StrategyRec>> &,
                                     const std::shared_ptr<Graph> &, const bool) {
  auto cost_ptr = std::make_shared<CostBatchParallel>();
  return cost_ptr->GetOptimalStr(node);
}

StrategyRec SoftmaxCrossEntropyWithLogitsStrategyRec(Graph::NodeType node,
                                                     const std::vector<std::pair<std::string, StrategyRec>> &,
                                                     const std::shared_ptr<Graph> &, const bool) {
  auto cost_ptr = std::make_shared<CostSoftmaxCrossEntropyWithLogits>();
  return cost_ptr->GetOptimalStr(node);
}

StrategyRec UnknownTypeStrategyRec(Graph::NodeType, const std::vector<std::pair<std::string, StrategyRec>> &,
                                   const std::shared_ptr<Graph> &, const bool) {
  StrategyRec default_strategy;
  return default_strategy;
}

StrategyRec StandAloneStrategyRec(Graph::NodeType, const std::vector<std::pair<std::string, StrategyRec>> &,
                                  const std::shared_ptr<Graph> &, const bool) {
  StrategyRec default_strategy;
  return default_strategy;
}

// Get optimal strategy to partition the target node
StrategyRec PartitionNode(Graph::NodeType node,
                          const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                          const std::shared_ptr<Graph> &graph, const bool isTraining) {
  MS_EXCEPTION_IF_NULL(graph);
  const std::map<OperatorType, StrategyRecFunc> strategy_rec_func_map = {
    {OperatorType::kRecMatMul, MatMulStrategyRec},
    {OperatorType::kRecBatchMatMul, BatchMatMulStrategyRec},
    {OperatorType::kRecConvolution, ConvolutionStrategyRec},
    {OperatorType::kRecPooling, PoolingStrategyRec},
    {OperatorType::kRecElmWiseOp, ElmWiseStrategyRec},
    {OperatorType::kRecReLU, CommonStrategyRec},
    {OperatorType::kRecReshape, ReshapeStrategyRec},
    {OperatorType::kRecBiasAdd, BiasAddStrategyRec},
    {OperatorType::kRecLog, CommonStrategyRec},
    {OperatorType::kRecExp, CommonStrategyRec},
    {OperatorType::kRecAdd, CommonStrategyRec},
    {OperatorType::kRecSub, CommonStrategyRec},
    {OperatorType::kRecMul, CommonStrategyRec},
    {OperatorType::kRecDiv, CommonStrategyRec},
    {OperatorType::kRecSqueeze, CommonStrategyRec},
    {OperatorType::kRecCast, CommonStrategyRec},
    {OperatorType::kRecBatchNorm, BatchParallelStrategyRec},
    {OperatorType::kRecOneHot, BatchParallelStrategyRec},
    {OperatorType::kRecPReLU, BatchParallelStrategyRec},
    {OperatorType::kRecSoftmax, BatchParallelStrategyRec},
    {OperatorType::kRecSparseSoftmaxCrossEntropyWithLogits, BatchParallelStrategyRec},
    {OperatorType::kRecUnsortedSegmentOp, BatchParallelStrategyRec},
    {OperatorType::kRecBatchParallel, BatchParallelStrategyRec},
    {OperatorType::kRecVirtual, BatchParallelStrategyRec},
    {OperatorType::kRecSoftmaxCrossEntropyWithLogits, SoftmaxCrossEntropyWithLogitsStrategyRec},
    {OperatorType::kRecUnknownType, UnknownTypeStrategyRec},
    {OperatorType::kRecStandAlone, StandAloneStrategyRec}};
  const auto &strategy_rec_func_iter = strategy_rec_func_map.find(node.apply.op_type);
  if (strategy_rec_func_iter == strategy_rec_func_map.end()) {
    MS_LOG(EXCEPTION) << "Failure: Partition Operator failed.";
  }
  return strategy_rec_func_iter->second(node, node_name_to_strategy, graph, isTraining);
}

StrategyRec GetOneLoopStrategy(size_t op_inputs_num, const StrategyRec &old_str, StrategyRec new_str) {
  for (size_t i = 0; i < op_inputs_num; i++) {
    if (abs(old_str.inputTensor[i].str_n) > EPS && abs(old_str.inputTensor[i].str_c) > EPS &&
        abs(old_str.inputTensor[i].str_h) > EPS && abs(old_str.inputTensor[i].str_w) > EPS) {
      new_str.inputTensor[i].str_n = new_str.inputTensor[i].str_n / old_str.inputTensor[i].str_n;
      new_str.inputTensor[i].str_c = new_str.inputTensor[i].str_c / old_str.inputTensor[i].str_c;
      new_str.inputTensor[i].str_h = new_str.inputTensor[i].str_h / old_str.inputTensor[i].str_h;
      new_str.inputTensor[i].str_w = new_str.inputTensor[i].str_w / old_str.inputTensor[i].str_w;
    }
  }

  if (old_str.outputTensor.str_n > EPS && old_str.outputTensor.str_c > EPS && old_str.outputTensor.str_h > EPS &&
      old_str.outputTensor.str_w > EPS) {
    new_str.outputTensor.str_n = new_str.outputTensor.str_n / old_str.outputTensor.str_n;
    new_str.outputTensor.str_c = new_str.outputTensor.str_c / old_str.outputTensor.str_c;
    new_str.outputTensor.str_h = new_str.outputTensor.str_h / old_str.outputTensor.str_h;
    new_str.outputTensor.str_w = new_str.outputTensor.str_w / old_str.outputTensor.str_w;
  }

  return new_str;
}

Graph::NodeType ChangeStrategy(Graph::NodeType Node, size_t n_cut) {
  if (n_cut >= Node.apply.strs.size()) {
    MS_LOG(EXCEPTION) << "Strategy not available";
  }
  Node.apply.str = Node.apply.strs[n_cut];
  Node = ApplyStrToTensor(Node);

  return Node;
}

size_t GetStratNumber(const Graph::NodeType &Node) { return Node.apply.strs.size(); }

void PartitionPipelineStages(double device_memory, const std::shared_ptr<Graph> &graph) {
  if (!ENABLE_PIPE_ALGO) {
    size_t n_stage = LongToSize(parallel::ParallelContext::GetInstance()->pipeline_stage_split_num());
    size_t n_node = graph->nodes.size();
    size_t roll_back = FloatToSize(log2(n_stage));

    MS_LOG(INFO) << "ROLLING BACK ACCORDING TO STAGE NUMBER (" << n_stage << ") BY " << roll_back << " LEVELS"
                 << std::endl;
    for (size_t i_node = 0; i_node < n_node; ++i_node) {
      Graph::NodeType &node_ptr = graph->nodes[i_node];
      size_t n_cut = GetStratNumber(graph->nodes[i_node]) - roll_back - 1;
      graph->nodes[i_node] = ChangeStrategy(node_ptr, n_cut);
    }
  }
}

// Partition graph into all devices.
Status PartitionForAllDevices(size_t num_device, double device_memory, const std::shared_ptr<Graph> &graph,
                              bool isTraining) {
  if (num_device < 1) {
    MS_LOG(EXCEPTION) << "ERROR: Number of devices can't be " << num_device << ".";
  }

  if (num_device > 1024) {
    MS_LOG(EXCEPTION) << "ERROR: Number of devices can't be larger than 1024.";
  }

  MS_EXCEPTION_IF_NULL(graph);

  // Comopute iter times
  int64_t iter_times = static_cast<int64_t>(log2(num_device));
  if (iter_times > 10) {
    MS_LOG(EXCEPTION) << "ERROR: Number of iter_times can't be larger than 10.";
  }

  // N-cuts loop
  for (int64_t loop = 0; loop < iter_times; loop++) {
    // Sort by weights
    std::vector<size_t> reorder_node_list = SortByWeight(graph);

    // get total node number
    size_t iter_nodes = reorder_node_list.size();

    // temp vector to map nodename to its strategy.
    std::vector<std::pair<std::string, StrategyRec>> node_name_to_strategy;

    // Loop for all the nodes
    for (size_t i_node = 0; i_node < iter_nodes; i_node++) {
      // get current node's index
      size_t index = reorder_node_list[i_node];

      Graph::NodeType &node_ptr = graph->nodes[index];

      // 2-parts partitioning StrategyRec of the last loop
      StrategyRec old_str = graph->nodes[index].apply.str;

      // Save first strategy too
      if (graph->nodes[index].apply.strs.size() == 0) {
        graph->nodes[index].apply.strs.push_back(old_str);
      }

      MS_LOG(INFO) << "------------Node_name: " << graph->nodes[index].name << " -------------";

      // Search optimal strategy to cut this operator. And store the result optimal strategy in graph.
      graph->nodes[index].apply.str = PartitionNode(node_ptr, node_name_to_strategy, graph, isTraining);
      graph->nodes[index].apply.strs.push_back(graph->nodes[index].apply.str);

      // Get Current 2-parts partitioning strategy of this loop
      size_t op_inputs_num = graph->nodes[index].node_in.size();
      StrategyRec one_loop_strategyrec = GetOneLoopStrategy(op_inputs_num, old_str, graph->nodes[index].apply.str);

      // Apply OP Strategy to Tensor Strategy.
      graph->nodes[index] = ApplyStrToTensor(node_ptr);

      // Note down the node name and its strategy in this loop.
      auto node_name_to_str = std::pair<std::string, StrategyRec>(graph->nodes[index].name, one_loop_strategyrec);
      node_name_to_strategy.push_back(node_name_to_str);
    }
  }

  // Partition stages
  if (parallel::ParallelContext::GetInstance()->pipeline_stage_split_num() > 1) {
    PartitionPipelineStages(device_memory, graph);
  }

  DevicesMemoryControl(num_device, device_memory, graph);
  return SUCCESS;
}

// Apply OP Strategy to Tensor Strategy
Graph::NodeType ApplyStrToTensor(Graph::NodeType Node) {
  // Set Node's tensor_parm
  Node.tensor_parm.tensor_str.str_n = Node.apply.str.outputTensor.str_n;
  Node.tensor_parm.tensor_str.str_c = Node.apply.str.outputTensor.str_c;
  Node.tensor_parm.tensor_str.str_h = Node.apply.str.outputTensor.str_h;
  Node.tensor_parm.tensor_str.str_w = Node.apply.str.outputTensor.str_w;

  // Set input tensors' tersor_parm
  for (int64_t i = 0; i < 2; i++) {
    Node.apply.arguments[i].tensor_str.str_n = Node.apply.str.inputTensor[i].str_n;
    Node.apply.arguments[i].tensor_str.str_c = Node.apply.str.inputTensor[i].str_c;
    Node.apply.arguments[i].tensor_str.str_h = Node.apply.str.inputTensor[i].str_h;
    Node.apply.arguments[i].tensor_str.str_w = Node.apply.str.inputTensor[i].str_w;
  }
  return Node;
}

void DevicesMemoryControl(const size_t num_device, const double device_memory, const std::shared_ptr<Graph> &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  if (num_device == 0) {
    MS_LOG(EXCEPTION) << "Failure: device number is 0.";
  }

  uint64_t iter_nodes = graph->nodes.size();
  double used_memory = 0.0;

  for (uint64_t i_node = 0; i_node < iter_nodes; i_node++) {
    if (graph->nodes[i_node].info == InfoType::kApplication) {
      Graph::NodeType &Node = graph->nodes[i_node];
      for (int64_t index = 0; index < 2; index++) {
        used_memory += Node.apply.arguments[index].tensor_str.str_n * Node.apply.arguments[index].tensor_shape.shape_n *
                       Node.apply.arguments[index].tensor_str.str_c * Node.apply.arguments[index].tensor_shape.shape_c *
                       Node.apply.arguments[index].tensor_str.str_h * Node.apply.arguments[index].tensor_shape.shape_h *
                       Node.apply.arguments[index].tensor_str.str_w * Node.apply.arguments[index].tensor_shape.shape_w *
                       GetDataTypeSize(Node.apply.arguments[index].tensor_type);
      }
    }
  }

  if (device_memory < (used_memory / num_device)) {
    MS_LOG(WARNING) << "It is estimated that the task may collapse due to out of memory!";
  }
}

size_t GetDataTypeSize(const TensorType &type) {
  switch (type) {
    case kInt8:
      return sizeof(int64_t);
    case kFloat16:
      return sizeof(float) / 2;
    case kFloat32:
      return sizeof(float);
    case kDouble64:
      return sizeof(double);
    default:
      MS_LOG(EXCEPTION) << "GetDataTypeSize Failed. Unexpected type";
  }
}
}  // namespace parallel
}  // namespace mindspore
