/**
 * Copyright 2023-2023 Huawei Technologies Co., Ltd
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
#include "backend/common/pass/merge_transdata.h"

#include <memory>
#include <map>
#include <utility>

#include "ir/graph_utils.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/parallel_context.h"
#include "backend/common/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kOne = 1;
}

bool MergeTransData::Run(const FuncGraphPtr &func_graph) {
  // Only pipeline parallel need run this pass
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto stages = parallel_context->pipeline_stage_split_num();
  if (stages <= 1) {
    return false;
  }

  std::map<std::pair<AnfNodePtr, std::string>, std::vector<CNodePtr>> transdata_map;
  MS_EXCEPTION_IF_NULL(func_graph);
  const std::vector<AnfNodePtr> &node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (IsOneOfPrimitiveCNode(node, {prim::kPrimTransData})) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto out_format = AnfAlgo::GetOutputFormat(cnode, 0);
      auto prenode = common::AnfAlgo::VisitKernelWithReturnType(cnode->input(kOne), 0, true);
      transdata_map[{prenode.first, out_format}].push_back(cnode);
    }
  }

  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  for (auto &kv : transdata_map) {
    if (kv.second.size() <= kOne) {
      continue;
    }
    for (size_t i = kOne; i < kv.second.size(); i++) {
      (void)manager->Replace(kv.second[i], kv.second[0]);
    }
  }

  return true;
}
}  // namespace opt
}  // namespace mindspore
