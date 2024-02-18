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
#include "pipeline/jit/pi/graph_capture/graph.h"
#include <set>
#include <algorithm>
#include <memory>
#include <string>
#include <regex>
#include <utility>
#include "pipeline/jit/pi/common.h"

namespace mindspore {
namespace pijit {
Graph::Graph(PyCodeObject *co, PyObject *globals, const GraphJitConfig &conf)
    : ret_val_(nullptr),
      generator_result_(nullptr),
      co_(py::cast<py::object>(reinterpret_cast<PyObject *>(co))),
      f_globals_(py::cast<py::object>(globals)),
      conf_(conf),
      guard_(nullptr),
      prune_branch_count_(0) {
  stop_trace_info_ = {-1, StopTraceReason::kNonStopTrace};
  if (!co) {
    frame_states_[0] = std::make_unique<FrameStates>();  // emplace empty frame
    module_name_ = "";
    return;
  }
  cfg_ = std::make_unique<CFG>(co);
  cfg_->GenerateCFG();

  if (conf_.GetBoolConfig(GraphJitConfig::kPrintBB)) {
    GRAPH_JIT_LOG_F("%s\n\n", cfg_->DumpBBs().c_str());
  }
  if (conf_.GetBoolConfig(GraphJitConfig::kPrintCFG)) {
    cfg_->DumpCFGGraph();
  }

  auto pyname = PyDict_GetItemString(globals, "__name__");
  if (pyname) {
    module_name_ = PyUnicode_AsUTF8(pyname);
  } else {
    module_name_ = "";
    PyErr_Clear();
  }

  if (conf_.GetBoolConfig(GraphJitConfig::kLoopUnrolling)) {
    LoopFinder loop_finder(this);
    loop_finder.FormSimpleLoopInfo();
    if (conf_.GetBoolConfig(GraphJitConfig::kPrintAfterAll) && !loops().empty()) {
      GRAPH_JIT_LOG_F("%s", DumpLoops().c_str());
    }
  }
}

static const std::set<int> support_constant_op = {
  BINARY_SUBSCR, COMPARE_OP, IS_OP,     CONTAINS_OP, LOAD_ATTR,           LIST_TO_TUPLE,
  BUILD_TUPLE,   BUILD_LIST, BUILD_MAP, BUILD_SLICE, BUILD_CONST_KEY_MAP,
};

bool IsConstantFold(int op, const std::vector<ValueNode *> &inputs) {
  if (op == LOAD_CONST) {
    return true;
  }
  auto iter = std::find_if_not(inputs.begin(), inputs.end(), [](ValueNode *i) { return i->is_constant(); });
  if (iter != inputs.end()) {
    return false;
  }
  if (support_constant_op.find(op) != support_constant_op.end()) {
    return true;
  }
  if (Utils::IsBinaryMathOp(op) && Utils::IsGeneralNoSideEffectOp(op)) {
    return true;
  }
  return false;
}

ValueNode *Graph::NewValueNode(AObject *obj_info, int op, int arg, const std::vector<ValueNode *> &inputs) {
  MS_EXCEPTION_IF_CHECK_FAIL(!Utils::IsCallOp(op), "must not be call function opcode");
  bool constant = false;
  if (IsConstantFold(op, inputs) && obj_info && obj_info->GetPyObject().ptr()) {
    // calculate real value...
    constant = true;
  }
  ValueNode *node;
  if (constant && CheckConstPyObject(obj_info->GetPyObject().ptr())) {
    node = this->allocator().NewNode<ValueNode>(obj_info, LOAD_CONST, -1);
  } else {
    node = this->allocator().NewNode<ValueNode>(obj_info, op, arg, inputs);
  }
  node->SetGraph(this);
  node->set_is_constant(constant);
  return node;
}

CallNode *Graph::NewCallNode(int op, int arg, const std::vector<ValueNode *> &inputs) {
  MS_EXCEPTION_IF_CHECK_FAIL(Utils::IsCallOp(op), "must be call function opcode");
  CallNode *node = this->allocator().NewNode<CallNode>(op, arg, inputs);
  node->SetGraph(this);
  return node;
}

/**
 * TODO: FindLoopEnd, FindLoopBegin, reset break bci
 * restore graph status. clean the variable that loop produced
 * restore frame status of break bci that override by loop analyze
 */
bool Graph::IsBreakAtLoop() const {
  int break_bci = this->GetStopTraceBci();
  if (break_bci == -1) {
    return false;
  }
  const auto &instr = this->cfg_->instr_pool();
  // find the last backward edge overlapping this break point
  int res = break_bci;
  for (int i = break_bci; i < static_cast<int>(instr.size()); ++i) {
    MS_EXCEPTION_IF_CHECK_FAIL(i == instr[i]->bci(), "!!!");
    if (instr[i]->extra_jump() != nullptr) {
      res = std::min(instr[i]->extra_jump()->bci(), res);
    }
  }
  return res != break_bci;
}

bool Graph::IsBreakAtLoopAfterUnrolling() const {
  if (!Config().GetBoolConfig(GraphJitConfig::kLoopUnrolling)) {
    return false;
  }
  if (GetStopTraceBci() == -1) {
    return false;
  }
  if (traced_nodes_.empty()) {
    return false;
  }
  if (GetStopTraceBci() > traced_nodes_.back()->bci()) {
    return false;
  }
  bool break_with_loop_unroll = false;
  for (auto node : traced_nodes_) {
    if (node->bci() >= GetStopTraceBci()) {
      break_with_loop_unroll |= node->GetBlock() != nullptr && node->GetBlock()->is_loop_body();
    }
  }
  return break_with_loop_unroll;
}

void Graph::SetFrame(int bci, const FrameStates &f) {
  // just set once, used to restore the first status if has a loop
  auto &ptr = frame_states_[bci];
  if (ptr == nullptr) {
    ptr = std::make_unique<FrameStates>(f);
  }
}

const FrameStates &Graph::GetFrame(int bci) const {
  auto iter = frame_states_.find(bci);
  MS_EXCEPTION_IF_CHECK_FAIL(iter != frame_states_.end(), "can't find frame status");
  return *(iter->second);
}

static bool CheckObjPtr(ValueNode *node) {
  return node->GetVobj() == nullptr || node->GetVobj()->GetPyObject().ptr() == nullptr;
}

TracePtr GetTrace(ValueNode *node, bool strict, bool print, int depth, int max_depth);
static bool PrepareTraceParam(ValueNode *node, TraceVector *tv, int depth, int max_depth, bool *has_unsupported,
                              bool strict, bool print) {
  const std::vector<ValueNode *> &inputs = node->getInputs();
  for (auto it : inputs) {
    auto t = GetTrace(it, strict, print, depth + 1, max_depth);
    if (t == nullptr) {
      return false;
    } else if (t->GetTraceType() == TraceType::Unsupported) {
      *has_unsupported = true;
    }
    tv->push_back(t);
  }
  return !CheckObjPtr(node);
}

static bool CheckDepth(int depth, int max_depth) { return depth < max_depth || max_depth == -1; }

TracePtr GetTrace(ValueNode *node, bool strict, bool print, int depth, int max_depth) {
  if (!CheckDepth(depth, max_depth)) {
    MS_LOG(DEBUG) << "too deep trace for guard";
    return nullptr;
  }
  if (node->GetType() == AbstractNode::Type::Call) {
    Graph *sub_graph = static_cast<CallNode *>(node)->GetSubGraph();
    if (sub_graph && sub_graph->GetRetVal() != nullptr) {
      return GetTrace(sub_graph->GetRetVal(), strict, print, depth, max_depth);
    }
  }

  PyObject *obj = node->GetVobj() ? node->GetVobj()->GetPyObject().ptr() : nullptr;
  int opcode = node->GetOpcode();
  int oparg = node->GetOparg();
  const std::string &name = node->GetName();
  const char *module_name = node->GetGraph() ? node->GetGraph()->GetModuleName() : "";

  TraceVector tv;
  bool has_unsupported = false;
  if (!PrepareTraceParam(node, &tv, depth, max_depth, &has_unsupported, strict, print)) {
    return strict ? nullptr : std::make_shared<UnsupportedTrace>(nullptr, tv, opcode, oparg);
  }
  TracePtr ret = nullptr;
  switch (node->GetType()) {
    case AbstractNode::Type::Value:
      if (!has_unsupported) {
        ret = CreateOpTrace(obj, opcode, oparg, tv, module_name, name, strict, print);
      }
      break;
    case AbstractNode::Type::Call:
      if (!has_unsupported) {
        const std::string &func_name = node->input(0)->GetName();
        ret = CreateOpTrace(obj, opcode, oparg, tv, module_name, func_name, strict, print);
      }
      break;
    case AbstractNode::Type::Param:
      return std::make_shared<RootTrace>(obj, mindspore::pijit::TraceType::Param, oparg, name);
    case AbstractNode::Type::CellVar: /* fall-through */
    case AbstractNode::Type::FreeVar:
      return std::make_shared<RootTrace>(obj, mindspore::pijit::TraceType::Param, oparg, name);
    case AbstractNode::Type::Unbound:
      break;
    default:
      break;
  }
  if (ret == nullptr && !strict) {
    ret = std::make_shared<UnsupportedTrace>(obj, tv, opcode, oparg);
  }
  return ret;
}

bool Graph::GuardValueNode(ValueNode *node) {
  AObject *vo = node->GetVobj();
  if (guard_ == nullptr || !vo || vo->GetPyObject().ptr() == nullptr) {
    return false;
  }
  if (node->GetOpcode() == LOAD_CONST) {
    return true;
  }
  TracePtr t = GetTrace(node, Config().GetBoolConfig(GraphJitConfig::kStrictTrace),
                        Config().GetBoolConfig(GraphJitConfig::kPrintGuard), 0,
                        Config().getIntConfig(GraphJitConfig::GraphJitConfig::kMaxTraceDepth));
  if (t != nullptr) {
    bool ret = guard_->GetGuard()->GuardOn(t, mindspore::pijit::GuardLevel::GEqual);
    if (Config().GetBoolConfig(GraphJitConfig::kGuardDetachObject)) {
      t->Detach();
    }
    return ret;
  }
  return false;
}

TracePtr Graph::TraceValueNode(ValueNode *node, int max_trace_depth) {
  AObject *vo = node->GetVobj();
  if (guard_ == nullptr || !vo || vo->GetPyObject().ptr() == nullptr) {
    return nullptr;
  }
  if (max_trace_depth < 0) {
    max_trace_depth = Config().getIntConfig(GraphJitConfig::GraphJitConfig::kMaxTraceDepth);
  }
  return GetTrace(node, Config().GetBoolConfig(GraphJitConfig::kStrictTrace),
                  Config().GetBoolConfig(GraphJitConfig::kPrintGuard), 0, max_trace_depth);
}

static std::string TraceInferFailed(ValueNode *node) {
  std::stringstream s;
  s << node << " ";
  switch (node->GetType()) {
    case AbstractNode::Call:
    case AbstractNode::Value: {
      s << "bci " << node->bci() << " " << Utils::GetOpName(node->GetOpcode()) << " " << node->GetOparg();
      if (Utils::IsNameRelated(node->GetOpcode())) {
        s << " " << node->GetName();
      }
      break;
    }
    case AbstractNode::Param: {
      s << "Parameter " << node->GetOparg();
      break;
    }
    case AbstractNode::CellVar:
    case AbstractNode::FreeVar: {
      s << "Closure " << node->GetOparg();
      break;
    }
    case AbstractNode::Unbound: {
      s << "(UnboundLocal)";
      break;
    }
    default: {
      break;
    }
  }
  s << " object is ";
  PyObject *op = node->GetVobj() ? node->GetVobj()->GetPyObject().ptr() : nullptr;
  if (op != nullptr) {
    s << AObject::ToString(op);
    return s.str();
  }
  s << "<NULL>:\n";
  for (size_t i = 0; i < node->getInputs().size(); ++i) {
    s << " " << std::regex_replace(TraceInferFailed(node->input(i)), std::regex("\n"), "\n  ") << "\n";
  }
  s.seekp(-1, s.cur);
  s << " ";
  return s.str();
}

std::string Graph::ToString(int depth) const {
  std::stringstream s;
  std::string prefix(depth << 1, ' ');
  std::string code_name = co_.ptr() != nullptr ? std::string(py::str(co_.ptr())) : "<no code>";

  s << prefix << "*** Trace Nodes [" << code_name << "] ***\n";
  if (depth == 0) {
    s << prefix << "Frame:\n" << GetFrame(0).ToString();
  }

  s << prefix << "Nodes:\n";
  for (auto i : GetTracedNodes()) {
    s << prefix << i->ToString() << "\n";
    if (i->GetType() != AbstractNode::Call) {
      continue;
    }
    CallNode *node = static_cast<CallNode *>(i);
    s << prefix << "{ inline stat " << GetInlineReasonDesc(node->GetInlineReason()) << "\n";
    if (node->GetSubGraph() != nullptr) {
      s << node->GetSubGraph()->ToString(depth + 1);
    }
    s << prefix << "}\n";
  }
  s << prefix << "return: " << (GetRetVal() ? GetRetVal()->ToString() : "trace break") << "\n";
  s << prefix << GetStopTraceReasonDesc(GetStopTraceReason()) << "\n";
  s << prefix << "break bci: " << GetStopTraceBci() << "\n\n";

  if (depth == 0) {
    std::string break_info;
    if (GetRetVal()) {
      PyObject *op = GetRetVal()->GetVobj() ? GetRetVal()->GetVobj()->GetPyObject().ptr() : nullptr;
      if (op == nullptr) {
        break_info = TraceInferFailed(GetRetVal());
      }
    } else {
      break_info = this->DumpBreakInfo();
    }
    if (!break_info.empty()) {
      s << prefix << std::regex_replace(break_info, std::regex("\n"), "\n" + prefix) << "\n";
    }
  }
  return s.str();
}

std::string Graph::DumpBreakInfo() const {
  if (GetStopTraceBci() == -1) {
    return std::string();
  }
  std::stringstream s;
  const auto &f = GetFrame(GetStopTraceBci());
  const auto &nodes = GetTracedNodes();
  const auto &instrs = cfg_->instr_pool();
  int break_bci = GetStopTraceBci();

  s << "graph break at: " << break_bci << ":\n";
  std::vector<ValueNode *> parameters;
  if (nodes.size() == 0 || nodes.back()->bci() < break_bci) {
    // break at unsupported bytecode
    int op = instrs[break_bci]->op();
    int arg = instrs[break_bci]->arg();
    s << Utils::GetOpName(op) << " " << arg << " is not support.\n";
    switch (op) {
      case POP_JUMP_IF_FALSE:
      case POP_JUMP_IF_TRUE:
      case JUMP_IF_FALSE_OR_POP:
      case JUMP_IF_TRUE_OR_POP:
      case FOR_ITER:
      case UNPACK_SEQUENCE:
      case UNPACK_EX: {
        parameters.push_back(f.Peek(0));
        break;
      }
      case CALL_FUNCTION_EX: {
        arg = (arg & 0x01);
      }
      case CALL_FUNCTION_KW: {
        arg++;
      }
      case CALL_METHOD:
      case CALL_FUNCTION: {
        for (int i = arg; i >= 0; --i) {
          parameters.push_back(f.Peek(i));
          AObject *v = f.Peek(i)->GetVobj();
          // just print the first infer failed value
          if (v == nullptr || v->GetPyObject().ptr() == nullptr) {
            parameters = {f.Peek(i)};
            break;
          }
        }
        break;
      }
      default: {
        return s.str();
      }
    }
  } else {
    auto iter = std::find_if(nodes.begin(), nodes.end(), [break_bci](ValueNode *n) { return n->bci() == break_bci; });
    MS_EXCEPTION_IF_CHECK_FAIL(iter != nodes.end(), "can't find break info.");
    parameters.push_back(*iter);
  }
  // print traced value
  for (auto node : parameters) {
    s << TraceInferFailed(node) << "\n";
  }
  return s.str();
}

std::string FrameStates::ToString() const {
  std::stringstream s;
  s << "locals:\n";
  for (size_t i = 0; i < locals.size(); ++i) {
    if (locals[i] != &ValueNode::UnboundLocal) {
      s << i << ": " << locals[i]->ToString() << "\n";
    }
  }
  s << "\nstacks:\n";
  std::for_each(stack.rbegin(), stack.rend(), [&s](ValueNode *i) { s << i->ToString() << "\n"; });
  s << "\ncell free:\n";
  std::for_each(cell_free.begin(), cell_free.end(), [&s](ValueNode *i) { s << i->ToString() << "\n"; });
  s << "\n";
  return s.str();
}

std::string Graph::DumpLoops() const {
  std::ostringstream os;
  if (loops_.empty()) {
    return os.str();
  }
  os << "*** Dump Loops on [" << py::str(co_.ptr()).cast<std::string>() << "] ***\n";
  for (const auto *lp : loops_) {
    os << lp->Dump();
  }
  os << '\n';
  return os.str();
}

}  // namespace pijit
}  // namespace mindspore