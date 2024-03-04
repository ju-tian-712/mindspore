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

#include "prompt_kv_cache_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr int index0 = 0;
constexpr int index1 = 1;
constexpr int index2 = 2;
constexpr int index3 = 3;
constexpr int index4 = 4;
constexpr int index5 = 5;
constexpr int index6 = 6;

constexpr int32_t kSize1 = 1;
constexpr int32_t kSize2 = 2;
constexpr int32_t kSize4 = 4;

constexpr int64_t kAxisOne = 1;
constexpr int64_t kAxisTwo = 2;

constexpr size_t k910bWS = 16 * 1024 * 1024;
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
  auto past_input = context->GetInputDesc(index0);
  auto dtype = past_input->GetDataType();
  auto type_size = ge::GetSizeByDataType(dtype);
  switch (type_size) {
    case kSize1:
      context->SetTilingKey(kSize1);
      break;
    case kSize2:
      context->SetTilingKey(kSize2);
      break;
    case kSize4:
      context->SetTilingKey(kSize4);
      break;
    default:
      return ge::GRAPH_PARAM_INVALID;
  }

  const gert::StorageShape *cache_shape = context->GetInputShape(index0);
  const gert::StorageShape *update_shape = context->GetInputShape(index1);

  bool is_dim4 = true;
  const size_t kDim3 = 3;
  const size_t kDim4 = 4;
  if (cache_shape->GetStorageShape().GetDimNum() == kDim4) {
    is_dim4 = true;
  } else if (cache_shape->GetStorageShape().GetDimNum() == kDim3) {
    is_dim4 = false;
  } else {
    return ge::GRAPH_PARAM_INVALID;
  }

  int64_t b = cache_shape->GetStorageShape().GetDim(index0);
  int64_t ub = update_shape->GetStorageShape().GetDim(index0);
  // s need get when run
  int64_t h = 0;
  int64_t s = 0;
  int64_t us = 0;
  int64_t d = 0;

  if (is_dim4) {
    h = update_shape->GetStorageShape().GetDim(index1);
    s = cache_shape->GetStorageShape().GetDim(index2);
    us = update_shape->GetStorageShape().GetDim(index2);
    d = update_shape->GetStorageShape().GetDim(index3);
  } else {
    h = 1;
    s = cache_shape->GetStorageShape().GetDim(index1);
    us = update_shape->GetStorageShape().GetDim(index1);
    d = update_shape->GetStorageShape().GetDim(index2);
  }

  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  auto aiv_num = platform.GetCoreNumAiv();

  context->SetBlockDim(aiv_num);

  // set workspace for 910B
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = k910bWS;

  TilingData tiling;
  tiling.set_core_num(aiv_num);
  tiling.set_b(b);
  tiling.set_h(h);
  tiling.set_s(s);
  tiling.set_d(d);
  tiling.set_ub(ub);
  tiling.set_us(us);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckSupported(const ge::Operator &op, ge::AscendString &result) {
  std::string resultStr{};
  constexpr size_t input_num = 7;
  if (op.GetInputsSize() != input_num) {
    resultStr = R"({"ret_code": "1", "reason": "input num is not 7"})";
    result = ge::AscendString(resultStr.c_str());
    return ge::GRAPH_FAILED;
  }
  if (op.GetOutputsSize() != 1) {
    resultStr = R"({"ret_code": "1", "reason": "output num is not 1"})";
    result = ge::AscendString(resultStr.c_str());
    return ge::GRAPH_FAILED;
  }

  for (size_t i = 0; i < input_num; ++i) {
    if (op.GetInputDesc(i).GetFormat() != ge::FORMAT_ND) {
      resultStr = R"({"ret_code": "1", "reason": "input format is not supported, only support ND."})";
      result = ge::AscendString(resultStr.c_str());
      return ge::GRAPH_FAILED;
    }
  }

  const int64_t input_dim3_num = 3;
  const int64_t input_dim4_num = 4;
  if (op.GetInputDesc(index0).GetShape().GetDimNum() != input_dim3_num &&
      op.GetInputDesc(index0).GetShape().GetDimNum() != input_dim4_num) {
    resultStr = R"({"ret_code": "1", "reason": "input dim is not supported, cache and update dim must be 4."})";
    result = ge::AscendString(resultStr.c_str());
    return ge::GRAPH_FAILED;
  }

  if (op.GetInputDesc(index1).GetShape().GetDimNum() != input_dim3_num &&
      op.GetInputDesc(index1).GetShape().GetDimNum() != input_dim4_num) {
    resultStr = R"({"ret_code": "1", "reason": "input dim is not supported, cache and update dim must be 4."})";
    result = ge::AscendString(resultStr.c_str());
    return ge::GRAPH_FAILED;
  }

  if (op.GetInputDesc(index2).GetShape().GetDimNum() != 1 || op.GetInputDesc(index3).GetShape().GetDimNum() != 1 ||
      op.GetInputDesc(index4).GetShape().GetDimNum() != 1 || op.GetInputDesc(index5).GetShape().GetDimNum() != 1 ||
      op.GetInputDesc(index6).GetShape().GetDimNum() != 1) {
    resultStr =
      R"({"ret_code": "1", "reason": "input dim is not supported, valid_seq_len, batch_index, seq_len_axis,
      new_max_seq_len and cur_max_seq_len dim must be 1."})";
    result = ge::AscendString(resultStr.c_str());
    return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context) {
  const gert::Shape *x1_shape = context->GetInputShape(0);
  gert::Shape *y_shape = context->GetOutputShape(0);
  *y_shape = *x1_shape;
  return GRAPH_SUCCESS;
}
ge::graphStatus InferPromptKvCacheDataType(gert::InferDataTypeContext *context) {
  const ge::DataType datatype = context->GetInputDataType(0);
  context->SetOutputDataType(0, datatype);
  return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class PromptKvCache : public OpDef {
 public:
  explicit PromptKvCache(const char *name) : OpDef(name) {
    this->Input("cache")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_UINT8, ge::DT_UINT16,
                 ge::DT_BF16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
               ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                           ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("update")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_UINT8, ge::DT_UINT16,
                 ge::DT_BF16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
               ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                           ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("valid_seq_len")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                 ge::DT_INT64})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
               ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                           ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("batch_index")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                 ge::DT_INT64})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
               ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                           ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("seq_len_axis")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                 ge::DT_INT64})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
               ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                           ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("new_max_seq_len")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                 ge::DT_INT64})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
               ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                           ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("cur_max_seq_len")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                 ge::DT_INT64})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
               ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                           ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("out")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_UINT8, ge::DT_UINT16,
                 ge::DT_BF16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
               ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                           ge::FORMAT_ND, ge::FORMAT_ND});

    this->SetInferShape(ge::InferShape);
    this->SetInferDataType(ge::InferPromptKvCacheDataType);

    this->AICore()
      .SetTiling(optiling::TilingFunc)
      .AddConfig("ascend910")
      .AddConfig("ascend910b")
      .AddConfig("ascend310p")
      .SetCheckSupport(optiling::CheckSupported);
  }
};

OP_ADD(PromptKvCache);
}  // namespace ops
