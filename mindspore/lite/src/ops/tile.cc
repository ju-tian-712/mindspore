/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/tile.h"
#include <algorithm>

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> Tile::GetMultiples() const { return this->primitive->value.AsTile()->multiples; }

void Tile::SetMultiples(const std::vector<int> &multiples) { this->primitive->value.AsTile()->multiples = multiples; }

std::vector<int> Tile::GetDims() const { return this->primitive->value.AsTile()->multiples; }

void Tile::SetDims(const std::vector<int> &dims) { this->primitive->value.AsTile()->dims = dims; }

#else

std::vector<int> Tile::GetMultiples() const {
  auto fb_vector = this->primitive->value_as_Tile()->multiples();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}

void Tile::SetMultiples(const std::vector<int> &multiples) {}

std::vector<int> Tile::GetDims() const {
  auto fb_vector = this->primitive->value_as_Tile()->dims();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}

void Tile::SetDims(const std::vector<int> &dims) {}
#endif

int Tile::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  output->set_data_type(input->data_type());
  output->SetFormat(input->GetFormat());
  if (!GetInferFlag()) {
    return RET_OK;
  }

  MS_ASSERT(tile_prim != nullptr);
  std::vector<int> out_shape;
  std::vector<int> multiples;
  for (int i = 0; i < GetMultiples().size(); ++i) {
    multiples.push_back(GetMultiples()[i]);
  }
//  std::copy(GetMultiples().begin(), GetMultiples().end(), std::back_inserter(multiples));
  for (size_t i = 0; i < input->shape().size(); ++i) {
    int tmp = input->shape()[i] * multiples[i];
    out_shape.push_back(tmp);
  }

  output->set_shape(out_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
