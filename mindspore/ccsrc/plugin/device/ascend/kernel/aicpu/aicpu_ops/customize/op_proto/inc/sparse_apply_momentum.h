/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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

#ifndef CUSTOMIZE_OP_PROTO_INC_CHOLESKY_SOLVE_OP_H
#define CUSTOMIZE_OP_PROTO_INC_CHOLESKY_SOLVE_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief Update relevant entries in '*var' and '*accum' according to the momentum scheme.
*   Set use_nesterov = True if you want to use Nesterov momentum.
*  computing process:
*  accum = accum * momentum + grad
*  var -= lr * accum
*
* @attention Constraints:
*  the input tensors expect indices must have the same shape.
*
* @par Inputs:
* @li var: A mutable tensor. Should be from a Variable().
* @li accum: A mutable tensor. Has the same type as "var".
*     Should be from a Variable().
* @li lr: A scalar. Has the same type as "var".
* @li grad: A tensor for the gradient. Has the same type as "var".
* @li indices: A vector of indices into the first dimension of "var" and "accum".
* @li momentum: Momentum. Must be a scalar.

* @par Attributes:
* @li use_nesterov: An optional bool. Defaults to "False".
*     If "True", the tensor passed to compute grad will be
*     var - lr * momentum * accum, so in the end, the var you get is actually
*     var - lr * momentum * accum.
*
* @li use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var", "ms", and "mom" tensors is protected by a lock;
*     otherwise the behavior is undefined, but may exhibit less contention.
*
* @par Outputs:
* var: A mutable tensor. Has the same type as input "var".
*
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseApplyMomentum.
*
*/
REG_CUST_OP(SparseApplyMomentum)
  .INPUT(var, TensorType::RealNumberType())
  .INPUT(accum, TensorType::RealNumberType())
  .INPUT(lr, TensorType::RealNumberType())
  .INPUT(grad, TensorType::RealNumberType())
  .INPUT(indices, TensorType::RealIndexNumberType())
  .INPUT(momentum, TensorType::RealNumberType())
  .OUTPUT(var, TensorType::RealNumberType())
  .ATTR(use_locking, Bool, false)
  .ATTR(use_nesterov, Bool, false)
  .CUST_OP_END_FACTORY_REG(SparseApplyMomentum)
}  // namespace ge

#endif  // CUSTOMIZE_OP_PROTO_INC_CHOLESKY_SOLVE_OP_H