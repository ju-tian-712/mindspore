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

#ifndef MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_MATH_OPS_H_
#define MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_MATH_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"
#include "transform/graph_ir/custom_op_proto/op_proto_macro.h"

/* clang-format off */

namespace ge {
REG_CUST_OP(CholeskySolve)
  .INPUT(x1, TensorType({DT_DOUBLE, DT_FLOAT}))
  .INPUT(x2, TensorType({DT_DOUBLE, DT_FLOAT}))
  .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT}))
  .REQUIRED_ATTR(upper, Bool)
  .CUST_OP_END_FACTORY_REG(CholeskySolve)

REG_CUST_OP(Cauchy)
  .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
  .REQUIRED_ATTR(size, ListInt)
  .REQUIRED_ATTR(sigma, Float)
  .REQUIRED_ATTR(median, Float)
  .CUST_OP_END_FACTORY_REG(Cauchy)

REG_CUST_OP(CholeskyInverse)
  .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT}))
  .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT}))
  .REQUIRED_ATTR(upper, Bool)
  .CUST_OP_END_FACTORY_REG(CholeskyInverse)

REG_CUST_OP(Eig)
  .INPUT(x, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT}))
  .OUTPUT(eigen_values, TensorType({DT_COMPLEX128, DT_COMPLEX64}))
  .OUTPUT(eigen_vectors, TensorType({DT_COMPLEX128, DT_COMPLEX64}))
  .REQUIRED_ATTR(compute_v, Bool)
  .CUST_OP_END_FACTORY_REG(Eig)

REG_CUST_OP(Eps)
  .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
  .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
  .CUST_OP_END_FACTORY_REG(Eps)

REG_CUST_OP(Hypot)
  .INPUT(x1, TensorType({DT_DOUBLE, DT_FLOAT}))
  .INPUT(x2, TensorType({DT_DOUBLE, DT_FLOAT}))
  .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT}))
  .CUST_OP_END_FACTORY_REG(Hypot)

REG_CUST_OP(MatrixLogarithm)
  .INPUT(x, TensorType({DT_COMPLEX128, DT_COMPLEX64}))
  .OUTPUT(y, TensorType({DT_COMPLEX128, DT_COMPLEX64}))
  .CUST_OP_END_FACTORY_REG(MatrixLogarithm)

REG_CUST_OP(Lcm)
  .INPUT(x1, TensorType({DT_INT32, DT_INT64}))
  .INPUT(x2, TensorType({DT_INT32, DT_INT64}))
  .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
  .CUST_OP_END_FACTORY_REG(Lcm)

REG_CUST_OP(MatrixExp)
  .INPUT(x, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
  .OUTPUT(y, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
  .CUST_OP_END_FACTORY_REG(MatrixExp)

REG_CUST_OP(Heaviside)
  .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT16,
                        DT_UINT32, DT_UINT64, DT_UINT8}))
  .INPUT(values, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT16,
                             DT_UINT32, DT_UINT64, DT_UINT8}))
  .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT16,
                         DT_UINT32, DT_UINT64, DT_UINT8}))
  .CUST_OP_END_FACTORY_REG(Heaviside)

REG_CUST_OP(Gcd)
  .INPUT(x1, TensorType({DT_INT32, DT_INT64}))
  .INPUT(x2, TensorType({DT_INT32, DT_INT64}))
  .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
  .CUST_OP_END_FACTORY_REG(Gcd)

REG_CUST_OP(Orgqr)
  .INPUT(x, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT}))
  .INPUT(tau, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT}))
  .OUTPUT(y, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT}))
  .CUST_OP_END_FACTORY_REG(Orgqr)

REG_CUST_OP(TraceGrad)
  .INPUT(y_grad, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT16,
         DT_UINT32, DT_UINT64, DT_UINT8}))
  .INPUT(x_shape, TensorType({DT_INT64}))
  .OUTPUT(x_grad, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT16,
          DT_UINT32, DT_UINT64, DT_UINT8}))
  .CUST_OP_END_FACTORY_REG(TraceGrad)

REG_CUST_OP(Lgamma)
  .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT32}))
  .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
  .CUST_OP_END_FACTORY_REG(Lgamma)

REG_CUST_OP(Diagonal)
  .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT16,
         DT_UINT32, DT_UINT64, DT_UINT8, DT_BOOL}))
  .REQUIRED_ATTR(offset, Int)
  .REQUIRED_ATTR(dim1, Int)
  .REQUIRED_ATTR(dim2, Int)
  .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT16,
         DT_UINT32, DT_UINT64, DT_UINT8, DT_BOOL}))
  .CUST_OP_END_FACTORY_REG(Diagonal)

REG_CUST_OP(FFT)
  .INPUT(input, TensorType({DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
         DT_COMPLEX64, DT_COMPLEX128}))
  .OPTIONAL_INPUT(n, TensorType({DT_INT64}))
  .INPUT(dim, TensorType({DT_INT64}))
  .OPTIONAL_INPUT(norm, TensorType({DT_INT64}))
  .OUTPUT(y, TensorType({DT_COMPLEX64, DT_COMPLEX128}))
  .CUST_OP_END_FACTORY_REG(FFT)

REG_CUST_OP(IFFT)
  .INPUT(input, TensorType({DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
         DT_COMPLEX64, DT_COMPLEX128}))
  .OPTIONAL_INPUT(n, TensorType({DT_INT64}))
  .INPUT(dim, TensorType({DT_INT64}))
  .OPTIONAL_INPUT(norm, TensorType({DT_INT64}))
  .OUTPUT(y, TensorType({DT_COMPLEX64, DT_COMPLEX128}))
  .CUST_OP_END_FACTORY_REG(IFFT)

REG_CUST_OP(FFTShift)
  .INPUT(input, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16,
         DT_UINT32, DT_UINT64, DT_BOOL, DT_COMPLEX128, DT_COMPLEX64}))
  .OPTIONAL_INPUT(dim, TensorType({DT_INT64}))
  .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16,
         DT_UINT32, DT_UINT64, DT_BOOL, DT_COMPLEX128, DT_COMPLEX64}))
  .CUST_OP_END_FACTORY_REG(FFTShift)

REG_CUST_OP(IFFTShift)
  .INPUT(input, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16,
         DT_UINT32, DT_UINT64, DT_BOOL, DT_COMPLEX128, DT_COMPLEX64}))
  .OPTIONAL_INPUT(dim, TensorType({DT_INT64}))
  .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16,
         DT_UINT32, DT_UINT64, DT_BOOL, DT_COMPLEX128, DT_COMPLEX64}))
  .CUST_OP_END_FACTORY_REG(IFFTShift)

REG_CUST_OP(Correlate)
  .INPUT(a, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_COMPLEX64,
                        DT_COMPLEX128}))
  .INPUT(v, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_COMPLEX64,
                        DT_COMPLEX128}))
  .OUTPUT(output, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
  .ATTR(mode, String, "valid")
  .CUST_OP_END_FACTORY_REG(Correlate)

REG_CUST_OP(DCT)
  .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16,
                        DT_UINT32, DT_UINT64, DT_BOOL, DT_COMPLEX128, DT_COMPLEX64}))
  .REQUIRED_ATTR(type, Int)
  .REQUIRED_ATTR(n, Int)
  .REQUIRED_ATTR(axis, Int)
  .REQUIRED_ATTR(norm, Int)
  .REQUIRED_ATTR(forward, Bool)
  .REQUIRED_ATTR(grad, Bool)
  .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT}))
  .CUST_OP_END_FACTORY_REG(DCT)

REG_CUST_OP(Polar)
  .INPUT(abs, TensorType({DT_FLOAT, DT_DOUBLE}))
  .INPUT(angle, TensorType({DT_FLOAT, DT_DOUBLE}))
  .OUTPUT(y, TensorType({DT_COMPLEX64, DT_COMPLEX128}))
  .CUST_OP_END_FACTORY_REG(Polar)

REG_CUST_OP(Real)
  .INPUT(input, TensorType({DT_COMPLEX64, DT_COMPLEX128}))
  .OUTPUT(output, TensorType({DT_FLOAT, DT_DOUBLE}))
  .CUST_OP_END_FACTORY_REG(Real)

REG_CUST_OP(TriuIndices)
  .OUTPUT(output, TensorType({DT_INT32, DT_INT64}))
  .REQUIRED_ATTR(row, Int)
  .REQUIRED_ATTR(col, Int)
  .REQUIRED_ATTR(offset, Int)
  .REQUIRED_ATTR(dtype, Type)
  .CUST_OP_END_FACTORY_REG(TriuIndices)

REG_CUST_OP(TrilIndices)
  .OUTPUT(output, TensorType({DT_INT32, DT_INT64}))
  .REQUIRED_ATTR(row, Int)
  .REQUIRED_ATTR(col, Int)
  .REQUIRED_ATTR(offset, Int)
  .REQUIRED_ATTR(dtype, Type)
  .CUST_OP_END_FACTORY_REG(TrilIndices)

REG_CUST_OP(Polygamma)
  .INPUT(a, TensorType({DT_INT32, DT_INT64}))
  .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
  .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
  .CUST_OP_END_FACTORY_REG(Polygamma)

REG_CUST_OP(FFTWithSize)
  .INPUT(x, TensorType({DT_BOOL, DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,
                        DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT8}))
  .OUTPUT(y, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT}))
  .REQUIRED_ATTR(signal_ndim, Int)
  .REQUIRED_ATTR(inverse, Bool)
  .REQUIRED_ATTR(signal_sizes, ListInt)
  .REQUIRED_ATTR(norm, String)
  .REQUIRED_ATTR(onesided, Bool)
  .REQUIRED_ATTR(real, Bool)
  .CUST_OP_END_FACTORY_REG(FFTWithSize)
}  // namespace ge
#endif  // MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_MATH_OPS_H_
