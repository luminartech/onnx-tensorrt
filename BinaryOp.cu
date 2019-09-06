/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "BinaryOp.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cassert>

#include <thrust/transform.h>
#include <thrust/system/cuda/execution_policy.h>

#include "CuOpHelpers.hpp"

#define CAPTURE(...) [__VA_ARGS__]
#define BINARY_TRANSFORM(captured_type, func) \
  thrust::transform(thrust::cuda::par.on(stream), \
                    idata0, idata0 + num_elements, idata1, odata, \
                    captured_type __device__ (float x, float y) { return func; })


template <typename Data>
int BinaryOpPlugin::doEnqueue(int batchSize,
                              const void *const *inputs, void **outputs,
                              void *workspace, cudaStream_t stream) {
  size_t num_elements = batchSize * get_num_elements(this->getInputDims(0));

  Data const* idata0 = static_cast<Data const*>(inputs[0]);
  Data const* idata1 = static_cast<Data const*>(inputs[1]);
  Data*       odata  = static_cast<Data*      >(outputs[0]);

  // Note: These local-scope copies are needed for lambda capture
  float alpha = _alpha;
  float beta = _beta;

  switch( _op_type ) {
    case LESS:  BINARY_TRANSFORM(CAPTURE(alpha), ((x<y) ? 0.0f : 1.0f)); break;
    default: return -1;
  }
  return cudaGetLastError() != cudaSuccess;
}

int BinaryOpPlugin::enqueue(int batchSize,
                            const void *const *inputs, void **outputs,
                            void *workspace, cudaStream_t stream) {
  if (getDataType()==nvinfer1::DataType::kFLOAT) {        
    return doEnqueue<float>(batchSize, inputs, outputs, workspace, stream);
  } else {
#if CUDART_VERSION < 9000
    throw std::runtime_error("FP16 plugin is not supported for CUDA < 9.0");
#else    
    return doEnqueue<__half>(batchSize, inputs, outputs, workspace, stream);
#endif  
  }
}

bool BinaryOpPlugin::supportsFormat(nvinfer1::DataType type,
                                    nvinfer1::PluginFormat format) const {
  return (type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kHALF);
}
