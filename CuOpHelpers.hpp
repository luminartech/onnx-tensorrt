#pragma once

#include <NvInfer.h>

inline size_t get_num_elements(nvinfer1::Dims dims) {
  size_t num_elements = 1;
  for( int d=0; d<dims.nbDims; ++d ) {
    num_elements *= dims.d[d];
  }
  return num_elements;
}
