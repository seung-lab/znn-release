#pragma once

#if defined(ZNN_USE_MKL_CONVOLUTION)
#  include "conv/mkl.hpp"
#else
#  include "conv/naive.hpp"
#endif
