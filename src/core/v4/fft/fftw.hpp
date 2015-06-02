#pragma once

#ifdef ZNN_USE_MKL_NATIVE_FFT
#  include "fftmkl.hpp"
#else
#  ifdef ZNN_USE_FLOATS
#    include "fftwf.hpp"
#  else
#    include "fftwd.hpp"
#  endif
#endif
