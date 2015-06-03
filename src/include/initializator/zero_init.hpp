#pragma once

#include "constant_init.hpp"

namespace znn { namespace v4 {

class zero_init: public constant_init
{
public:
    zero_init(): constant_init(0) {}

}; // class zero_init

}} // namespace znn::v4
