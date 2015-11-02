#include "cost_fn/malis.hpp"

using namespace znn::v4;

class CMalis_test
{
private:
    cube_p<real> image;
    cube_p<real> label;

    vec3i size_;

public:
    CMalis_test(cube_p in_image,
                cube_p in_lable)
        : image(in_image)
        , label(in_label)
    {
        size_ = in_image;
    }
}
