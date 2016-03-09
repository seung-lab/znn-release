#include <zi/zunit/zunit.hpp>

#include "../volume_data.hpp"

using namespace znn::v4;

ZiTEST( VolumeData )
{
    real data[4*4*4] =
            { 1, 1, 1, 1,
              1, 1, 1, 1,
              1, 1, 1, 1,
              1, 1, 1, 1,

              1, 2, 1, 2,
              2, 2, 2, 2,
              1, 2, 1, 2,
              2, 2, 2, 2,

              3, 3, 3, 3,
              3, 3, 3, 3,
              3, 3, 3, 3,
              3, 3, 3, 3,

              3, 2, 3, 2,
              2, 2, 2, 2,
              3, 2, 3, 2,
              2, 2, 2, 2 };

    auto c = get_cube<real>(vec3i(4,4,4));
    std::copy(data, data + 4*4*4, c->data());

    volume_data<real> vd(c);
    EXPECT_EQ( vd.dim(),   vec3i(4,4,4) );
    EXPECT_EQ( vd.fov(),   vec3i(4,4,4) );
    EXPECT_EQ( vd.off(),   vec3i(0,0,0) );
    EXPECT_EQ( vd.bbox(),  box(vec3i(0,0,0),vec3i(4,4,4)) );
    EXPECT_EQ( vd.range(), box(vec3i(2,2,2),vec3i(3,3,3)) );

    vd.set_fov(vec3i(3,3,3));
    EXPECT_EQ( vd.range(), box(vec3i(1,1,1),vec3i(3,3,3)) );

    vd.set_offset(vec3i(-1,-1,-1));
    EXPECT_EQ( vd.bbox(),  box(vec3i(-1,-1,-1),vec3i(3,3,3)) );
    EXPECT_EQ( vd.range(), box(vec3i(0,0,0),vec3i(2,2,2)) );

    auto p = vd.get_patch(vec3i(0,0,0));
    EXPECT_EQ( size(*p), vec3i(3,3,3) );
    EXPECT_EQ( (*p)[0][0][0], static_cast<real>(1) );
    EXPECT_EQ( (*p)[1][1][2], static_cast<real>(2) );
    EXPECT_EQ( (*p)[2][2][2], static_cast<real>(3) );

    vd.set_fov(vec3i(2,2,2));
    auto p2 = vd.get_patch(vec3i(1,1,1));
    EXPECT_EQ( size(*p2), vec3i(2,2,2) );
    EXPECT_EQ( (*p2)[0][0][0], static_cast<real>(2) );
    EXPECT_EQ( (*p2)[0][0][1], static_cast<real>(2) );
    EXPECT_EQ( (*p2)[0][1][0], static_cast<real>(2) );
    EXPECT_EQ( (*p2)[0][1][1], static_cast<real>(1) );
    EXPECT_EQ( (*p2)[1][0][0], static_cast<real>(3) );
    EXPECT_EQ( (*p2)[1][0][1], static_cast<real>(3) );
    EXPECT_EQ( (*p2)[1][1][0], static_cast<real>(3) );
    EXPECT_EQ( (*p2)[1][1][1], static_cast<real>(3) );
}

ZiTEST( WritableVolumeData )
{
    real data[4*4*4] =
            { 1, 1, 1, 1,
              1, 1, 1, 1,
              1, 1, 1, 1,
              1, 1, 1, 1,

              1, 2, 1, 2,
              2, 2, 2, 2,
              1, 2, 1, 2,
              2, 2, 2, 2,

              3, 3, 3, 3,
              3, 3, 3, 3,
              3, 3, 3, 3,
              3, 3, 3, 3,

              3, 2, 3, 2,
              2, 2, 2, 2,
              3, 2, 3, 2,
              2, 2, 2, 2 };

    auto c = get_cube<real>(vec3i(4,4,4));
    std::copy(data, data + 4*4*4, c->data());
    rw_volume_data<real> vd(c,vec3i(3,3,3));

    real patch[3*3*3] =
            { 5, 1, 1,
              7, 5, 1,
              1, 1, 1,

              1, 2, 1,
              2, 6, 2,
              1, 2, 1,

              3, 3, 3,
              3, 2, 3,
              3, 3, 3, };

    auto p = get_cube<real>(vec3i(3,3,3));
    std::copy(patch, patch + 3*3*3, p->data());

    vd.set_patch(vec3i(1,1,1), p);
    auto p2 = vd.get_patch(vec3i(1,1,1));
    for ( size_t i = 0; i < p->num_elements(); ++i )
        EXPECT_EQ( p->data()[i], p2->data()[i] );
}