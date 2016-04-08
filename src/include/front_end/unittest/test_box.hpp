#include <zi/zunit/zunit.hpp>

#include "../box.hpp"

using namespace znn::v4;

ZiTEST( BoxConstruction )
{
    // ---------------------------------------------------------------
    // default
    // ---------------------------------------------------------------
    box b;

    EXPECT_EQ( b.min(),     vec3i::zero );
    EXPECT_EQ( b.max(),     vec3i::zero );
    EXPECT_EQ( b.size(),    vec3i::zero );
    EXPECT_EQ( b.empty(),   true        );

    // ---------------------------------------------------------------
    // degenerate
    // ---------------------------------------------------------------
    b = box(vec3i::one,vec3i::one);

    EXPECT_EQ( b.min(),     vec3i::one  );
    EXPECT_EQ( b.max(),     vec3i::one  );
    EXPECT_EQ( b.size(),    vec3i::zero );
    EXPECT_EQ( b.empty(),   true        );

    // ---------------------------------------------------------------
    // normal
    // ---------------------------------------------------------------
    vec3i v1(1,2,3);
    vec3i v2(2,3,4);
    b = box(v1,v2);

    EXPECT_EQ( b.min(),     v1          );
    EXPECT_EQ( b.max(),     v2          );
    EXPECT_EQ( b.size(),    v2 - v1     );
    EXPECT_EQ( b.empty(),   false       );

    // ---------------------------------------------------------------
    // abnormal
    // ---------------------------------------------------------------
    b = box(vec3i(1,1,0),vec3i(0,0,1));

    EXPECT_EQ( b.min(),     vec3i::zero );
    EXPECT_EQ( b.max(),     vec3i::one  );
    EXPECT_EQ( b.size(),    vec3i::one  );
    EXPECT_EQ( b.empty(),   false       );
}

ZiTEST( BoxEquality )
{
    box b1(vec3i(0,0,0),vec3i(1,1,1)); // box(1,1,1):[0,0,0][1,1,1]
    box b2(vec3i(1,1,1),vec3i(0,0,0)); // box(1,1,1):[0,0,0][1,1,1]
    box b3(vec3i(1,1,0),vec3i(0,0,1)); // box(1,1,1):[0,0,0][1,1,1]
    box b4(vec3i(0,1,0),vec3i(1,0,1)); // box(1,1,1):[0,0,0][1,1,1]

    EXPECT_EQ( b1 == b2, true );
    EXPECT_EQ( b2 == b3, true );
    EXPECT_EQ( b3 == b4, true );
    EXPECT_EQ( b4 == b1, true );

    box b0(vec3i(1,1,1),vec3i(2,2,2)); // box(1,1,1):[1,1,1][2,2,2]

    EXPECT_EQ( b0 != b1, true );
    EXPECT_EQ( b0 != b2, true );
    EXPECT_EQ( b0 != b3, true );
    EXPECT_EQ( b0 != b4, true );
}

ZiTEST( BoxOperations )
{
    // ---------------------------------------------------------------
    // overlapping
    // ---------------------------------------------------------------
    box b1(vec3i::zero,vec3i(2,2,2)); // box(2,2,2):[0,0,0][2,2,2]
    box b2(vec3i::one, vec3i(3,3,3)); // box(2,2,2):[1,1,1][3,3,3]
    box b3(vec3i::one, vec3i(2,2,2)); // box(1,1,1):[1,1,1][2,2,2]
    box b4(vec3i::zero,vec3i(3,3,3)); // box(3,3,3):[0,0,0][3,3,3]

    EXPECT_EQ( b1.contains(b2),         false   );
    EXPECT_EQ( b1.contains(b3),         true    );
    EXPECT_EQ( b2.contains(b3),         true    );
    EXPECT_EQ( b4.contains(b1),         true    );
    EXPECT_EQ( b4.contains(b2),         true    );
    EXPECT_EQ( b4.contains(b3),         true    );
    EXPECT_EQ( b1.overlaps(b2),         true    );
    EXPECT_EQ( b1.intersect(b2),        b3      );
    EXPECT_EQ( b1.merge(b2),            b4      );
    EXPECT_EQ( b1 + b2,                 b4      );
    EXPECT_EQ( b1 + vec3i::one,         b2      );
    EXPECT_EQ( b2 + vec3i(-1,-1,-1),    b1      );
    EXPECT_EQ( b2 - vec3i::one,         b1      );
    EXPECT_EQ( b1 - vec3i(-1,-1,-1),    b2      );

    b1.translate(vec3i::one);
    EXPECT_EQ( b1, b2 );

    // ---------------------------------------------------------------
    // non-overlapping
    // ---------------------------------------------------------------
    b1 = box(vec3i(0,0,0),vec3i(1,1,1)); // box(1,1,1):[0,0,0][1,1,1]
    b2 = box(vec3i(2,2,2),vec3i(3,3,3)); // box(1,1,1):[2,2,2][3,3,3]
    b3 = box(vec3i(0,0,0),vec3i(3,3,3)); // box(3,3,3):[0,0,0][3,3,3]

    EXPECT_EQ( b1.overlaps(b2), false   );
    EXPECT_EQ( b1.merge(b2),    b3      );
    EXPECT_EQ( b1 + b2,         b3      );

    b4 = b1.intersect(b2);
    EXPECT_EQ( b4.empty(), true );

    // ---------------------------------------------------------------
    // centered box
    // ---------------------------------------------------------------
    b1 = box::centered_box(vec3i::one, vec3i(3,3,3));
    b2 = box(vec3i(0,0,0),vec3i(3,3,3));
    EXPECT_EQ( b1, b2 );

    b1 = box::centered_box(vec3i::one, vec3i(4,4,4));
    b2 = box(vec3i(-1,-1,-1),vec3i(3,3,3));
    EXPECT_EQ( b1, b2 );
}