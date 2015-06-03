#pragma once

#include "../types.hpp"
#include "../cube/cube.hpp"
#include "../cube/cube_operators.hpp"


#include <utility>
#include <set>

namespace znn { namespace v4 {

inline cube_p<int> make_indices( const vec3i& s )
{
    auto ret = get_cube<int>(s);

    for ( int i = 0; i < static_cast<int>(ret->num_elements()); ++i )
    {
        ret->data()[i] = i;
    }

    return ret;
}


template<typename F>
inline void pooling_filter_pass_2( real *  head1,
                                   real *  end,
                                   int *     head2,
                                   size_t    stride,
                                   F const & cmp ) noexcept
{
    real * tail1 = head1 + stride;
    int *  tail2 = head2 + stride;

    while ( tail1 <= end )
    {
        if ( cmp(*tail1, *head1) )
        {
            *head1 = *tail1;
            *head2 = *tail2;
        }

        head1 += stride;
        head2 += stride;
        tail1 += stride;
        tail2 += stride;
    }
}

template<typename F>
inline void pooling_filter_pass_3( real *  head1,
                                   real *  end,
                                   int *     head2,
                                   size_t    stride,
                                   F const & cmp ) noexcept
{
    pooling_filter_pass_2(head1,end,head2,stride,cmp);
    pooling_filter_pass_2(head1,end-stride,head2,stride,cmp);
}

template<typename F>
inline void pooling_filter_pass_4( real *  head1,
                                   real *  end,
                                   int *     head2,
                                   size_t    stride,
                                   F const & cmp ) noexcept
{
    pooling_filter_pass_2(head1,end,head2,stride,cmp);
    pooling_filter_pass_2(head1,end-stride,head2,stride,cmp);
    pooling_filter_pass_2(head1,end-stride-stride,head2,stride,cmp);
}


template<typename F>
inline void pooling_filter_pass( real *  head1,
                                 real *  end,
                                 int *     head2,
                                 size_t    size,
                                 size_t    stride,
                                 F const & cmp ) noexcept
{
    ZI_ASSERT(size > 1);

    if ( size == 2 )
    {
        pooling_filter_pass_2(head1, end, head2, stride, cmp);
        return;
    }

    if ( size == 3 )
    {
        pooling_filter_pass_3(head1, end, head2, stride, cmp);
        return;
    }

    if ( size == 4 )
    {
        pooling_filter_pass_4(head1, end, head2, stride, cmp);
        return;
    }


    typedef std::pair<real,int> pair_type;

    auto cmpf =
        [&cmp](const pair_type& l, const pair_type& r)
        {
            return cmp(l.first, r.first) ? true :
            ( cmp(r.first, l.first) ? false : ( l.second < r.second) );
        };

    std::set<pair_type, decltype(cmpf)> set(cmpf);

    real * tail1 = head1;
    int *  tail2 = head2;

    ZI_ASSERT(size>0);

    for (; size > 1; --size, tail1 += stride, tail2 += stride )
    {
        //set.emplace(*tail1, *tail2);
        set.insert(pair_type(*tail1, *tail2));
    }

    ZI_ASSERT(tail1<=end);

    while ( tail1 <= end )
    {
        //set.emplace(*tail1, *tail2);
        set.insert(pair_type(*tail1, *tail2));
        pair_type r = *set.begin();

        set.erase(pair_type(*head1,*head2));

        *head1 = r.first ;
        *head2 = r.second;

        head1 += stride;
        head2 += stride;
        tail1 += stride;
        tail2 += stride;
    }
}

template<typename F>
inline void pooling_filter_pass_2_no_indices( real *  head1,
                                              real *  end,
                                              size_t    stride,
                                              F const & cmp ) noexcept
{
    real * tail1 = head1 + stride;

    while ( tail1 <= end )
    {
        if ( cmp(*tail1, *head1) )
        {
            *head1 = *tail1;
        }

        head1 += stride;
        tail1 += stride;
    }
}

template<typename F>
inline void pooling_filter_pass_3_no_indices( real *  head1,
                                              real *  end,
                                              size_t    stride,
                                              F const & cmp ) noexcept
{
    pooling_filter_pass_2_no_indices(head1,end,stride,cmp);
    pooling_filter_pass_2_no_indices(head1,end-stride,stride,cmp);
}

template<typename F>
inline void pooling_filter_pass_4_no_indices( real *  head1,
                                              real *  end,
                                              size_t    stride,
                                              F const & cmp ) noexcept
{
    pooling_filter_pass_2_no_indices(head1,end,stride,cmp);
    pooling_filter_pass_2_no_indices(head1,end-stride,stride,cmp);
    pooling_filter_pass_2_no_indices(head1,end-stride-stride,stride,cmp);
}


template<typename F>
inline void pooling_filter_pass_no_indices( real *  head1,
                                            real *  end,
                                            size_t    size,
                                            size_t    stride,
                                            F const & cmp ) noexcept
{
    ZI_ASSERT(size > 1);

    if ( size == 2 )
    {
        pooling_filter_pass_2_no_indices(head1, end, stride, cmp);
        return;
    }

    if ( size == 3 )
    {
        pooling_filter_pass_3_no_indices(head1, end, stride, cmp);
        return;
    }

    if ( size == 4 )
    {
        pooling_filter_pass_4_no_indices(head1, end, stride, cmp);
        return;
    }

    UNIMPLEMENTED();
}


}} // namespace znn::v4
