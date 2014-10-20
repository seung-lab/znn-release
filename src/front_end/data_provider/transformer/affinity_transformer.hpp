//
// Copyright (C) 2014  Kisuk Lee <kisuklee@mit.edu>
// ----------------------------------------------------------
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef ZNN_AFFINITY_TRANSFORMER_HPP_INCLUDED
#define ZNN_AFFINITY_TRANSFORMER_HPP_INCLUDED

#include "volume_transformer.hpp"
#include "../sample.hpp"
#include "../../../core/volume_utils.hpp"


namespace zi {
namespace znn {

class affinity_transformer: virtual public volume_transformer
{
public:
    virtual void transform( sample_ptr s )
    {
    	volume_transformer::transform(s);

        // crop
        crop_affinity(s->labels);
        crop_affinity(s->masks);
    }

private:
    template <typename T>
    T crop_affinity( T vol )
    {
        vec3i sz = size_of(vol);
        STRONG_ASSERT(sz[0] > 1);
        STRONG_ASSERT(sz[1] > 1);
        STRONG_ASSERT(sz[2] > 1);

        return volume_utils::crop(vol, vec3i::zero, sz-vec3i::one);
    }

    template <typename T>
    void crop_affinity( std::list<T>& vl )
    {
        FOR_EACH( it, vl )
        {
            (*it) = crop_affinity(*it);
        }
    }

}; // abstract class affinity_transformer

typedef boost::shared_ptr<affinity_transformer> affinity_transformer_ptr;

}} // namespace zi::znn

#endif // ZNN_AFFINITY_TRANSFORMER_HPP_INCLUDED
