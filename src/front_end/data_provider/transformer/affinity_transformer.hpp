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

#include "transformer.hpp"
#include "../sample.hpp"
#include "../../../core/volume_utils.hpp"


namespace zi {
namespace znn {

class affinity_transformer: virtual public transformer
{
public:
    virtual void transform( sample_ptr s )
    {
    	// random transpose
        if ( rand() % 2 )
        {
            volume_utils::transpose(s->inputs);
            transpose_affinity(s->labels);
            transpose_affinity(s->masks);
        }

        // random flip
        std::size_t dim = rand() % 8;
        {
            volume_utils::flipdim(s->inputs, dim);
            volume_utils::flipdim(s->labels, dim);
            volume_utils::flipdim(s->masks,  dim);
        }
    }

private:
    template <typename T>
    void transpose_affinity( std::list<T>& affs )
    {
        STRONG_ASSERT(affs.size() == 3);

        T xaff = affs.front(); affs.pop_front();
        T yaff = affs.front(); affs.pop_front();
        T zaff = affs.front(); affs.pop_front();

        // x-affinity and y-affinity should be exchanged
        affs.push_back(volume_utils::transpose(yaff));
        affs.push_back(volume_utils::transpose(xaff));
        affs.push_back(volume_utils::transpose(zaff));
    }

}; // abstract class affinity_transformer

typedef boost::shared_ptr<affinity_transformer> affinity_transformer_ptr;

}} // namespace zi::znn

#endif // ZNN_AFFINITY_TRANSFORMER_HPP_INCLUDED
