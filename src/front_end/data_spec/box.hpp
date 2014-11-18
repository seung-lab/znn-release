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

#ifndef ZNN_BOX_HPP_INCLUDED
#define ZNN_BOX_HPP_INCLUDED

#include "../../core/types.hpp"


namespace zi {
namespace znn {

class box
{
private:
	vec3i	uc;
	vec3i	lc;
	vec3i	sz;


public:
	bool contains( const vec3i& p ) const
	{
		return !(p[0] < uc[0]) && !(lc[0] <= p[0]) &&
			   !(p[1] < uc[1]) && !(lc[1] <= p[1]) &&
			   !(p[2] < uc[2]) && !(lc[2] <= p[2]);
	}

	bool contains( const box& rhs ) const
	{
		box b = intersect(rhs);
		return (!b.empty() && (b == rhs));
	}

	const vec3i& upper_corner() const
	{
		return uc;
	}

	const vec3i& lower_corner() const
	{
		return lc;
	}

	const vec3i& size() const
	{
		return sz;
	}

	bool empty() const
	{
		return (vec3i::zero == sz);
	}


public:
	static
	box intersect( const box& a, const box& b )
	{
		// upper corner
		vec3i uc;
		vec3i uc1 = a.upper_corner();
		vec3i uc2 = b.upper_corner();
		uc[0] = std::max(uc1[0],uc2[0]);
		uc[1] = std::max(uc1[1],uc2[1]);
		uc[2] = std::max(uc1[2],uc2[2]);

		// lower corner
		vec3i lc;
		vec3i lc1 = a.lower_corner();
		vec3i lc2 = b.lower_corner();
		lc[0] = std::min(lc1[0],lc2[0]);
		lc[1] = std::min(lc1[1],lc2[1]);
		lc[2] = std::min(lc1[2],lc2[2]);

		return box(uc,lc);
	}

	box intersect( const box& rhs ) const
	{
		return intersect(*this,rhs);
	}

	static
	box merge( const box& a, const box& b )
	{
		// upper corner
		vec3i uc;
		vec3i uc1 = a.upper_corner();
		vec3i uc2 = b.upper_corner();
		uc[0] = std::min(uc1[0],uc2[0]);
		uc[1] = std::min(uc1[1],uc2[1]);
		uc[2] = std::min(uc1[2],uc2[2]);

		// lower corner
		vec3i lc;
		vec3i lc1 = a.lower_corner();
		vec3i lc2 = b.lower_corner();
		lc[0] = std::max(lc1[0],lc2[0]);
		lc[1] = std::max(lc1[1],lc2[1]);
		lc[2] = std::max(lc1[2],lc2[2]);

		return box(uc,lc);
	}

	box merge( const box& rhs ) const
	{
		return merge(*this,rhs);
	}

	void translate( const vec3i& offset )
	{
		uc += offset;
		lc += offset;
	}


public:
	bool operator==( const box& rhs )
	{
		return (uc == rhs.uc) && (lc == rhs.lc);
	}

	box operator+( const vec3i& offset )
	{
		return box(uc+offset,lc+offset);
	}

	box operator+( const box& rhs )
	{
		return this->merge(rhs);
	}


public:
	friend std::ostream&
	operator<<( std::ostream& os, const box& rhs )
	{
		return (os << rhs.uc << " " << rhs.lc);
	}


public:
	static
	box centered_box( const vec3i& center, const vec3i& size )
	{
		vec3i hsz = size/vec3i(2,2,2);

		STRONG_ASSERT(center[0] >= hsz[0]);
		STRONG_ASSERT(center[1] >= hsz[1]);
		STRONG_ASSERT(center[2] >= hsz[2]);

		vec3i uc = center - hsz;
		vec3i lc = uc + size;

		return box(uc,lc);
	}


public:
	box( const vec3i& _uc = vec3i::zero, const vec3i& _lc = vec3i::zero )
		: uc(_uc)
		, lc(_lc)
	{
		lc[0] = std::max(uc[0],lc[0]);
		lc[1] = std::max(uc[1],lc[1]);
		lc[2] = std::max(uc[2],lc[2]);

		sz = lc - uc;
	}

	~box()
	{}

}; // class box

}} // namespace zi::znn

#endif // ZNN_BOX_HPP_INCLUDED