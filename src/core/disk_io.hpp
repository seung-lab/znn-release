//
// Copyright (C) 2015-present  Aleksandar Zlateski <zlateski@mit.edu>
// ------------------------------------------------------------------
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

#ifndef ZNN_CORE_DISK_IO_HPP_INCLUDED
#define ZNN_CORE_DISK_IO_HPP_INCLUDED

#include <string>
#include <fstream>

namespace zi {
namespace znn {
namespace io {

std::string file_get_contents(const std::string& fname)
{
    std::ifstream fin(fname.c_str());
    if ( !fin ) return "";

    std::string str( (std::istreambuf_iterator<char>(fin)),
                     (std::istreambuf_iterator<char>()) );
    fin.close();

    return str;
}

}}} // namespace zi::znn::io


#endif // ZNN_CORE_DISK_IO_HPP_INCLUDED
