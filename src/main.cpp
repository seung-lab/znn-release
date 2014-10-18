//
// Copyright (C) 2014  Aleksandar Zlateski <zlateski@mit.edu>
//                     Kisuk Lee           <kisuklee@mit.edu>
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

#include "front_end/options.hpp"
#include "core/network.hpp"

#include <iostream>
#include <zi/time.hpp>
#include <zi/zargs/zargs.hpp>

ZiARG_string(options, "", "Option file path");
ZiARG_bool(test_only, false, "Test only");

using namespace zi::znn;

int main(int argc, char** argv)
{
    // options
    zi::parse_arguments(argc, argv);
    options_ptr op = options_ptr(new options(ZiARG_options));
    op->save();

    // create network
    network net(op);
    
    // training/forward scanning
    if( ZiARG_test_only )
    {
        net.forward_scan();
    }
    else
    {
        net.train();
    }
}