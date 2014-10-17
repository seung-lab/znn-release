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
    zi::parse_arguments( argc, argv );
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