#include "network/parallel/network.hpp"
#include "network/trivial/trivial_forward_network.hpp"
#include <zi/zargs/zargs.hpp>

using namespace znn::v4;

int main(int argc, char** argv)
{
    std::vector<options> nodes(11);
    std::vector<options> edges(10);

    nodes[0].push("name", "input")
        .push("type", "input")
        .push("size", 1);

    edges[0].push("name", "conv1")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "1,6,6")
        .push("stride", "1,1,1")
        .push("input", "input")
        .push("output", "nl1");

    nodes[1].push("name", "nl1")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 12);

    edges[1].push("name", "pool1")
        .push("type", "max_filter")
        .push("size", "1,2,2")
        .push("stride", "1,2,2")
        .push("input", "nl1")
        .push("output", "nl2");

    nodes[2].push("name", "nl2")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 12);

    edges[2].push("name", "conv2")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "1,4,4")
        .push("stride", "1,1,1")
        .push("input", "nl2")
        .push("output", "nl3");

    nodes[3].push("name", "nl3")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 24);

    edges[3].push("name", "pool2")
        .push("type", "max_filter")
        .push("size", "1,2,2")
        .push("stride", "1,2,2")
        .push("input", "nl3")
        .push("output", "nl4");

    nodes[4].push("name", "nl4")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 24);

    edges[4].push("name", "conv3")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "4,4,4")
        .push("stride", "1,1,1")
        .push("input", "nl4")
        .push("output", "nl5");

    nodes[5].push("name", "nl5")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 36);

    edges[5].push("name", "pool3")
        .push("type", "max_filter")
        .push("size", "2,2,2")
        .push("stride", "2,2,2")
        .push("input", "nl5")
        .push("output", "nl6");

    nodes[6].push("name", "nl6")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 36);

    edges[6].push("name", "conv4")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "2,4,4")
        .push("stride", "1,1,1")
        .push("input", "nl6")
        .push("output", "nl7");

    nodes[7].push("name", "nl7")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 48);

    edges[7].push("name", "pool4")
        .push("type", "max_filter")
        .push("size", "2,2,2")
        .push("stride", "2,2,2")
        .push("input", "nl7")
        .push("output", "nl8");

    nodes[8].push("name", "nl8")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 48);


    edges[8].push("name", "conv5")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "2,4,4")
        .push("stride", "1,1,1")
        .push("input", "nl8")
        .push("output", "nl9");


    nodes[9].push("name", "nl9")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 48);


    edges[9].push("name", "conv6")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "1,1,1")
        .push("stride", "1,1,1")
        .push("input", "nl9")
        .push("output", "nl10");


    nodes[10].push("name", "nl10")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 4);

    int64_t x = 48;
    int64_t y = 48;
    int64_t z = 9;

    if ( argc >= 4 )
    {
        x = atoi(argv[1]);
        y = atoi(argv[2]);
        z = atoi(argv[3]);
    }

    size_t nx = 1; //std::thread::hardware_concurrency();

    if ( argc >= 5 )
    {
        nx = atoi(argv[4]);
    }

    size_t nt = 36;

    if ( argc >= 6 )
    {
        nt = atoi(argv[5]);
    }

    trivial_forward_network::forward_network net(nt);
    net.add_layer(vec3i(1,6,6),vec3i(1,2,2),12);
    net.add_layer(vec3i(1,4,4),vec3i(1,2,2),24);
    net.add_layer(vec3i(4,4,4),vec3i(2,2,2),36);
    net.add_layer(vec3i(2,4,4),vec3i(2,2,2),48);
    net.add_layer(vec3i(2,4,4),vec3i(1,1,1),48);
    net.add_layer(vec3i(1,1,1),vec3i(1,1,1),4);

    auto in_size = net.init( vec3i(z,y,x) );

    net.warmup();

    std::cout << "WARMUP DONE!" << std::endl;

    auto initf = std::make_shared<gaussian_init>(0,0.01);

    //size_t nx = 50;

    std::vector<std::vector<cube_p<real>>> in(1);

    for ( size_t i = 0; i < nx; ++i )
    {
        in[0].push_back(get_cube<real>(in_size));
        initf->initialize(in[0][i]);
    }

    net.forward(in);


    return 0;

    size_t tc = std::thread::hardware_concurrency();

    if ( argc == 5 )
    {
        tc = atoi(argv[4]);
    }

    //parallel_network::network::optimize_forward(nodes, edges, {z,y,x}, tc , 10);
    trivial_forward_network::network(nodes, edges, {z,y,x});

    auto r = get_cube<real>(vec3i(1024,1024,32));

    zi::wall_timer wt;
    wt.reset();

    auto ss = fftw::forward(std::move(r));

    std::cout << wt.elapsed<real>() << std::endl;

    {
        auto r = get_cube<real>(vec3i(1,6,6));

        zi::wall_timer wt;
        wt.reset();

        auto s1 = fftw::forward_pad(std::move(r), vec3i(512,512,64));
        std::cout << wt.elapsed<real>() << std::endl;

        auto s2 = fftw::forward_pad(std::move(r), vec3i(512,512,64));
        std::cout << wt.elapsed<real>() << std::endl;

        *s1 *= *s2;
        *s1 *= *s2;
        std::cout << wt.elapsed<real>() << std::endl;
    }



}
