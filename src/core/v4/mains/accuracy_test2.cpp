#include "network/parallel/network.hpp"

#include "network/parallel/filter_edge.hpp"
#include "network/parallel/dummy_edge.hpp"
//
#include "network/parallel/max_pooling_edge.hpp"
#include "network/parallel/input_nodes.hpp"
#include "network/parallel/transfer_nodes.hpp"
#include "utils/waiter.hpp"
#include "utils/accumulator.hpp"
#include "network/trivial/trivial_network.hpp"
#include "network/trivial/trivial_fft_network.hpp"

using namespace znn::v4;

void compare_options( options const & a, options const & b )
{
    std::string sa, sb;
    if ( a.contains("biases") )
    {
        std::cout << "found biases\n";
        sa = a.require_as<std::string>("biases");
        sb = b.require_as<std::string>("biases");
    }
    if ( a.contains("filters") )
    {
        std::cout << "found filters\n";
        sa = a.require_as<std::string>("filters");
        sb = b.require_as<std::string>("filters");
    }

    size_t s = sa.size();

    if ( s == 0 ) return;


    std::cout << "Found string of size: " << s << " " << sb.size() << std::endl;

    s /= sizeof(real);

    real maxx = 0;

    const real * pa = reinterpret_cast<const real*>(sa.data());
    const real * pb = reinterpret_cast<const real*>(sb.data());

    for ( size_t i = 0; i < s; ++i )
        maxx = std::max(maxx, std::abs(pa[i]-pb[i]));

    std::cout << "MAX DIFF: " << maxx << std::endl;

}

int main()
{
    // {
    //     auto v = get_cube<real>({1,1,9});
    //     (*v)[0][0][1] = 1;
    //     auto f = fftw::forward(std::move(v));

    //     std::cout << *f << std::endl;

    //     complex c(0,1);

    //     //c = std::pow(c, static_cast<real>(1)/9);
    //     //complex d = c;
    //     const real pi = 3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651e+00;

    //     for ( int i = 0; i < 9; ++i )
    //     {
    //         complex d = c;
    //         d *= 2 * i;
    //         d *= pi;
    //         d /= 9;
    //         d = std::exp(d);
    //         std::cout << d << "\n";
    //     }
    // }
    // return 0;

    std::vector<options> nodes(9);
    std::vector<options> edges(8);

    nodes[0].push("name", "input")
        .push("type", "input")
        .push("size", 1);

    edges[0].push("name", "conv1")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "3,3,1")
        .push("input", "input")
        .push("output", "nl1");

    nodes[1].push("name", "nl1")
        .push("type", "transfer")
        .push("function", "tanh")
        .push("function_args", "1.7159,0.6666")
        .push("size", 48);

    edges[1].push("name", "pool1")
        .push("type", "max_filter")
        .push("size", "3,3,1")
        .push("stride", "3,3,1")
        .push("input", "nl1")
        .push("output", "mp1");

    nodes[2].push("name", "mp1")
        .push("type", "sum")
        .push("size", 48);

    edges[2].push("name", "conv2")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "4,4,1")
        .push("input", "mp1")
        .push("output", "nl2");

    nodes[3].push("name", "nl2")
        .push("type", "transfer")
        .push("function", "tanh")
        .push("function_args", "1.7159,0.6666")
        .push("size", 48);

    edges[3].push("name", "pool2")
        .push("type", "max_filter")
        .push("size", "2,2,1")
        .push("stride", "2,2,1")
        .push("input", "nl2")
        .push("output", "mp2");

    nodes[4].push("name", "mp2")
        .push("type", "sum")
        .push("size", 48);

    edges[4].push("name", "conv3")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "4,4,1")
        .push("input", "mp2")
        .push("output", "nl3");

    nodes[5].push("name", "nl3")
        .push("type", "transfer")
        .push("function", "tanh")
        .push("function_args", "1.7159,0.6666")
        .push("size", 48);

    edges[5].push("name", "pool3")
        .push("type", "max_filter")
        .push("size", "2,2,1")
        .push("stride", "2,2,1")
        .push("input", "nl3")
        .push("output", "mp3");

    nodes[6].push("name", "mp3")
        .push("type", "sum")
        .push("size", 48);

    edges[6].push("name", "conv4")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "3,3,1")
        .push("input", "mp3")
        .push("output", "nl4");

    nodes[7].push("name", "nl4")
        .push("type", "transfer")
        .push("function", "tanh")
        .push("function_args", "1.7159,0.6666")
        .push("size", 100);

    edges[7].push("name", "conv5")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "1,1,1")
        .push("input", "nl4")
        .push("output", "output");

    nodes[8].push("name", "output")
        .push("type", "transfer")
        .push("function", "linear")
        .push("function_args", "1,0")
        .push("size", 2);



    trivial_network::network tn(nodes, edges, {100,100,1});
    //tn.set_eta(0.1);

    std::map<std::string, std::vector<cube_p<real>>> in, in2;
    in["input"].push_back(get_cube<real>({164,164,1}));

    uniform_init init(1);
    init.initialize(*in["input"][0]);

    in2["input"].push_back(get_copy(*in["input"][0]));

    auto nd = tn.serialize();

    parallel_network::network tn2(nd.first, nd.second, {100,100,1});

    std::cout << "QURAC\n\n";

    zi::wall_timer wt;
    wt.reset();


    auto r2 = tn2.forward(std::move(in2));
    std::cout << "FFT FWD: " << wt.elapsed<real>() << std::endl;
    wt.reset();

    auto r = tn.forward(std::move(in));
    std::cout << "DIRECT FWD: " << wt.elapsed<real>() << std::endl;
    wt.reset();

    for ( auto & it: r )
    {
        std::cout << it.first << "\n";
        for ( auto & v: it.second )
        {
            std::cout << *v << "\n\n";
        }
    }


    for ( auto & it: r2 )
    {
        std::cout << it.first << "\n";
        for ( auto & v: it.second )
        {
                std::cout << *v << "\n\n";
        }
    }

    wt.reset();

    r = tn.backward(std::move(r));

    std::cout << "-----------" << std::endl;
    std::cout << "DIRECT BWD: " << wt.elapsed<real>() << std::endl;
    wt.reset();

    r2 = tn2.backward(std::move(r2));
    std::cout << "FFT BWD: " << wt.elapsed<real>() << std::endl;
    wt.reset();



    //return 0;

    std::map<std::string, std::vector<cube_p<real>>> inx1, inx2;
    inx1["input"].push_back(get_cube<real>({164,164,1}));


    init.initialize(*inx1["input"][0]);

    inx2["input"].push_back(get_copy(*inx1["input"][0]));

    wt.reset();
    r = tn.forward(std::move(inx1));
    std::cout << "DIRECT FWD: " << wt.elapsed<real>() << std::endl;
    wt.reset();
    r2 = tn2.forward(std::move(inx2));
    std::cout << "FFT BWD: " << wt.elapsed<real>() << std::endl;
    wt.reset();

    for ( auto & it: r )
    {
        std::cout << it.first << "\n";
        for ( auto & v: it.second )
        {
            std::cout << *v << "\n\n";
        }
    }


    for ( auto & it: r2 )
    {
        std::cout << it.first << "\n";
        for ( auto & v: it.second )
        {
            std::cout << *v << "\n\n";
        }
    }

    wt.reset();
    r = tn.backward(std::move(r));
    std::cout << "DIRECT BWD: " << wt.elapsed<real>() << std::endl;
    wt.reset();
        std::cout << "-----------" << std::endl;
    r2 = tn2.backward(std::move(r2));
    std::cout << "FFT BWD: " << wt.elapsed<real>() << std::endl;
    wt.reset();

    auto net1 = tn.serialize();
    auto net2 = tn2.serialize();

    std::cout << "Will compare\n";

    for ( int i = 0; i < net1.first.size(); ++i )
    {
        compare_options(net1.first[i], net2.first[i]);
    }

    for ( int i = 0; i < net1.second.size(); ++i )
    {
        compare_options(net1.second[i], net2.second[i]);
    }


    std::cout << "HERE\n\n\n\n";

    //int i;
    //std::cin >> i;


    std::cout << "HERE\n\n\n\n";

    //int i;
    //std::cin >> i;

    //std::cout << (*r["input"][0]) << "\n\n\n\n"; // - *r2["input"][0]);

    return 0;

    for ( auto & it: r )
    {
        std::cout << it.first << "\n";
        for ( auto & v: it.second )
        {
            std::cout << size(*v) << "\n\n";
        }
    }

    auto state = tn.serialize();

    std::cout << "Node Groups: " << std::endl;
    for ( auto & n: state.first )
    {
        n.dump();
        std::cout << "\n";
    }

    std::cout << "Edge Groups: " << std::endl;
    for ( auto & n: state.second )
    {
        n.dump();
        std::cout << "\n";
    }

    return 0;
}
