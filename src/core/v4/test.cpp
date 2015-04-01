#include "network/parallel/network.hpp"
#include "network/parallel/nodes.hpp"
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

    s /= sizeof(double);

    double maxx = 0;

    const double * pa = reinterpret_cast<const double*>(sa.data());
    const double * pb = reinterpret_cast<const double*>(sb.data());

    for ( size_t i = 0; i < s; ++i )
        maxx = std::max(maxx, std::abs(pa[i]-pb[i]));

    std::cout << "MAX DIFF: " << maxx << std::endl;

}

const double pi = 3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651328230664709384460955058223172535940812848111745028410270193852110555964462294895493038196e0;

std::vector<std::vector<complex>> roots;

void calc_roots()
{
    roots.resize(1000);
    for ( size_t N = 1; N < 1000; ++ N )
    {
        roots[N].resize(N+1);

        roots[N][N] = roots[N][0] = complex(1,0);

        for ( std::size_t i = 1; i <= N; ++i )
        {
            double cosine = std::cos( -pi * 2 * i / N );
            double sine   = std::sin( -pi * 2 * i / N );
            roots[N][i] = complex(cosine, sine);
        }

        // for ( std::size_t i = 1; i < N/2; ++i )
        // {
        //     roots[N][i+N/2] = -roots[N][i];
        // }
    }
}

inline cube_p<complex> myfft( cube<double> const & in, const vec3i& osize )
{
    auto sz  = size(in);
    auto cs  = fft_complex_size(osize);
    auto ret = get_cube<complex>(cs);

    auto const & xroots = roots[osize[0]];
    auto const & yroots = roots[osize[1]];
    auto const & zroots = roots[osize[2]];

    auto& r = *ret;

    for ( int64_t x = 0; x < cs[0]; ++x )
        for ( int64_t y = 0; y < cs[1]; ++y )
            for ( int64_t z = 0; z < cs[2]; ++z )
            {
                auto& d = r[x][y][z];
                d = 0;

                for ( int64_t ix = 0; ix < sz[0]; ++ix )
                    for ( int64_t iy = 0; iy < sz[1]; ++iy )
                        for ( int64_t iz = 0; iz < sz[2]; ++iz )
                        {
                            d += in[ix][iy][iz] * xroots[ (x*ix) % osize[0] ] *
                                yroots[ (y*iy) % osize[1] ] *
                                zroots[ (z*iz) % osize[2] ];
                        }
            }

    return ret;

}

int main()
{
    // for ( int64_t sz = 1; sz < 128; ++sz )
    // {
    //     auto x = fftw_plans.get_forward(vec3i{sz,sz,sz});
    //     std::cout << sz << ", " << fftw_cost(x) << " ;\n";
    // }

    // return 0;

    // calc_roots();

    // for ( int64_t sz = 1; sz < 7; ++sz )
    // {

    //     vec3i fsize{sz,sz,sz};
    //     std::cout << "Filter size: " << fsize << "\n";

    //     auto v1 = get_cube<double>(fsize);
    //     auto v2 = get_cube<double>(fsize);
    //     uniform_init uinit(1);
    //     uinit.initialize(*v1);
    //     *v2 = *v1;

    //     zi::wall_timer wt;
    //     wt.reset();
    //     auto x = fftw::forward_pad(v1, {120,120,5});
    //     std::cout << "\tFFTW: " << wt.elapsed<double>() << std::endl;


    //     wt.reset();
    //     auto y = myfft(*v2, {120,120,5});
    //     std::cout << "\tMY FFT:" << wt.elapsed<double>() << std::endl;

    //     *y -= *x;

    //     double maxdiff = 0;
    //     for ( size_t i = 0; i < y->num_elements(); ++i )
    //     {
    //         maxdiff = std::max(maxdiff, std::abs(y->data()[i]));
    //     }


    //     std::cout << "\tMAX DIF: " << maxdiff << std::endl;
    // }

    // return 0;

    std::vector<options> nodes(11);
    std::vector<options> edges(12);


    edges[10].push("name", "conv1a")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "2,2,1")
        .push("input", "input")
        .push("output", "nl1a");

    nodes[10].push("name", "nl1a")
        .push("type", "transfer")
        .push("function", "tanh")
        .push("function_args", "1.7159,0.6666")
        .push("size", 8);

    edges[11].push("name", "conv1b")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "2,2,1")
        .push("input", "nl1a")
        .push("output", "nl1");

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

    edges[8].push("name", "conv1x")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "3,3,1")
        .push("input", "input")
        .push("output", "nl1x");

    nodes[9].push("name", "nl1x")
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

    edges[9].push("name", "pool1x")
        .push("type", "max_filter")
        .push("size", "3,3,1")
        .push("stride", "3,3,1")
        .push("input", "nl1x")
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

    nodes[8] = options{{{"name", "output"},{"type", "transfer"},{"function", "linear"},
                        {"function_args", "1,0"}, {"size","2"}}};


    trivial_fft_network::network tn1(nodes, edges, {11,11,2});

    tn1.set_eta(0.00001);
    auto nd = tn1.serialize();

    parallel_network::network tn2(nd.first, nd.second, {11,11,2}, 16);


    std::vector<std::map<std::string, std::vector<cube_p<double>>>>
        inputs1, inputs2, outputs1, outputs2;

    uniform_init init(1);

    for ( size_t n = 0; n < 100; ++n )
    {
        std::map<std::string, std::vector<cube_p<double>>> in1, in2;

        in1["input"].push_back(get_cube<double>({75,75,2}));
        init.initialize(*in1["input"][0]);
        in2["input"].push_back(get_copy(*in1["input"][0]));

        inputs1.push_back(in1);
        inputs2.push_back(in2);

        std::map<std::string, std::vector<cube_p<double>>> out1, out2;

        out1["output"].push_back(get_cube<double>({11,11,2}));
        out1["output"].push_back(get_cube<double>({11,11,2}));

        init.initialize(*out1["output"][0]);
        init.initialize(*out1["output"][1]);

        out2["output"].push_back(get_copy(*out1["output"][0]));
        out2["output"].push_back(get_copy(*out1["output"][1]));

        outputs1.push_back(out1);
        outputs2.push_back(out2);

    }

    //tn2.set_eta(0.00001);
    //tn2.set_eta(0.0001);

    std::cout << "HERE" << std::endl;



    for ( int xx = 0; xx < 20; ++xx )
    {
        auto r2 = tn2.forward(std::move(inputs2[xx]));
        r2 = tn2.backward(std::move(outputs2[xx]));
        //tn2.zap();
        std::cout << "NET2: " << xx << std::endl;
    }

    for ( int xx = 0; xx < 20; ++xx )
    {
        auto r1 = tn1.forward(std::move(inputs1[xx]));
        r1 = tn1.backward(std::move(outputs1[xx]));
        std::cout << "NET1: " << xx << std::endl;
    }

    for ( int xx = 20; xx < 22; ++xx )
    {
        // for ( auto & it: r )
        // {
        //     std::cout << it.first << "\n";
        //     for ( auto & v: it.second )
        //     {
        //         std::cout << *v << "\n\n";
        //     }
        // }

        auto r1 = tn1.forward(std::move(inputs1[xx]));
        auto r2 = tn2.forward(std::move(inputs2[xx]));

        // for ( auto & it: r2 )
        // {
        //     std::cout << it.first << "\n";
        //     for ( auto & v: it.second )
        //     {
        //         std::cout << *v << "\n\n";
        //     }
        // }

        r1 = tn1.backward(std::move(outputs1[xx]));
        r2 = tn2.backward(std::move(outputs2[xx]));

        //tn1.zap();
        auto net1 = tn1.serialize();
        //tn2.zap();
        auto net2 = tn2.serialize();

        std::cout << "Will compare\n";

        for ( size_t i = 0; i < net1.first.size(); ++i )
        {
            compare_options(net1.first[i], net2.first[i]);
        }

        for ( size_t i = 0; i < net1.second.size(); ++i )
        {
            compare_options(net1.second[i], net2.second[i]);
        }
    }


    return 0;
}
