#include "cube/cube.hpp"
#include "network/parallel/network.hpp"
#include <zi/zargs/zargs.hpp>

using namespace znn::v4;

typedef std::map<std::string, std::vector<cube_p<real>>> data_type;
typedef std::map<std::string, std::pair<vec3i,size_t>>   data_spec;

data_type get_random_data(data_spec const & spec, bool display_value = true)
{
    data_type ret;
    std::shared_ptr<initializator<real>> init =
        std::make_shared<uniform_init>(0,1);

    for ( auto& l: spec )
    {
        auto name = l.first;
        auto pair = l.second;
        vec3i sz = pair.first;
        std::vector<cube_p<real>> data;
        for ( size_t i = 0; i < pair.second; ++i )
        {
            auto r = get_cube<real>(sz);
            init->initialize(*r);
            data.push_back(r);
        }
        ret[name] = data;
    }

    return ret;
}

void display( data_type const & data )
{
    for ( auto& l: data )
    {
        auto name  = l.first;
        auto nodes = l.second;

        for ( size_t i = 0; i < nodes.size(); ++i )
        {
            std::cout << "[" << name << ":" << i << "] " << "\n";
            std::cout << *nodes[i] << "\n\n";
        }
    }
}

int main(int argc, char** argv)
{
    std::vector<options> nodes, edges;

    std::string fname(argv[1]);

    parse_net_file(nodes, edges, fname);

    int64_t x = 9;
    int64_t y = 9;
    int64_t z = 9;

    if ( argc >= 5 )
    {
        x = atoi(argv[2]);
        y = atoi(argv[3]);
        z = atoi(argv[4]);
    }

    size_t tc = std::thread::hardware_concurrency();

    if ( argc == 6 )
    {
        tc = atoi(argv[5]);
    }

    vec3i out_sz(x,y,z);
    parallel_network::network net(nodes,edges,out_sz,tc);
    vec3i in_sz = out_sz + net.fov() - vec3i::one;

    // forward
    auto insample = get_random_data(net.inputs());
    display(insample);
    auto prop = net.forward(std::move(insample));
    display(prop);

    // backward
    auto outsample = get_random_data(net.outputs());
    display(outsample);
    auto ret = net.backward(std::move(outsample));
    display(ret);

    std::cout << "Done." << std::endl;
}
