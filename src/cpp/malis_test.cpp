#include "network/computation/make_affinity.hpp"
#include "network/computation/get_segmentation.hpp"
#include <zi/zargs/zargs.hpp>

using namespace znn::v4;

namespace malis_test {

template<typename F, typename T>
inline cube_p<T> load( std::string const & fname, vec3i const & sz )
{
    FILE* fvol = fopen(fname.c_str(), "r");

    STRONG_ASSERT(fvol);

    auto ret = get_cube<T>(sz);
    F v;

    for ( long_t z = 0; z < sz[0]; ++z )
        for ( long_t y = 0; y < sz[1]; ++y )
            for ( long_t x = 0; x < sz[2]; ++x )
            {
                static_cast<void>(fread(&v, sizeof(F), 1, fvol));
                (*ret)[z][y][x] = static_cast<T>(v);
            }

    fclose(fvol);

    return ret;
}

template<typename F, typename T>
inline bool write( std::string const & fname, cube_p<T> vol )
{
    FILE* fvol = fopen(fname.c_str(), "w");

    STRONG_ASSERT(fvol);

    vec3i sz = size(*vol);
    F v;

    for ( long_t z = 0; z < sz[0]; ++z )
        for ( long_t y = 0; y < sz[1]; ++y )
            for ( long_t x = 0; x < sz[2]; ++x )
            {
                v = static_cast<T>((*vol)[z][y][x]);
                static_cast<void>(fwrite(&v, sizeof(F), 1, fvol));
            }

    fclose(fvol);

    return true;
}

}

int main(int argc, char** argv)
{
    // path to the input volume
    std::string ifname = argv[1];

    // input volume size
    int64_t x = atoi(argv[2]);
    int64_t y = atoi(argv[3]);
    int64_t z = atoi(argv[4]);
    vec3i   s = {z,y,x};

    // path to the output volume
    std::string ofname = argv[5];

    // affinity graph dimension
    size_t dim = atoi(argv[6]);

    // load input volume
    auto bmap = malis_test::load<double,int>(ifname,s);
#if defined( DEBUG )
    std::cout << "\n[bmap]\n" << *bmap << std::endl;
#endif

    // make affinity
    auto aff = make_affinity( *bmap, dim );
#if defined( DEBUG )
    std::cout << "\n[aff]" << std::endl;
    for ( auto& a: aff )
    {
        std::cout << *a << "\n\n";
    }
#endif

    // connected component segmentation
    auto seg = get_segmentation( aff );
#if defined( DEBUG )
    std::cout << "[seg]\n" << *seg << std::endl;
#endif

    // write
    malis_test::write<double,real>(ofname + "xaff.bin",aff[0]);
    malis_test::write<double,real>(ofname + "yaff.bin",aff[1]);
    if ( aff.size() == 3 )
    {
        malis_test::write<double,real>(ofname + "zaff.bin",aff[2]);
    }
    malis_test::write<int,int>(ofname + "seg.bin",seg);
}
