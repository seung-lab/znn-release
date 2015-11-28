#include "flow_graph/computation/make_affinity.hpp"
#include "flow_graph/computation/get_segmentation.hpp"
#include "flow_graph/computation/zalis.hpp"
#include "flow_graph/computation/constrain_affinity.hpp"
#include "options/options.hpp"
#include "cube/cube_io.hpp"
#include "network/parallel/nodes.hpp"

#include <zi/time.hpp>
#include <zi/zargs/zargs.hpp>

using namespace znn::v4;

namespace malis_io {

template<typename F, typename T>
inline bool write_vector( std::string const & fname, std::vector<T> vec )
{
    FILE* fvec = fopen(fname.c_str(), "w");

    STRONG_ASSERT(fvec);

    F v;

    for ( auto& elem: vec )
    {
        v = static_cast<T>(elem);
        static_cast<void>(fwrite(&v, sizeof(F), 1, fvec));
    }

    fclose(fvec);

    return true;
}

}

int main(int argc, char** argv)
{
    // ---------------------------------------------------------------
    // option parsing
    // ---------------------------------------------------------------
    options op;

    std::string fname(argv[1]);

    parse_option_file(op, fname);

    // input volume size
    vec3i s = op.require_as<ovec3i>("size");

    // I/O paths
    std::string xfname = op.require_as<std::string>("xaff");
    std::string yfname = op.require_as<std::string>("yaff");
    std::string zfname = op.require_as<std::string>("zaff");
    std::string lfname = op.require_as<std::string>("lbl");
    std::string ofname = op.require_as<std::string>("out");

    // high & low threshold
    real high = op.optional_as<real>("high","1");
    real low  = op.optional_as<real>("low","0");

    // zalis phase
    zalis_phase  phs;
    std::string sphs = op.optional_as<std::string>("phase","BOTH");
    if ( sphs == "BOTH" )
    {
        phs = zalis_phase::BOTH;
    }
    else if ( sphs == "MERGER" )
    {
        phs = zalis_phase::MERGER;
    }
    else if ( sphs == "SPLITTER" )
    {
        phs = zalis_phase::SPLITTER;
    }
    else
    {
        throw std::logic_error(HERE() + "unknown zalis_phase: " + sphs);
    }

    // affinity dimension
    size_t dim  = op.optional_as<size_t>("dim","2");

    // constrained
    bool is_constrained = op.optional_as<bool>("constrain","0");

    // fraction normalize
    bool is_frac_norm = op.optional_as<bool>("frac_norm","0");

    // debug print
    bool debug_print = op.optional_as<bool>("debug_print","0");

    // load input
    auto xaff = read<double,real>(xfname,s);
    auto yaff = read<double,real>(yfname,s);
    auto zaff = read<double,real>(zfname,s);
    auto lbl  = read<double,int>(lfname,s);

    if ( debug_print )
    {
        std::cout << "\n[xaff]\n" << *xaff << std::endl;
        std::cout << "\n[yaff]\n" << *yaff << std::endl;
        std::cout << "\n[zaff]\n" << *zaff << std::endl;
        std::cout << "\n[lbl]\n"  << *lbl << std::endl;
    }

    // ---------------------------------------------------------------
    // make affinity
    // ---------------------------------------------------------------
    zi::wall_timer wt;
    wt.reset();

    std::vector<cube_p<real>> affs;
    affs.push_back(xaff);
    affs.push_back(yaff);
    affs.push_back(zaff);

    auto true_affs = make_affinity( *lbl, dim );

    // std::cout << "\n[make_affinity] done, elapsed: "
    //           << wt.elapsed<double>() << " secs" << std::endl;

    if ( debug_print )
    {
        std::cout << "\n[affs]" << std::endl;
        for ( auto& aff: affs )
        {
            std::cout << *aff << "\n\n";
        }

        std::cout << "\n[true_affs]" << std::endl;
        for ( auto& aff: true_affs )
        {
            std::cout << *aff << "\n\n";
        }
    }

    // ---------------------------------------------------------------
    // constrain affinity
    // ---------------------------------------------------------------
    wt.reset();

    if ( is_constrained )
    {
        affs = constrain_affinity(true_affs, affs, phs);

        // std::cout << "\n[constrain_affinity] done, elapsed: "
        //           << wt.elapsed<double>() << " secs" << std::endl;

        if ( debug_print )
        {
            std::cout << "\n[constrained affs]" << std::endl;
            for ( auto& aff: affs )
            {
                std::cout << *aff << "\n\n";
            }
        }
    }

    // ---------------------------------------------------------------
    // ZALIS weight
    // ---------------------------------------------------------------
    wt.reset();

    auto weight = zalis(true_affs, affs, is_frac_norm, high, low);

    std::cout << "\n[zalis] done, elapsed: "
              << wt.elapsed<double>() << std::endl;

    if ( debug_print )
    {
        std::cout << "\n[merger weight]" << std::endl;
        for ( auto& w: weight.merger )
        {
            std::cout << *w << "\n\n";
        }

        std::cout << "\n[splitter weight]" << std::endl;
        for ( auto& w: weight.splitter )
        {
            std::cout << *w << "\n\n";
        }
    }

    // ---------------------------------------------------------------
    // save results
    // ---------------------------------------------------------------
    wt.reset();

    // write affinity
    write_tensor<double,real>(ofname + ".affs",affs);

    if ( phs == zalis_phase::BOTH )
    {
        write_tensor<double,real>(ofname + ".merger",weight.merger);
        write_tensor<double,real>(ofname + ".splitter",weight.splitter);
    }
    else if ( phs == zalis_phase::MERGER )
    {
        write_tensor<double,real>(ofname + ".merger",weight.merger);
    }
    else if ( phs == zalis_phase::SPLITTER )
    {
        write_tensor<double,real>(ofname + ".splitter",weight.splitter);
    }


#if defined( DEBUG )
    // write watershed snapshots
    write_tensor<double,int>(ofname + ".snapshots",weight.ws_snapshots);

    // write snapshot timestamp
    malis_io::write_vector<double,int>(ofname + ".snapshots_time",
                                       weight.ws_timestamp);

    // write timestamp
    write_tensor<double,int>(ofname + ".timestamp",weight.timestamp);
#endif

    std::cout << "[save]  done, elapsed: "
              << wt.elapsed<double>() << '\n' << std::endl;
}
