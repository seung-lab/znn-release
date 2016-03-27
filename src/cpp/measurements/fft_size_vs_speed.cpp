//
// Copyright (C) 2012-2015  Aleksandar Zlateski <zlateski@mit.edu>
// ---------------------------------------------------------------
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
#include "network/parallel/network.hpp"

using namespace znn::v4;

int main(int argc, char** argv)
{
    int64_t n0 = 64;
    int64_t n1 = 256;

    if ( argc >= 3 )
    {
        n0 = atoi(argv[1]);
        n1 = atoi(argv[2]);
    }

    size_t rounds = 30;
    if ( argc >= 4 )
    {
        rounds = atoi(argv[3]);
    }


    // for ( int64_t s = n0; s <= n1; ++s )
    // {
    //     vec3i ss(s,s,s);

    //     uniform_init init(1);
    //     auto v = get_cube<real>(ss);
    //     init.initialize(v);

    //     fftw::transformer fft(vec3i(s,s,s));
    //     zi::wall_timer wt;

    //     vec3i sparse(2,2,2);

    //     wt.reset();
    //     for ( int j = 0; j < rounds; ++j )
    //     {
    //         auto t = fft.forward(std::move(v));
    //         v = fft.backward(std::move(t));
    //         *v /= s * s * s;
    //         //auto t3 = sparse_implode_slow(*t2, sparse, k);
    //     }

    //     double t = wt.elapsed<double>();
    //     double per_pix = t / s / s / s / rounds;
    //     std::cout << s << ' ' << per_pix << '\n';
    //     double bps = s * s * s * 4 * rounds / t / 1024 / 1024;
    //     std::cout << "    " << bps << " Mbps\n";



    //     auto s2 = ss;
    //     s2[2] += ss[2] % 4;
    //     //s2[1] += ss[1] % 2;

    //     fft_image_plan fftfp( ss, s2 );

    //     init.initialize(v);
    //     auto vc = get_copy(*v);
    //     wt.reset();
    //     for ( int j = 0; j < rounds; ++j )
    //     {
    //         auto t1 = fftfp.forward(*v);
    //         v = fftfp.backward(*t1);
    //         *v /= s2[0] * s2[1] * s2[2];
    //     }

    //     t = wt.elapsed<double>();
    //     per_pix = t / s / s / s / rounds;
    //     std::cout << " -- " << s << ' ' << per_pix << '\n';
    //     double bps2 = s * s * s * 4 * rounds / t / 1024 / 1024;
    //     std::cout << "    " << bps2 << " Mbps\n";

    //     std::cout << "SPEEDUP: ------> " << ( bps2 / bps ) << "\n\n";

    // }

    // return 0;



    vec3i k(5,5,5);

    for ( int64_t s = n0; s <= n1; ++s )
    {
        vec3i ss(s,s,s);

        uniform_init init(1);
        auto v = get_cube<real>(k);
        init.initialize(v);

        fftw::transformer fft(vec3i(s,s,s));
        zi::wall_timer wt;

        vec3i sparse(2,2,2);

        wt.reset();
        for ( int j = 0; j < rounds; ++j )
        {
            auto x = sparse_explode_slow(*v, sparse, ss);
            auto t = fft.forward(std::move(x));
            //auto t2 = fft.backward(std::move(t));
            //auto t3 = sparse_implode_slow(*t2, sparse, k);
        }

        double t = wt.elapsed<double>();
        double per_pix = t / s / s / s / rounds;
        std::cout << s << ' ' << per_pix << '\n';
        double bps = s * s * s * 4 * rounds / t / 1024 / 1024;
        std::cout << "    " << bps << " Mbps\n";



        ss[2] += ss[2] % 4;
        //ss[1] += ss[1] % 2;

        fft_filter_plan fftfp( k, sparse, ss );

        wt.reset();
        for ( int j = 0; j < rounds; ++j )
        {
            auto t1 = fftfp.forward(*v);
            //auto t2 = fftfp.backward(*t1);
        }

        t = wt.elapsed<double>();
        per_pix = t / s / s / s / rounds;
        std::cout << " -- " << s << ' ' << per_pix << '\n';
        double bps2 = s * s * s * 4 * rounds / t / 1024 / 1024;
        std::cout << "    " << bps2 << " Mbps\n";

        std::cout << "SPEEDUP: ------> " << ( bps2 / bps ) << "\n\n";

    }

    return 0;


    {
        vec3i k(3,3,3);
        vec3i s(5,5,5);

        uniform_init init(1);
        auto v = get_cube<real>(k);
        init.initialize(v);

        fft_filter_plan fftfp( k, vec3i::one, s );

        std::cout << *v << "\n\n\n";

        auto t1 = fftfp.forward(*v);

        std::cout << *t1 << "\n\n\n";

        auto t2 = fftfp.backward(*t1);

        *t2 /= 125;

        std::cout << *t2 << "\n\n\n";

    }
    return 0;


    for ( int64_t s = n0; s <= n1; ++s )
    {
        uniform_init init(1);
        auto v = get_cube<real>(vec3i(s,s,s));
        init.initialize(v);

        fftw::transformer fft(vec3i(s,s,s));
        zi::wall_timer wt;

        wt.reset();
        for ( int j = 0; j < rounds; ++j )
        {
            auto t = fft.forward(std::move(v));
            //v = fft.backward(std::move(t));
        }

        double t = wt.elapsed<double>();
        double per_pix = t / s / s / s / rounds;
        std::cout << s << ' ' << per_pix << '\n';
        double bps = s * s * s * 4 * rounds / t / 1024 / 1024;
        std::cout << "    " << bps << " Mbps\n";

        int64_t s2 = s + s % 2;
        fftw::transformer fft2(vec3i(s2,s2,s2));

        wt.reset();
        for ( int j = 0; j < rounds; ++j )
        {
            auto v2 = pad_zeros(*v, vec3i(s2,s2,s2));
            auto t = fft2.forward(std::move(v2));
            //v = fft.backward(std::move(t));
        }

        t = wt.elapsed<double>();
        per_pix = t / s / s / s / rounds;
        std::cout << s << ' ' << per_pix << '\n';
        bps = s * s * s * 4 * rounds / t / 1024 / 1024;
        std::cout << "    " << bps << " Mbps\n";


    }

    // generate
}
