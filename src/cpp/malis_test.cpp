#include "cost_fn/malis.hpp"
#include <assert.h>

using namespace znn::v4;

class CMalis_test
{
private:
    cube_p<real> image;
    cube_p<real> label;

    vec3i size_;

public:
    // initialize this instance
    CMalis_test(cube_p<real> in_image,
                cube_p<real> in_label)
        : image(in_image)
        , label(in_label)
    {
        size_ = size<real>( in_image );
        assert( size<real>(in_image) == size<int>(in_label) );
        // 2D image
        assert( size_[0]==1 );
    }

    template<typename T>
    inline void _mark_line(cube_p<T> mat, int row, T val, int col=0)
    {
    	if(col==0)
    	{   // erase the whole line
            for (std::size_t i; i<size_[1]; i++)
                (*mat)[row][i][0] = val;
    	}
    	else
    	{
            // change one value
            (*mat)[row][col][0] = val;
    	}
    	return;
    }

	template<typename T>
	inline CMalis_test()
	{
            std::size_t s=10;
            // create the image
            image = get_cube<T>(s,s,1);
            for (std::size_t x=0; x<s; x++)
                for (std::size_t y=0; y<s; y++)
                    (*image)[x][y][0] = 1;
            _make_line(image, 3, 0.5);
            _make_line(image, 3, 0.8, 7);
            _make_line(image, 3, 0.2, 3);
            _make_line(image, 6, 0.5);
            _make_line(image, 6, 0.2, 3);
            _make_line(image, 6, 0.8, 7);

            // create label
            label = get_cube<T>(s,s,1);
            for (std::size_t x=0; x<6; x++)
                _make_line(label, x, 1);
            _make_line(label, 6, 0);
            for (std::size_t x = 7; x<s; x++)
                _make_line(label, x, 2);

            return;
	}

};




void main(int argc, char **argv)
{

}
