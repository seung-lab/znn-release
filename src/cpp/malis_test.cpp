#include "cost_fn/malis.hpp"
#include <assert.h>
#include "H5Cpp.h"

using namespace H5;
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
        size_ = size<real>( *in_image );
        assert( size<real>(*in_image) == size<real>(*in_label) );
        // 2D image
        assert( size_[0]==1 );
        return;
    }

    CMalis_test(std::string img_name,
    			std::string lbl_name)
    {
    	// only support hdf5 now
    	assert(	img_name.find("h5"  ) != std::string::npos or
    			img_name.find("hdf5") != std::string::npos);
    	assert( lbl_name.find("h5"  ) != std::string::npos or
    			lbl_name.find("hdf5") != std::string::npos);

    }

    inline void _mark_line(cube_p<real> mat, const int row, const real val, const int col=0)
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

	CMalis_test()
	{
		std::long_t s=10;
		// create the image
		image = get_cube<real>(s,s,1);
		for (std::size_t x=0; x<s; x++)
			for (std::size_t y=0; y<s; y++)
				(*image)[x][y][0] = 1;
		this->_mark_line(image, 3, 0.5);
		this->_mark_line(image, 3, 0.8, 7);
		this->_mark_line(image, 3, 0.2, 3);
		this->_mark_line(image, 6, 0.5);
		this->_mark_line(image, 6, 0.2, 3);
		this->_mark_line(image, 6, 0.8, 7);

		// create label
		label = get_cube<real>(s,s,1);
		for (std::size_t x=0; x<6; x++)
			this->_mark_line(label, x, 1);
		this->_mark_line(label, 6, 0);
		for (std::size_t x = 7; x<s; x++)
			this->_mark_line(label, x, 2);
		return;
	}

};


int main(int argc, char **argv)
{
	CMalis_test malis_test();

}
