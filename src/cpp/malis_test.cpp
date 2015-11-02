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
    	{	// erase the whole line
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
		//

		label = get_cube<T>(s,s,1);

		return;
	}

};




void main(int argc, char **argv)
{

}
