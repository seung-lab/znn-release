#pragma once

#include <zi/assert.hpp>
#include <string>
#include <sstream>


#define UNIMPLEMENTED()                                                 \
    {                                                                   \
        std::cout << "unimplemented function" << std::endl              \
                  << "file: " << __FILE__ << " line: "                  \
                  << __LINE__ << std::endl;                             \
        abort();                                                        \
    }                                                                   \
    static_cast<void>(0)


template<typename F, typename L>
inline std::string ___this_file_this_line( const F& f, const L& l)
{
    std::ostringstream oss;
    oss << "\nfile: " << f << "\nline: " << l << "\n";
    return oss.str();
}

#define HERE() ___this_file_this_line(__FILE__,__LINE__)
