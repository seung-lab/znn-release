#ifndef ZNN_PROFILER_HPP_INCLUDED
#define ZNN_PROFILER_HPP_INCLUDED

#include <zi/concurrency.hpp>
#include <zi/utility/for_each.hpp>
#include <zi/utility/singleton.hpp>
#include <zi/time.hpp>
#include <map>
#include <utility>
#include <string>
#include <iostream>

namespace zi {
namespace znn {
namespace profiler {

class performance_profiler_impl
{
private:
    zi::mutex                     m_  ;
    std::map<std::pair<std::string,int>, std::pair<double,int> > map_;

public:
    void report(const std::string& s, int l, double t)
    {
        zi::mutex::guard g(m_);
        map_[std::pair<std::string,int>(s,l)].first += t;
        ++map_[std::pair<std::string,int>(s,l)].second;
    }

    void report()
    {
        zi::mutex::guard g(m_);
        FOR_EACH( it, map_ )
        {
            std::cout << it->first.first << ":" << it->first.second
                      << ": " << it->second.first << " ("
                      << it->second.second << ")" << '\n';
        }
    }

}; // class performance_profiler_impl

namespace {
performance_profiler_impl& performance_profiler =
    zi::singleton<performance_profiler_impl>::instance();
} // anonymous namespace

class performance_profiler_reporter
{
private:
    zi::wall_timer wt_;
    const char*    fn_;
    int            ln_;

public:
    performance_profiler_reporter(const char* fn, int ln)
        : wt_()
        , fn_(fn)
        , ln_(ln)
    {
        wt_.reset();
    }

    ~performance_profiler_reporter()
    {
        performance_profiler.report(fn_, ln_, wt_.elapsed<double>());
    }
};

}}} // namespace zi::znn

#ifdef DO_NOT_DEBUG

#  define PROFILE_FUNCTION()                    \
    static_cast<void>(0)

#else

#  define PROFILE_FUNCTION()                    \
    ::zi::znn::profiler::performance_profiler_reporter \
    _______p______(__FUNCTION__,__LINE__)              \

#endif

#endif // ZNN_PROFILER_HPP_INCLUDED
