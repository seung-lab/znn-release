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
#pragma once

#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#include <zi/utility/singleton.hpp>
#include <zi/utility/non_copyable.hpp>
#include <zi/time.hpp>

namespace znn { namespace v4 {

class log_output_impl
{
private:
    std::mutex                                     m_;
    std::list<std::unique_ptr<std::ostringstream>> o_;
    bool                                           d_;
    std::thread                                    t_;

    void loop()
    {
        bool done = false;
        while (!done)
        {
            std::list<std::unique_ptr<std::ostringstream>> l;
            {
                std::lock_guard<std::mutex> g(m_);
                l.swap(o_);
                done = d_;
            }

            for( auto & it: l )
            {
                std::cout << it->str() << "\n";
            }

            std::cout << std::flush;

            //std::this_thread::sleep_for(std::chrono::milliseconds(200));
            usleep(200000);
        }
    }

public:
    log_output_impl()
        : m_()
        , o_()
        , d_(false)
    {
        start();
    }

    ~log_output_impl()
    {
        {
            std::lock_guard<std::mutex> g(m_);
            d_ = true;
        }
        if ( t_.joinable() )
        {
            t_.join();
        }
    }

    void start()
    {
        t_ = std::thread(&log_output_impl::loop, this);
    }

    void reg(std::unique_ptr<std::ostringstream>& o)
    {
        std::lock_guard<std::mutex> g(m_);
        o_.push_back(std::move(o));
    }
}; // class log_output_impl

namespace {

log_output_impl& log_output =
    zi::singleton<log_output_impl>::instance();

} // namespace

class log_token: zi::non_copyable
{
public:
    std::unique_ptr<std::ostringstream> i_;

    log_token()
        : i_(new std::ostringstream)
    { }

    ~log_token()
    {
        log_output.reg(i_);
    }

    template< typename T >
    log_token& operator<< ( const T& v )
    {
        (*i_) << v;
        return *this;
    }
}; // log_token


#define LOGx(what) (log_token()) << "LOG(" << #what << ") "      \
    << "[" << zi::now::usecs() << "] "

#define LOG(what) (log_token()) << zi::now::usecs() << ", "


}} // namespace znn::v4
