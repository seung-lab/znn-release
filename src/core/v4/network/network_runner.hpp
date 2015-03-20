#pragma once

namespace znn { namespace v4 {

template<typename Network>
class network_runner
{
private:
    typedef Network                          network_type;
    typedef typename network_type::data_type data_type   ;

private:
    network_type& network_;

public:
    network_runner(network_type& network): network_(network_) {}
    ~network_runner() { network_.zap(); }

    data_type forward(const data_type& f)
    {
        return network_.forward(f);
    }

    void backward(const data_type& g)
    {
        return network_.backward(g);
    }

    void zap()
    {
        network_.zap();
    }

}; // class network_runner

}} // namespace znn::v4
