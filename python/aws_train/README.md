ZNN training on AWS using spot instances
===
this script can create a cluster including an on-demand master node and several spot-instance worker nodes. whenever the spot instance node got terminated by price, the script will create a new spot instance request. Thus, creating a kind of "persistance" spot worker node.

##Setup

* [install starcluster](http://star.mit.edu/cluster/docs/latest/installation.html). `easy_install StarCluster`
* download [StarCluster](https://github.com/jtriley/StarCluster) and set the StarCluster folder as a PYTHONPATH.
  * ``git clone https://github.com/jtriley/StarCluster.git``
  * put this line `export PYTHONPATH=$PYTHONPATH:"/path/to/StarCluster"` to the end of `~/.bashrc` file.
  * run `source ~/.bashrc`
* edit and move `config` file to `~/.starcluster/`.
  * setup the keys in `config`.
  * set the AMI and volume id.
  * setup all the parameters with a mark of `XXX`
* set some additional parameters in the script.
        * cluster name
        * instance type
        * biding for the spot instance
        * commands for each spot instance


##Tutorial
now, you are almost ready.
* set the `node_name` in script to choose the command you want to run. 
* run the main script: `python persistent_spot_cluster.py`
