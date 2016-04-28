ZNN training on AWS using spot instances
===
this script can create a cluster including an on-demand master node and several spot-instance worker nodes. whenever the spot instance node got terminated by price, the script will create a new spot instance request. Thus, creating a kind of "persistance" spot worker node.

##Setup

* [install starcluster](http://star.mit.edu/cluster/docs/latest/installation.html). `easy_install StarCluster`
* download [StarCluster](https://github.com/jtriley/StarCluster) and set the StarCluster folder as a PYTHONPATH.
  * ``git clone https://github.com/jtriley/StarCluster.git``
  * put this line `export PYTHONPATH=$PYTHONPATH:"/path/to/StarCluster"` to the end of `~/.bashrc` file.
  * run `source ~/.bashrc`
  * you may need to create a new terminal (make the .bashrc file be effective)
* edit and move `config` file to `~/.starcluster/`.
  * setup the keys in `config`.
  * set the AMI and volume id.
  * setup all the parameters with a mark of `XXX`
* copy the `train_example.cfg` file as `train.cfg`
* set some additional parameters in the `train.cfg`.
    * cluster name
    * node name (Note that do not use `_`!)
    * instance type
    * biding for the spot instance
    * commands for each spot instance

##Tutorial
now, you are almost ready. 
* create a volume using starcluster (it won't work for volume created in web console!): `starcluster createvolume 50 us-east-1c`, you can get a volume ID from this step. This volume will be your persistent storage for all the training files.
* check your cluster: `starcluster listclusters`
* terminate your volume-creator cluster by: `starcluster terminate -f volumecreator`
* setup the volume id in starcluster configure file.
* launch a cluster only has the `master`: `starcluster start mycluster`
* set the `node_name` in script to choose the command you want to run. (normally, we use network name as node name)
* modify the command dict to execute training commands after the node was launched. the `node_name` is the key of command dict.

# compile and run
for the first time, you'd better update the code and compile
* run the main script: `python aws_train.py mynode`, `mynode` is the node name
* use `starcluster sshnode mycluster mynode` to login your node. 
* go to the persistent volume: `cd /home`
* clone the latest znn code: `git clone https://github.com/seung-lab/znn-release.git`
* compile the python core: `cd znn-release/python/core`; `make mkl -j 32`. all the libraries should be available.
* start training and have fun!
once setup, you don't need to compile it anymore.

## Usage
* Check your cluster: `starcluster listclusters`
* ssh: `starcluster sshmaster mycluster`
* upload: `starcluster put mycluster myfile clusterfile`
* download: `starcluster get mycluster clusterfile myfile`
* get help: `starcluster help`
