#!/usr/bin/python
from starcluster import config
import time
import threading
import ConfigParser

#%% a thread to run
class ThreadRun(object):
    def __init__(self, cl):
        self.cl = cl
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True                            # Daemonize thread
        thread.start()                                  # Start the execution
    def run(self):
        """ Method that runs forever """
        self.cl.start()
        self.cl.wait_for_cluster(msg='Waiting for cluster to come up...')

def node_search(cl, node_name):
    for node in cl.nodes:
        if node.alias == node_name:
            return node
    return None


def main(sec, train_cfg='train.cfg', sc_cfg='~/.starcluster/config'):
    """
    parameters
    ----------
    sec: string, section name and is also the node name
    train_cfg: configuration file name of training
    sc_cfg : starcluster configuration file name
    """

    #%% parameters
    tncfg = ConfigParser.ConfigParser()
    tncfg.read( train_cfg )
    # cluster name
    cluster_name = tncfg.get(sec, 'cluster_name')

    # node tag or name
    node_name = sec

    # your bidding of spot instance
    spot_bid = tncfg.getfloat(sec, 'spot_bid')

    # tag user and project name
    user = tncfg.get(sec, 'User')
    prj = tncfg.get(sec, 'Project')

    # command
    command = tncfg.get(sec, 'command')

    # instance type
    instance_type = tncfg.get(sec, 'instance_type')

    # sleep interval (secs)
    if tncfg.has_option(sec, 'node_check_interval'):
        node_check_interval = tncfg.getint(sec, 'node_check_interval')
    else:
        node_check_interval = 10 * 60

    # if there are several cluster template in config file,
    # you have to set the cluster id to a specific cluster template
    cluster_id = 0


    #%% configuration
    cfg = config.get_config( sc_cfg )
    cl = cfg.get_clusters()[ cluster_id ]
    cl.spot_bid = spot_bid
    cl.cluster_tag = cluster_name
    cl.node_instance_type = instance_type


    #%% start the cluster
    print "constantly check whether this cluster is stopped or terminated."
    cid=0
    f = open('log.txt','a+')
    f.write( "try to start a cluster with id: {}\n".format( cid ) )
    while True:
        # if cluster not started start the cluster
        if (not cl.nodes) or cl.is_cluster_stopped() or cl.is_cluster_terminated():
            cid += 1
            print "try to start a cluster with id: {}\n".format( cid )
            time.sleep(1)
            # run the start in a separate thread
            try:
                ThreadRun(cl)
                print "clulster creater thread running..."
                # wait for the volume mounted
                print "wait for the volume to attach..."
                vol_id = cl.volumes['data']['volume_id']
                volume = cl.ec2.get_volume( vol_id )
                cl.ec2.wait_for_volume( volume, state='attached' )
                time.sleep(3*60)
            except:
                print "running failed"
                time.sleep(1)
                pass

        # if node not started, start the node
        mynode = node_search(cl, node_name)
        if mynode is None:
            try:
                print "add node ", node_name, " with a bid of $", spot_bid
                cl.add_node( alias=node_name, spot_bid=spot_bid )
            except:
                print "node creation failed."
                print "please check the starcluster config options, such as subnet."
                continue
            print "wait for the launch of node {} ...".format(node_name)
            cl.ec2.wait_for_propagation( spot_requests=mynode )
            cl.wait_for_ssh()
            cl.wait_for_cluster(msg="Waiting for node(s) to come up...")
            time.sleep( 1*60 )

            mynode = node_search(cl, node_name)
            # tag the user and project name
            print "add tags: User--",user,", Project--",prj
            mynode.add_tag("User", user)
            mynode.add_tag("Project", prj)

            try:
                print "run command after node launch."
                mynode.ssh.execute( command )
            except:
                print "command execution failed!"

        f.write('wait for cluster...\n')
        # sleep for a while
        print "node {} is running, wait for {} secs to check.".format( node_name, node_check_interval )
        time.sleep( node_check_interval )

    f.close()

if __name__ == '__main__':
    print """
    usage:
    python aws_train.py node_name train_config starcluster_config
    default parameters:
    train_config: train.cfg
    starcluster_config: ~/.starcluster/config

    example:
    python aws_train.py N4
    """
    from sys import argv
    if len(argv)==1:
        raise NameError( "please specify the node name!")
    elif len(argv)==2:
        main(argv[1])
    elif len(argv)==3:
        main(argv[1], argv[2])
    elif len(argv)==4:
        main(argv[1], argv[2], argv[3])
    else:
        print "too many parameters, please check usage!"
