#!/usr/bin/python
from starcluster import config
import time
import threading

#%% parameters
conf_file = "~/.starcluster/config"

# cluster name
cluster_name = 'jpcluster'

# node tag or name
node_name = 'VD2D'

# your bidding of spot instance
spot_bid = 0.71

# command
cmds = {'W5': 'cd /home/znn-release/python/; python train.py ../experiments/W5/config.cfg',\
        'N4': 'cd /home/znn-release/python/; python train.py ../experiments/N4/config.cfg',\
        'VD2D': 'cd /home/znn-release/python/; python train.py ../experiments/VD2D/config.cfg',\
        'W10':'cd /home/znn-release/python/; python train.py ../experiments/W10/config.cfg'}

# instance type
instance_type = 'c4.8xlarge'

# if there are several cluster template in config file, you have to set the cluster id to a specific cluster template
cluster_id = 0

# sleep interval (secs)
sleep_interval = 1 * 60

#%% configuration
cfg = config.get_config( conf_file )
cl = cfg.get_clusters()[ cluster_id ]
cl.spot_bid = spot_bid
cl.cluster_tag = cluster_name
cl.node_instance_type = instance_type

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
        cl.wait_for_cluster(msg='Waiting for cluster to come up...')

def node_search(cl, node_name):
    for node in cl.nodes:
        if node.alias == node_name:
            return node
    return None

#%% start the cluster
print "constantly check whether this cluster is stopped or terminated."
cid = 0
f = open('log.txt','a+')
f.write( "try to start a cluster with id: {}\n".format( cid ) )
while True:
    # if cluster not started start the cluster
    if (not cl.nodes) or cl.is_cluster_stopped() or cl.is_cluster_terminated():
        cid = cid + 1

        print "try to start a cluster with id: {}\n".format( cid )
        time.sleep(1)

        # run the start in a separate thread
        try:
            threadRun = ThreadRun(cl)
            print "clulster creater thread running..."
            # wait for the volume mounted
            print "wait for the volume to attach..."
            vol_id = cl.volumes['data']['volume_id']
            volume = cl.ec2.get_volume( vol_id )
            cl.ec2.wait_for_volume( volume, state='attached' )
        except:
            print "running failed"
            time.sleep(1)
            pass
    # if node not started, start the node
    hasnode = False
    mynode = node_search(cl, node_name)
    if mynode == None:
        try:
            mynode = cl.add_node( alias=node_name, spot_bid=spot_bid )
        except:
            print "node creation failed"
            continue
        print "run plugin"
	mynode = node_search(cl, node_name)
        mynode.ssh.execute( cmds[node_name] )

    f.write('wait for cluster...\n')
    time.sleep(1)

    # sleep for a while
    print "node is running, wait for {} secs to check.".format( sleep_interval )
    time.sleep( sleep_interval )

f.close()
