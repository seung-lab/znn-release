import numpy as np
import pyznn

vin = np.random.rand(100,100,100)

# print vin
outsz = np.asarray([1,1,1])
print "output volume size: {}x{}x{}".format(outsz[0], outsz[1], outsz[2])
net = pyznn.CNet('/usr/people/jingpeng/seungmount/research/Jingpeng/01_ZNN/znn-v4python/networks/N4.znn',\
                          outsz[0],outsz[1],outsz[2],2)

fov = np.asarray(net.get_fov())

print "field of view: {}x{}x{}".format(fov[0],fov[1], fov[2])

insz = fov + outsz - 1
vin = vin[:insz[0], :insz[1], :insz[2]]
print "input volume shape: {}x{}x{}".format(vin.shape[0], vin.shape[1], vin.shape[2])
vout = net.forward(vin)
print "successfully returned volume"
#print vout
