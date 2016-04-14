import sys
caffe_root = '/home/eeb-418/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

from pylab import *
import numpy as np

# define net works
from caffe import layers as L
from caffe import params as P


weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]

frozen_param = [dict(lr_mult=0)] * 2



def conv(bottom, ks, nout, stride=1, pad=0, group=1,
			  param=learned_param,
			  weight_filler=dict(type='gaussian', std=0.01),
			  bias_filler=dict(type='constant', value=0.0)):
	conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
						 num_output=nout, pad=pad, group=group,
						 param=param, weight_filler=weight_filler,
						 bias_filler=bias_filler)
	return conv
def relu(layer):
	return L.ReLU(layer, in_place=True)
def norm(layer, local_size, alpha, beta):
	return L.LRN(layer,local_size = local_size, alpha=alpha, beta=beta)
def fc(bottom, nout, param=learned_param,
			weight_filler=dict(type='gaussian', std=0.005),
			bias_filler=dict(type='constant', value=0.0)):
	fc = L.InnerProduct(bottom, num_output=nout, param=param,
						weight_filler=weight_filler,
						bias_filler=bias_filler)
	return fc
def max_pool(bottom, ks, stride=1):
	return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def customize_network(layers,net_save_path):

	n = caffe.NetSpec()
	for i in range(len(layers)):
		n.tops[layers[i]['name']] = layers[i]['layer']

	with open(net_save_path,'w') as f:
		f.write(str(n.to_proto()))

	return caffe.Net(net_save_path, caffe.TEST)


