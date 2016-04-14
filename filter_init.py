from os import environ
environ['GLOG_minloglevel'] = '2'
import sys
caffe_root = '/home/eeb-418/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

from argparse import ArgumentParser
import numpy as np
import random
import matplotlib.pyplot as plt
import os

import customize_network as cn

RECEPTIVE_FIELD = {'conv1':11,'conv2':51,'conv3':99,'conv4':131,'conv5':163}


def vis_square(data, outputpath):
	#Take an array of shape (n, height, width) or (n, height, width, 3)
	#and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
	# normalize data for display
	plot_data = data.copy()
	plot_data = (plot_data - plot_data.min(axis=(1,2,3),keepdims=True)) / (plot_data.max(axis=(1,2,3),keepdims=True) - plot_data.min(axis=(1,2,3),keepdims=True)+10e-6)
	plot_data = plot_data[:,:,:,::-1]
	# force the number of filters to be square
	n = int(np.ceil(np.sqrt(plot_data.shape[0])))
	padding = (((0, n ** 2 - plot_data.shape[0]),
				(0, 1), (0, 1))                 # add some space between filters
			   + ((0, 0),) * (plot_data.ndim - 3))  # don't pad the last dimension (if there is one)
	plot_data = np.pad(plot_data, padding, mode='constant', constant_values=1)  # pad with ones (white)

	# tile the filters into an image
	plot_data = plot_data.reshape((n, n) + plot_data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, plot_data.ndim + 1)))
	plot_data = plot_data.reshape((n * plot_data.shape[1], n * plot_data.shape[3]) + plot_data.shape[4:])


	if plot_data.ndim == 3:
		fig = plt.imshow(plot_data,interpolation = 'nearest')
	else:
		fig = plt.imshow(plot_data,cmap = plt.get_cmap('gray'),interpolation = 'nearest')
	plt.axis('off')
	fig.axes.get_xaxis().set_visible(False)
	fig.axes.get_yaxis().set_visible(False)
	plt.savefig(outputpath, dpi=200, bbox_inches='tight',pad_inches=0)
	plt.clf()
	plt.close()

def forward(net, layer_from, layer_to, NIT, layer_from_data):
	layer_from_id = -1
	layer_to_id = -1
	for i,n in enumerate(net._layer_names):
		if layer_from == n: layer_from_id = i
		if layer_to == n: layer_to_id = i
	if layer_from_id==-1 or layer_to_id==-1:
		print('Cannot find both the two layers: %s and %s'%(layer_from,layer_to))
		return None

	output = [None]*NIT
	label = [None]*NIT
	for it in range(NIT):
		if layer_to !='data' or len(layer_from_data)>0:
			net.blobs[layer_from].data[...] = layer_from_data[it]
			net._forward(layer_from_id+1, layer_to_id)
		else:
			net._forward(layer_from_id, layer_to_id)


		output[it] = 1*net.blobs[layer_to].data
		label[it] = 1*net.blobs['label'].data

	return output, label

def collect_training_input(imgs,labs,filter_size,sample_per_img):

	D = []
	L = []
	half_win_size = np.floor(filter_size/2)
	input_size = imgs[0].shape[2]
	stride = 5
	indexs = np.arange(half_win_size, input_size-half_win_size, stride)

	random.seed(10)
	import copy
	for bid,b in enumerate(imgs):
		for item in range(b.shape[0]):
			for iter in range(sample_per_img*2):
				idxi = random.randint(0,len(indexs)-1)
				idxj = random.randint(0,len(indexs)-1)
				i = indexs[idxi]
				j = indexs[idxj]

				patch = b[item,:,i-half_win_size:i+half_win_size+1,j-half_win_size:j+half_win_size+1].copy()
				patch = patch[None,:]
				D.append(patch)
				L.append(labs[bid][item].reshape((1,1)))
	D = np.concatenate(D,axis=0)
	L = np.concatenate(L,axis=0)

	D_nomean = D-D.mean(axis=(2,3),keepdims=True)
	stds = np.std(D_nomean,axis=(1,2,3))
	id = np.argsort(stds)[::-1]

	output = D[id[0:int(np.floor(id.shape[0]*0.5))],:]
	label = L[id[0:int(np.floor(id.shape[0]*0.5))],:]
	print('Got %d patches with size %d x %d'%(output.shape[0],filter_size,filter_size))

	output = np.split(output,len(imgs),axis=0)
	label = np.split(label,len(imgs),axis=0)
	return output,label

def train_dict(D, N_OUT,type):

	wD = np.concatenate([b for b in D],axis=0)
	wD = wD.reshape((wD.shape[0],-1))

	wD -= np.mean(wD,axis=0,keepdims=True)
	wD /= np.std(wD,axis=0,keepdims=True)


	if type == "kmeans_wtdc" or type=="kmeans_wodc":
		from sklearn.cluster import MiniBatchKMeans
		km = MiniBatchKMeans(init='k-means++', n_clusters = N_OUT, batch_size=10*N_OUT,random_state=np.random.RandomState(0)).fit(wD).cluster_centers_
	elif type =="rand":
		km = np.random.normal(0,1,(N_OUT,wD.shape[1]))
		km -= np.mean(wD,axis=0,keepdims=True)
		# km = wD[np.random.choice(wD.shape[0], N_OUT,replace=False),:]
	elif type =="pca":
		print("")
	C = km.reshape((N_OUT,)+(D[0].shape[1:]))
	return C

def visualize_filter_response_map(net, layer_from, layer_to, NIT, layer_from_data, output_prefix):
	net_dict = {}
	for i,lname in enumerate(net._layer_names):
		net_dict[lname] = i
	layer_to_data = forward(net, layer_from, layer_to, NIT, layer_from_data)[0]
	filter_bank = net.layers[net_dict[layer_to]].blobs[0].data.copy()

	cnt = 0
	for i,b_from in enumerate(layer_from_data):
		b_to = layer_to_data[i]
		for img_id in range(5):#b_from.shape[0]):
			sys.stdout.write("output response for image %d:"%(cnt))
			folder = "%s%s_response_map_%d/%d/"%(output_prefix,layer_to,filter_bank.shape[0],cnt)
			cnt +=1
			if not os.path.isdir(folder):
				os.makedirs(folder)
			n = int(np.ceil(np.sqrt(filter_bank.shape[1])))
			filter_bank = (filter_bank-filter_bank.min(axis=(1,2,3),keepdims = True))/(filter_bank.max(axis=(1,2,3),keepdims = True)-filter_bank.min(axis=(1,2,3),keepdims = True)+1e-10)
			for filter_id in range(filter_bank.shape[0]):
				# sys.stdout.write("%d."%filter_id)
				fname = "%s%d.png"%(folder,filter_id)
				input = b_from[img_id,:].transpose((1,2,0)).copy()[:,:,::-1]
				filter = filter_bank[filter_id,:].transpose((1,2,0)).copy()
				output = np.expand_dims(b_to[img_id,filter_id,:],axis = 2).copy()

				input = (input-input.min())/(input.max()-input.min()+1e-10)
				output= (output-output.min())/(output.max()-output.min()+1e-10)

				if layer_to != 'conv1':
					padding = ((0, 1),(0, 1), (0, n ** 2-filter.shape[2]))
					filter_map = np.pad(filter, padding, mode='constant', constant_values=1)  # pad with ones (white)
					# tile the filters into an image
					filter_map = filter_map.reshape(filter_map.shape[:-1]+(n, n)).transpose((2, 0, 3, 1))
					filter_map = filter_map.reshape((n * filter_map.shape[1], n * filter_map.shape[3],1))
					filter_map = np.tile(filter_map,[1,1,3])
				else:
					filter_map = filter[:,:,::-1]

				output[0:output.shape[0]:4,0:output.shape[1]:4] = 0
				output_map = np.tile(output,[1,1,3])
				output_map[0:output_map.shape[0]:4,0:output_map.shape[1]:4,0] = 1

				max_height = np.max([input.shape[0],filter_map.shape[0],output_map.shape[0]])

				input = np.pad(input, ((0,max_height-input.shape[0]),(0,10),(0,0)), mode='constant', constant_values=1)
				filter_map = np.pad(filter_map, ((0,max_height-filter_map.shape[0]),(0,10),(0,0)), mode='constant', constant_values=1)
				output_map = np.pad(output_map, ((0,max_height-output_map.shape[0]),(0,10),(0,0)), mode='constant', constant_values=1)

				img_show = np.concatenate((input,filter_map,output_map),axis=1)

				fig = plt.imshow(img_show,interpolation = 'nearest')
				plt.axis('off')

				fig.axes.get_xaxis().set_visible(False)
				fig.axes.get_yaxis().set_visible(False)
				plt.savefig(fname, dpi=200, bbox_inches='tight',pad_inches=0)
				plt.clf()
			print("Done")
	plt.close("All")

def visualize_filter_response_sparsity(net, layer_from, layer_to, NIT, layer_from_data, output_prefix):
	net_dict = {}
	for i,lname in enumerate(net._layer_names):
		net_dict[lname] = i
	layer_to_data = forward(net, layer_from, layer_to, NIT, layer_from_data)[0]

	filter_bank = net.layers[net_dict[layer_to]].blobs[0].data.copy()
	filter_bank = (filter_bank-filter_bank.min(axis=(1,2,3),keepdims = True))/(filter_bank.max(axis=(1,2,3),keepdims = True)-filter_bank.min(axis=(1,2,3),keepdims = True)+1e-10)
	padding = ((0,0),(0,0),(0,0),(0,1))
	filter_width = filter_bank.shape[2]
	filter_map = np.pad(filter_bank, padding, mode='constant', constant_values=1)  # pad with ones (white)
	# tile the filters into an image
	# filter_width = filter_map.shape[2]
	filter_map = filter_map.transpose((2,0,3,1))
	filter_map = filter_map.reshape((filter_map.shape[0], filter_map.shape[1]*filter_map.shape[2],filter_map.shape[3]))

	max_height = 35

	folder = "%s%s_response_hist_%d/"%(output_prefix,layer_to,filter_bank.shape[0])
	if not os.path.isdir(folder):
		os.makedirs(folder)
	cnt = 0
	for i,b_from in enumerate(layer_from_data):
		b_to = layer_to_data[i]
		for img_id in range(25):#b_from.shape[0]):
			sys.stdout.write("output response for image %d:"%(cnt))
			response = b_to[img_id,:].copy()
			response = np.abs(response.reshape((response.shape[0])))
			response = (response - response.min())/(response.max() - response.min()+1e-10)

			hist = np.ones((response.shape[0],max_height,filter_width,1))
			hist = np.tile(hist,[1,1,1,3])
			for id, r in enumerate(response):
				hist[id,np.floor((1-r)*(max_height-1)):,:,1] = 0
				hist[id,np.floor((1-r)*(max_height-1)):,:,2] = 0


			padding = ((0,0),(0,1),(0,1),(0,0))
			hist = np.pad(hist, padding, mode='constant', constant_values=1)  # pad with ones (white)

			hist = hist.transpose((1,0,2,3))
			hist = hist.reshape((hist.shape[0],hist.shape[1]*hist.shape[2],hist.shape[3]))

			hist = np.concatenate([hist,filter_map],axis=0)

			ori = b_from[img_id,:].copy().transpose((1,2,0))
			ori = (ori-np.min(ori,axis = (0,1),keepdims = True))/(np.max(ori,axis = (0,1),keepdims = True)-np.min(ori,axis = (0,1),keepdims = True))
			pad_top = np.floor((hist.shape[0]-ori.shape[0])/2.0)
			pad_bot = hist.shape[0]-ori.shape[0]-pad_top
			padding = ((pad_top,pad_bot),(0,5),(0,0))
			ori = np.pad(ori, padding, mode='constant', constant_values=1)  # pad with ones (white)

			out = np.concatenate([ori,hist],axis=1)

			fname = "%s%d.png"%(folder,cnt)

			fig = plt.imshow(out,interpolation = 'nearest')
			plt.axis('off')
			fig.axes.get_xaxis().set_visible(False)
			fig.axes.get_yaxis().set_visible(False)
			plt.savefig(fname, dpi=200, bbox_inches='tight',pad_inches=0)
			plt.clf()
			print("Done!")
			cnt+=1
	plt.close('All')

def visualize_discriminate_power(net,layer_from, layer_to, NIT, layer_from_data,layer_from_lab,output_prefix):
	net_dict = {}
	for i,lname in enumerate(net._layer_names):
		net_dict[lname] = i
	layer_to_data = forward(net, layer_from, layer_to, NIT, layer_from_data)[0]
	layer_to_data = np.concatenate(layer_to_data,axis=0).copy()
	layer_to_data = layer_to_data.reshape(layer_to_data.shape[:2])
	layer_to_lab = np.concatenate(layer_from_lab,axis=0)
	layer_from_data = np.concatenate(layer_from_data,axis=0)

	u = np.unique(layer_to_lab)

	data_portion = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

	# discriminate powers
	dps = np.zeros(len(data_portion))
	for i,p in enumerate(data_portion):
		mus = []
		inner_dis = 0
		for l in u:
			id = np.nonzero(layer_to_lab==l)[0]
			data = layer_to_data[id,:].copy()
			energy =np.sum(data**2,axis=1)
			id_rank = np.argsort(energy)[::-1]
			id_select = id_rank[:int(np.floor((1-p)*data.shape[0]))]

			data = data[id_select,:]

			vis_square(layer_from_data[id_select,:].transpose((0,2,3,1)),"%sdp_%03.1f_c%d.png"%(output_prefix,p,l))
			mu = data.mean(axis=0,keepdims=True)
			mus.append(mu)
			data -= mu
			data = np.sum(data**2,axis=1)
			inner_dis += np.mean(data)

		mus = np.concatenate(mus,axis=0)
		inter_dis = 2*np.sum(mus**2)-np.sum(mus.dot(mus.T))

		dps[i] = inter_dis/(inner_dis+1e-10)

	output_path = output_prefix+"dp_plot.png"
	plt.figure()
	plt.plot(np.asarray(data_portion)*100, dps, color = [1,0,0],marker = "o", linewidth = 2, alpha = 0.5)
	plt.title("Discriminate power vs. top x percentage of largest energy response")
	plt.xlabel("Percentage of largest energy response used")
	plt.ylabel("Discriminate power (inter class distance/ inner class distance)")
	# fig.axes.get_xaxis().set_visible(False)
	# fig.axes.get_yaxis().set_visible(False)
	plt.savefig(output_path, dpi=200, bbox_inches='tight',pad_inches=0)
	plt.clf()
	plt.close()

def rescale_weights(net,layer_from, layer_to, NIT, layer_from_data):
	net_dict = {}
	for i,lname in enumerate(net._layer_names):
		net_dict[lname] = i
	layer_to_data = forward(net, layer_from, layer_to, NIT, layer_from_data)[0]

	layer_to_data = np.concatenate(layer_to_data,axis=0).transpose((1,0,2,3))
	layer_to_data = layer_to_data.reshape((layer_to_data.shape[0],-1))


	filter_bank = net.layers[net_dict[layer_to]].blobs[0].data.copy()
	filter_bank /= layer_to_data.std(axis = 1,keepdims=True).reshape(filter_bank.shape[0],1,1,1)

	return filter_bank


#----------------------------------------------------------------------------------------------------------------------#
def main():
	parser = ArgumentParser()
	parser.add_argument('-d', '--data',
						default='/home/eeb-418/hdd/Chen/trainvalsplit_places205/codebook_places_bedroom_coast_200_wtlabel.txt',
						help='Image list to use [default prototxt data]')
	parser.add_argument('-bs', '--batch_size',type=int, default=100, help='Batch size [only custom data "-d"]')
	parser.add_argument('-t', '--type', default='rand',
						help='ways to train dictionary: kmeans_wtdc, kmeans_wodc, rand, sparse, pca')
	parser.add_argument('-nit', '--NIT', type=int, default=2, help='Number of iterations')
	parser.add_argument('--gpu', type=int, default=0, help='What gpu to run it on?')
	parser.add_argument('-mfn','--mean_file_name',
						default='/home/eeb-418/Documents/Chen/data/lmdb/places_bedroom_coast_mean.binaryproto',
						help='path to mean_files')
	args = parser.parse_args()


	caffe.set_mode_gpu()
	if args.gpu is not None:
		caffe.set_device(args.gpu)

	filter_num = 96
	result_folder = './result/%s_%s/'%(args.type,filter_num)
	if not os.path.isdir(result_folder):
		os.makedirs(result_folder)
	#------------------------------------------------------------------------------------------------------------------#
	# Step1: Obtain images and labels
	data, label = cn.L.ImageData(source=args.data, batch_size=args.batch_size, new_width=227, new_height=227,
								 transform_param=dict(mean_file=args.mean_file_name, scale=1.0),ntop=2)
	conv1= cn.conv(data, 11, filter_num, stride=1, param=cn.learned_param)
	layers = []
	layers.append({'name':'data','layer':data})
	layers.append({'name':'label','layer':label})
	layers.append({'name':'conv1','layer':conv1})

	net_save_path = '%sinit_ori.prototxt'%result_folder
	net_ori = cn.customize_network(layers, net_save_path)
	net_ori_dict = {}
	for i,lname in enumerate(net_ori._layer_names):
		net_ori_dict[lname] = i
	imgs, labs = forward(net_ori, 'data', 'data', args.NIT, [])


	#------------------------------------------------------------------------------------------------------------------#
	# Step2: Initialize conv1
	filter_bank_name = 'conv1'
	filter_size = RECEPTIVE_FIELD[filter_bank_name]
	# get code training data
	input_samples, input_samples_lab = collect_training_input(imgs,labs,filter_size,50)
	# get no dc inputs
	input_samples_nodc = []
	for b in input_samples:
		d = b.copy()
		d-= d.mean(axis=(2,3),keepdims=True)
		d/= d.std(axis=(2,3),keepdims=True)
		input_samples_nodc.append(d)


	data, label = cn.L.ImageData(source=args.data,batch_size=input_samples[0].shape[0], new_width=filter_size, new_height=filter_size,
								 transform_param=dict(scale=1.0),ntop=2)

	conv1= cn.conv(data, 11, filter_num, stride=4, param=cn.learned_param)
	relu1 = cn.relu(conv1)

	layers = []
	layers.append({'name':'data','layer':data})
	layers.append({'name':'label','layer':label})
	layers.append({'name':'conv1','layer':conv1})
	layers.append({'name':'relu1','layer':relu1})
	net_save_path = '%sinit_tmp_conv1.prototxt'%result_folder
	net = cn.customize_network(layers, net_save_path)
	net_dict = {}
	for i,lname in enumerate(net._layer_names):
		net_dict[lname] = i


	# train codes
	dict_file = '%sconv1.npy'%result_folder
	if os.path.isfile(dict_file):
		filter_weights_rescale = np.load(dict_file)
	else:
		if args.type=="kmeans_wodc":
			# visualize input
			output_path='%s%s_input.png'%(result_folder,filter_bank_name)
			idx = np.random.randint(0,np.concatenate(input_samples_nodc,axis=0).shape[0]-1,512)
			samples = np.concatenate(input_samples_nodc,axis=0)[idx,:]
			vis_square(samples.transpose((0,2,3,1)), output_path)

			input_code_training = forward(net, 'data', 'data', args.NIT, input_samples_nodc)[0]
			conv1_fb = train_dict(input_code_training, filter_num-3, args.type)
			r = np.ones((1,1,filter_size,filter_size))
			r /= np.sqrt(np.sum(r**2))
			r = np.concatenate([np.zeros((1,1,filter_size,filter_size)),
			np.zeros((1,1,filter_size,filter_size)),r],axis = 1)
			g = r[:,(0,2,1),:,:]
			b = r[:,(2,0,1),:,:]


			filter_weights = np.concatenate([conv1_fb,r,g,b],axis=0)
			net_ori.layers[net_ori_dict['conv1']].blobs[0].data[...] = filter_weights.copy()
			filter_weights_rescale = rescale_weights(net_ori,'data','conv1',args.NIT,imgs)
			# visualize filter weights
			vis_square(filter_weights_rescale.transpose((0,2,3,1)), '%sconv1_weights.png'%result_folder)
			np.save(dict_file, filter_weights_rescale)
		elif args.type=="rand" or args.type=="kmeans_wtdc" or args.type=="pca":

			# visualize input
			output_path='%s%s_input.png'%(result_folder,filter_bank_name)
			idx = np.random.randint(0,np.concatenate(input_samples,axis=0).shape[0]-1,512)
			samples = np.concatenate(input_samples,axis=0)[idx,:]
			vis_square(samples.transpose((0,2,3,1)), output_path)

			input_code_training = forward(net, 'data', 'data', args.NIT, input_samples)[0]
			conv1_fb = train_dict(input_code_training,96, args.type)
			net_ori.layers[net_ori_dict['conv1']].blobs[0].data[...] = conv1_fb.copy()
			filter_weights_rescale = rescale_weights(net_ori,'data','conv1',args.NIT,imgs)
			vis_square(filter_weights_rescale.transpose((0,2,3,1)), '%sconv1_weights.png'%result_folder)
			np.save(dict_file, filter_weights_rescale)

	net_ori.layers[net_ori_dict['conv1']].blobs[0].data[...] = filter_weights_rescale.copy()
	net.layers[net_dict['conv1']].blobs[0].data[...] = filter_weights_rescale







	#------------------------------------------------------------------------------------------------------------------#
	# Step3: analysis
	# visualize discriminate power
	visualize_discriminate_power(net, 'data', 'conv1', args.NIT, input_samples, input_samples_lab,result_folder)

	# visualize filter response map
	# visualize_filter_response_map(net_ori, 'data', 'conv1', args.NIT, imgs,result_folder)
	# visualize filter response distribution
	visualize_filter_response_sparsity(net, 'data', 'conv1', args.NIT, input_samples,result_folder)






if __name__ == "__main__":
	main()