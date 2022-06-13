import os
import pickle
import argparse
import torch
from transformer.dataset import *
from transformer.utils import *
from transformer.transformer import *

#Extracting the image descriptors using a Transformer model pre-trained on ImageNet.

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('directory', metavar='EXPORT_DIR')
	parser.add_argument('--net', default='vb16') #network used
	parser.add_argument('--mini', action='store_true') #use the mini training set
	parser.add_argument('--truth_dir',default=None, type=str, help = 'folder containing the ground truth files')
	parser.add_argument('--im_dir',default=None, type=str, help = 'folder containing the image files')
	
	args = parser.parse_args()

	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)

	#folder name
	network_variant = args.net
	exp_name = network_variant
	
	exp_dir = args.directory+"/"+exp_name+"/"
	
	if not os.path.exists(exp_dir):
		os.makedirs(exp_dir, exist_ok=True)
	
	print("Will save descriptors in {}".format(exp_dir))
	
	extraction_transform = augmentation("augment")

	train_root = args.truth_dir

	train_dataset = MET_database(root = train_root,mini = args.mini,transform = extraction_transform,im_dir = args.im_dir)

	query_root = train_root

	test_dataset = MET_queries(root = query_root,test = True,transform = extraction_transform,im_dir = args.im_dir)
	val_dataset = MET_queries(root = query_root,transform = extraction_transform,im_dir = args.im_dir)

	train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=1,shuffle=False,num_workers=8,pin_memory=True)
	print("Images in the training set: {}".format(len(train_dataset)))

	test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=8,pin_memory=True)
	print("Images in the test set: {}".format(len(test_dataset)))

	val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1,shuffle=False,num_workers=8,pin_memory=True)
	print("Images in the validation set: {}".format(len(val_dataset)))

	#initialization of the global descriptor extractor model

	if network_variant == 'vitb16':
		net = Transformer("vit_b_16", pretrained_flag = True)
	elif network_variant == 'vitl16':
		net = Transformer("vit_l_16", pretrained_flag = True)
	else:
		raise ValueError('The architecture you selected is not available: {}!'.format(network_variant))

	net.cuda()

	scales = [1, 1/np.sqrt(2), 1/2]

	print("Extracting descriptors from images:")
	
	train_descr = extract_features(net,train_loader,ms = scales,msp = 1.0)
	print("Extracted descriptors from the training set.")
	test_descr = extract_features(net,test_loader,ms = scales,msp = 1.0)
	print("Extracted descriptors from the test set.")
	val_descr = extract_features(net,val_loader,ms = scales,msp = 1.0)
	print("Extracted descriptors from the validation set.")

	descriptors_dict = {}

	descriptors_dict["train_descriptors"] = np.array(train_descr).astype("float32")

	descriptors_dict["test_descriptors"] = np.array(test_descr).astype("float32")
	descriptors_dict["val_descriptors"] = np.array(val_descr).astype("float32")

	#saving descriptors
	with open(exp_dir+"descriptors.pkl", 'wb') as data:
		pickle.dump(descriptors_dict,data,protocol = pickle.HIGHEST_PROTOCOL)
		print("descriptors pickle file complete: {}".format(exp_dir+"descriptors.pkl"))


if __name__ == '__main__':
	main()