import numpy as np
import pickle5 as pickle
import json
import argparse
from cnn.utils import *
from cnn.knn_classifier import *

def main():
	
	arg_parser = argparse.ArgumentParser()
	
	arg_parser.add_argument('directory', metavar='EXPORT_DIR', help='folder where train descriptors are and where results will be saved')
	arg_parser.add_argument('--truth_dir', default=None, type=str, help = 'folder where ground truth is')

	args = arg_parser.parse_args()

	log = open(args.directory+"/knn_res.txt", 'a')

	#ground truth files

	train_file = args.truth_dir + "/mini_MET_database.json"
	test_file = args.truth_dir + "/testset.json"
	val_file = args.truth_dir + "/valset.json"

	#loading descriptors

	print('Loading descriptor files.')
	with open(args.directory+"/descriptors.pkl", 'rb') as data:
		data_dict = pickle.load(data)
		train_descr = np.array(data_dict["train_descriptors"]).astype("float32") 
		test_descr = np.array(data_dict["test_descriptors"]).astype("float32")
		val_descr = np.array(data_dict["val_descriptors"]).astype("float32")

	print("{} training descriptors.".format(len(train_descr)))
	print("{} test descriptors.".format(len(test_descr)))
	print("{} validation descriptors.".format(len(val_descr)))

	train_descr = np.ascontiguousarray(train_descr, dtype=np.float32)
	test_descr = np.ascontiguousarray(test_descr, dtype=np.float32) 
	val_descr = np.ascontiguousarray(val_descr, dtype=np.float32)

	#loading ground truth info for all sets (train, val, test)

	print('Loading ground truth information.')

	#training set (mini met db)
	with open(train_file) as data:
		train_info = json.load(data)
	train_truth = []
	for item in train_info:
		train_truth.append(item["id"])
	train_truth = np.array(train_truth)

	#test set
	with open(test_file) as data:
		test_info = json.load(data)
	test_truth = []
	for item in test_info:
		try:
			test_truth.append(item["MET_id"])
		except:
			test_truth.append(-1)
	test_truth = np.array(test_truth)

	#validation set
	with open(val_file) as data:
		val_info = json.load(data)
	val_truth = []
	for item in val_info:
		try:
			val_truth.append(item["MET_id"])
		except:
			val_truth.append(-1)
	val_truth = np.array(val_truth)

	#pca whitening and normalisation

	print('Applying pca whitening to the descriptors.')
	mean,R = estimate_pca_whiten_with_shrinkage(train_descr,shrinkage=1.0,dimensions=512)

	train_descr = apply_pca_whiten_and_normalise(train_descr,mean,R).astype("float32")
	val_descr = apply_pca_whiten_and_normalise(val_descr,mean,R).astype("float32")
	test_descr = apply_pca_whiten_and_normalise(test_descr,mean,R).astype("float32")

	#hyperparameter tuning on the validation set

	print('Performing hyperparameter tuning on the KNN classifier.')
	grid = {'K' : np.array([1,2,3,5,10,25,50]), 
					't' :np.array([1.0,5.0,10.0,25.0,50.0,100.0,500.0])}        
	best_params = tune_KNN(grid, train_descr, train_truth, val_descr, val_truth)
	print('Best parameters are: ' + str(best_params[1]))
	log.write('Best parameters are: ' + str(best_params[1])); log.write('\n')

	#initialising the knn classifier with the values for the best parameters
		
	clf = KNN_Classifier(K = int(best_params[1]['K']),t = float(best_params[1]['t']))

	#fitting the knn classifier on the training set

	print('Fitting the KNN classifier.')
	clf.fit(train_descr,train_truth)

	#predicting on the test set using the knn classifier

	print('Predicting using the KNN classifier.')
	test_preds,test_confs= clf.predict(test_descr)

	#writing the final results

	print('Final results obtained.')
	gap_score,gap_score_non_distr,acc_score = evaluate(test_preds,test_confs,test_truth)
	print("GAP: {} \n GAP-no-dis: {} \n ACC: {} \n".format(gap_score,gap_score_non_distr,acc_score))
	log.write("GAP: {} \n GAP-no-dis: {} \n ACC: {} \n".format(gap_score,gap_score_non_distr,acc_score))

if __name__ == '__main__':
	main()
