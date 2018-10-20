import pickle
import numpy as np
from functools import reduce
import pandas as pd
from pathlib import Path
from sklearn.metrics import fbeta_score
import warnings

def val_idxs_from_csv(csv_path, ratio=0.2, load_from_disk=False, file='val_idxs.pkl'):
	"""
	Get a random selection of indexes from a csv file containing the filenames of training images.

	Parameters
	----------
	casv_path : str
		path of csv file that contains the list of filenames for the training images
		
		unused if load_from_disk is True

	ratio : float
		the percentage of indexes that are selected for validation
		
		unused if load_from_disk is True
	
	load_from_disk : bool
		if set to True, will attempt to load a saved sequence of indexes from disk
	
	file : str
		path of file to save and load indexes from

	Returns
	-------
	sequence of int
		Random selection of indexes
	"""

	train = pd.read_csv(csv_path)

	if load_from_disk:
		val_file = open(file, 'rb')
		val_idxs = pickle.load(val_file)
		print(f"Loaded {len(val_idxs)} indexes from file {val_file}.")
	else:
		index_count = int(ratio*len(train.index))
		val_idxs = np.random.choice(train.index, index_count)
		val_file = open(file, 'wb')
		pickle.dump(val_idxs, val_file)
		print(f"Generated {len(val_idxs)} indexes from a total of {len(train.index)} indexes.")

	val_file.close()

	return val_idxs

def create_submission(probs, data, threshold=0.2):
	classes = np.array(data.classes, dtype=str)
	filenames = [Path(f).stem for f in data.test_ds.fnames]

	labels = np.array([" ".join(classes[p>threshold]) for p in probs])

	res = np.array(tuple(zip(filenames, labels)))
	res_df = pd.DataFrame(res, columns=['image_name', 'tags'])

	return res_df

def f2(preds, targs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fbeta_score(targs, preds, beta=2, average='samples')

def f2_th_raw(probs, targs, start=0.1, stop=0.4, step=0.01):
	thresholds = np.arange(start, stop, step)
	f2s = [f2(targs, probs>th) for th in thresholds]
	return thresholds[np.argmax(f2s)], np.max(f2s)

def get_threshold(probs, targs, start=0.1, stop=0.4, step=0.01):
    return f2_th_raw(probs, targs, start, stop, step)[0]

def f2_th(probs, targs, start=0.1, stop=0.4, step=0.01):
    return f2_th_raw(probs, targs, start, stop, step)[1]
		