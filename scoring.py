from sklearn.metrics import *
import numpy as np

def get_homogeneity_score(ground_truth, predictions):
	return homogeneity_score(ground_truth, predictions)

def get_rand_index(ground_truth, predictions):
	return adjusted_rand_score(ground_truth, predictions)