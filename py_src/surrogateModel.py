import numpy as np
import os
from copy import copy, deepcopy
import time
import pickle
from time import process_time
from itertools import product

from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import r2_score

from sklearn.svm import SVR
from catboost import CatBoostRegressor
from pyrfr import regression
from NeuralNet import NeuralNet

def r2_metric(y_true, y_pred):
	r2 = r2_score(y_true, y_pred)	
	return ('R2', r2, True)

def alphabetToList(alphabet, numberOfVariables):
	if alphabet.isnumeric():
		return [int(alphabet) for i in range(numberOfVariables)]
	file = open(alphabet,'r')
	alphabetSizes = file.readline().split(' ')
	file.close()

	return [int(alphabetSizes[i]) for i in range(numberOfVariables)]

class surrogateModel:
	def __init__(self, cur_folder, L, alphabet, randomSeed=42, args={}):
		print(cur_folder, L, alphabet, randomSeed, args)
		
		self.numberOfVariables = L
		self.alphabet = alphabetToList(alphabet, L)
		print('alphabet', self.alphabet)
		self.folder = cur_folder
		self.randomSeed = randomSeed
		np.random.seed(randomSeed)
		self.K = np.max(self.alphabet)

		if 'evalsLimit' not in args:
			args['evalsLimit'] = 10**6
		if 'timeLimit' not in args:
			args['timeLimit'] = 10**6		
		self.args = {'numberOfVariables':self.numberOfVariables, 'alphabet':self.alphabet, 'evalsLimit':args['evalsLimit'], 'timeLimit':args['timeLimit']}
		self.data = {}
		
		self.n_splits = 3
		self.tuneCnt = 0
		self.trainSize = 0

	def loadData(self):
		'''
		Surrogate models are trained on data obtained from file */collectedData,txt
		'''

		raw_data = open(self.folder+'/collectedData.txt', 'r').readlines()[1:]
			
		X,y = [], []
		for line in raw_data:
			line = line.split(' ')
			x = np.array([int(i) for i in line[2].split('_')])
			y_ = float(line[3])
			X.append(x)
			y.append(y_)
		X = np.array(X)
		y = np.array(y)
		data = np.hstack([X, y.reshape(-1,1)])
		print('loaded data shape', data.shape)
		
		self.data['X'] = data[:,:-1]
		self.data['y'] = data[:,-1]
		
		self.min = np.min(self.data['y'])
		self.max = np.max(self.data['y'])
		self.range = self.max-self.min		
		self.data['y'] -= self.min
		if self.range != 0:
			self.data['y'] /= self.range
		print('mean', np.min(self.data['y']), 'std', np.max(self.data['y']))

		print('obtained data for training:', self.data['X'].shape, np.min(self.data['y']), np.max(self.data['y']))
		
		return(self.data['X'], self.data['y'])

	def transformData(self, data):
		'''
		One-hot encoding if there are variables with cardinality > 2
		'''

		if np.max(self.alphabet) > 2:
			result = np.zeros((data.shape[0], int(np.sum(self.alphabet))), dtype=np.int)
			for i in range(data.shape[0]):
				curAlphaSum = 0
				for j in range(data.shape[1]):
					result[i, int(curAlphaSum + data[i,j])] = 1
					curAlphaSum += int(self.alphabet[j])				
			return result
		else:
			return data
	
	def fitRegressor(self, model, X, y):
		print('fitting regressor... ', model, X.shape, y.shape)
		if isinstance(model, NeuralNet):
			ind = np.random.permutation(X.shape[0])
			ind_train = ind[:int(ind.shape[0]*0.9)]
			ind_val = ind[int(ind.shape[0]*0.9):]

			train_X, train_y = X[ind_train], y[ind_train]
			val_X, val_y = X[ind_val], y[ind_val]
			
			encoded_train_X = self.transformData(train_X)
			encoded_val_X = self.transformData(val_X)
			
			try:
				model.fit(encoded_train_X, train_y, eval_metric=r2_metric, eval_set=(encoded_val_X, val_y), early_stopping_rounds=3, verbose=1)
			except Exception as e:
				print(e)
			return model

		elif isinstance(model, CatBoostRegressor):

			model = CatBoostRegressor()
			print(self.cur_params)
			model.set_params(**self.cur_params)

			ind = np.random.permutation(X.shape[0])
			ind_train = ind[:int(ind.shape[0]*0.9)]
			ind_val = ind[int(ind.shape[0]*0.9):]

			train_X, train_y = X[ind_train], y[ind_train]
			val_X, val_y = X[ind_val], y[ind_val]
			
			if train_y.min() == train_y.max():
				train_y[0] += 1e-3
			model.fit(train_X.astype(np.int8), train_y, eval_set=(val_X.astype(np.int8), val_y), early_stopping_rounds=5, 
				verbose_eval=True, use_best_model=True)
			return model

		elif isinstance(model, regression.binary_rss_forest):

			encoded_train_X = self.transformData(X)

			try:
				print(self.cur_params)
				model = None
				num_trees=self.cur_params['num_trees']
				do_bootstrapping = True
				n_points_per_tree = -1
				ratio_features = self.cur_params['ratio_features']
				min_samples_split = self.cur_params['min_samples_split']
				min_samples_leaf = self.cur_params['min_samples_leaf']
				max_depth = 2**20
				eps_purity = 1e-8
				max_num_nodes = 2**20
				model = regression.binary_rss_forest()
				rf_opts = regression.forest_opts()
				rf_opts.num_trees = num_trees
				rf_opts.do_bootstrapping = do_bootstrapping
				max_features = int(encoded_train_X.shape[1]*ratio_features)
				rf_opts.tree_opts.max_features = max_features
				rf_opts.tree_opts.min_samples_to_split = min_samples_split
				rf_opts.tree_opts.min_samples_in_leaf = min_samples_leaf
				rf_opts.tree_opts.max_depth = max_depth
				rf_opts.tree_opts.epsilon_purity = eps_purity
				rf_opts.tree_opts.max_num_nodes = max_num_nodes
				rf_opts.compute_law_of_total_variance = False
				if n_points_per_tree <= 0:
					rf_opts.num_data_points_per_tree = encoded_train_X.shape[0]
				else:
					rf_opts.num_data_points_per_tree = n_points_per_tree
				model.options = rf_opts
				
				data = regression.default_data_container(encoded_train_X.shape[1])
				for i in range(encoded_train_X.shape[0]):
					#print(encoded_train_X[i], expanded_train_y[i])
					data.add_data_point(list(encoded_train_X[i].astype(np.float64)), y[i].astype(np.float64))
				for i in range(encoded_train_X.shape[1]):
					data.set_bounds_of_feature(i, 0.0, 1.0)

				rng = regression.default_random_engine()
				model.fit(data,rng=rng)
				return model

			except Exception as e:
				print(e)
			
		else:
			try:
				encoded_train_X = self.transformData(X)
			except Exception as e:
				print(e)
			print(encoded_train_X.shape, y.shape)
			model.fit(encoded_train_X, y)
			return model
		#except Exception as e:
		#	print(e)

	def predictRegressor(self, model, X):
		encoded_X = self.transformData(X)
		print('predict',X.shape, encoded_X.shape, model)
		if isinstance(model, NeuralNet):
			return model.predict(encoded_X)
		elif isinstance(model, CatBoostRegressor):
			return model.predict(X.astype(np.int8))
		elif isinstance(model, regression.binary_rss_forest):
			data = regression.default_data_container(encoded_X.shape[1])
			preds = []
			for i in range(encoded_X.shape[0]):
				data.add_data_point(list(encoded_X[i].astype(np.float64)), np.array([-1.0]).astype(np.float64)[0])			
			for i in range(encoded_X.shape[1]):
				data.set_bounds_of_feature(i, 0.0, 1.0)

			for i in range(encoded_X.shape[0]):
				feature_vector = data.retrieve_data_point(i)
				try:
					pred = np.array(model.predict(feature_vector))
				except Exception as e:
					print(e)
				preds.append(pred)
			preds = np.array(preds)
			print(preds)
			return preds
		else:
			return model.predict(encoded_X)


	def crossValidationRegression(self, model, X, y):
		y_true, y_preds = [], []
		
		for train_index, val_index in self.cv_split.split(X, y):
			
			train_X, train_y = X[train_index], y[train_index]
			val_X, val_y = X[val_index], y[val_index]
			
			model = self.fitRegressor(model, train_X, train_y)
			
			val_y = list(val_y)
			cur_preds = list(self.predictRegressor(model, val_X))

			y_true += val_y
			y_preds += cur_preds

		score = r2_score(y_true, y_preds)
		return score

	def tuneModelRegression(self, model, data_X, data_y):
		print('tuning regresion surrogate model ', model)
		start = time.process_time()
		#try:
		if self.tuneCnt >= 1:
			print('tuning not needed!')
			if not isinstance(model, CatBoostRegressor):
				return model, 0
			else:
				model = CatBoostRegressor()
				print(self.best_params)
				model.set_params(**self.best_params)
				return model, 0
		
		self.tuneCnt += 1	
		
		if isinstance(model, SVR):
			params = {'kernel' : ['rbf', 'sigmoid']}		
			
		elif isinstance(model, NeuralNet):
			params = {'lr':[1e-3], 'epochs':[1000]}		
			
		elif isinstance(model, CatBoostRegressor):
			n = data_X.shape[1]
			params = {'eval_metric':['RMSE'], 'iterations':[300], 'learning_rate':[1.0, 1e-1, 1e-2], 'cat_features':[list(np.arange(n))], 'verbose':[0],
			'depth':[3,6,9,12]}
			model = CatBoostRegressor()
		
		elif isinstance(model, regression.binary_rss_forest):
			params = {'num_trees':[10],'ratio_features':[5.0/6, 1.0], 'min_samples_split':[1,3,10], 'min_samples_leaf':[1,3,10]}
				
		else:	
			print(model, type(model), 'Model not implemented!')

		try:
			keys, values = zip(*params.items())
			best_params = None
			best_score = -1e+308
			print(product(*values))
			for bundle in product(*values):
				cur_params = dict(zip(keys, bundle))
				cur_model = deepcopy(model)
				if not isinstance(model, regression.binary_rss_forest):
					cur_model.set_params(**cur_params)
				self.cur_params = deepcopy(cur_params)

				if len(list(product(*values))) > 1:
					score = self.crossValidationRegression(cur_model, data_X, data_y)
					print(cur_params, best_params, 'score', score, 'best score so far:', best_score)
					if score > best_score:
						best_score = score
						best_params = copy(cur_params)
				else:
					best_score = 1.0
					best_params = copy(cur_params)
					
			self.best_params = deepcopy(best_params)
			self.cur_params = deepcopy(best_params)
			if not isinstance(model, regression.binary_rss_forest):
				model.set_params(**best_params)
		except Exception as e:
			print(e)
		print('best params=', best_params, 'best R^2=', best_score)
		#self.tunedModel = True
		print('tuning surrogate model finished!')
		print('tuning time', time.process_time()-start)
		print('tuned model', model)
		if isinstance(model, CatBoostRegressor):
			model = CatBoostRegressor()
			model.set_params(**self.best_params)
		print(model)
		return model, best_score

	def trainModel(self):
		try:
			self.data['X'], self.data['y'] = self.loadData()
		except Exception as e:
			print(e)

		if not isinstance(self.data['X'], np.ndarray):
			print('Data error! exiting model Training!...')
			return -1e+308, -1e+308

		print(self.data['X'].shape[0], self.trainSize, 'fitting regressor')
		if self.data['X'].shape[0] > self.trainSize:
			self.cv_split = KFold(n_splits=self.n_splits, random_state=42, shuffle=True)	
			self.regressor,_ = self.tuneModelRegression(self.regressor, self.data['X'], self.data['y'])
			if isinstance(self.regressor, regression.binary_rss_forest) or isinstance(self.regressor, CatBoostRegressor):
				self.regressor = self.fitRegressor(self.regressor, self.data['X'], self.data['y'])
			else:
				self.fitRegressor(self.regressor, self.data['X'], self.data['y'])
				
		self.trainSize = self.data['X'].shape[0]

	def surrogateFitness(self, solution):
		try:
			solution = np.array(solution).astype(np.int8).reshape(1,-1)
			prediction = self.predictRegressor(self.regressor, solution)
			#print('model prediction', prediction, prediction.shape, np.mean(prediction), np.min(prediction), np.max(prediction))
			prediction = np.mean(prediction)
			#print(prediction, self.range, self.min)
			prediction = prediction * self.range + self.min
		except Exception as e:
			print(e)
		return prediction

	def updateModel(self):
		
		print ('updating surrogate')
		self.trainModel()
		print ('updating model finished!')

#################################################################################

class surrogateModelSVR(surrogateModel):
	def __init__(self, cur_folder, L, alphabet, randomSeed=42, args={}):
		try:
			surrogateModel.__init__(self, cur_folder, L, alphabet, randomSeed, args)
			self.regressor = SVR()
			print('init finished')
		except Exception as e:
			print(e)

	
class surrogateModelCatboost(surrogateModel):
	def __init__(self, cur_folder, L, alphabet, randomSeed=42, args={}):
		try:
			surrogateModel.__init__(self, cur_folder, L, alphabet, randomSeed, args)
			self.regressor = CatBoostRegressor()
			print('init finished')
		except Exception as e:
			print(e)

class surrogateModelRF(surrogateModel):
	def __init__(self, cur_folder, L, alphabet, randomSeed=42, args={}):
		try:
			surrogateModel.__init__(self, cur_folder, L, alphabet, randomSeed, args)
			self.regressor = regression.binary_rss_forest()
			print('init finished')
		except Exception as e:
			print(e)

class surrogateModelNeuralNet(surrogateModel):
	def __init__(self, cur_folder, L, alphabet, randomSeed=42, args={}):
		try:
			surrogateModel.__init__(self, cur_folder, L, alphabet, randomSeed, args)
			self.regressor = NeuralNet()
			print('init finished')
		except Exception as e:
			print(e)
#################################################################################
