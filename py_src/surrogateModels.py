import numpy as np
import os

from lightgbm import LGBMRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import spearmanr
from itertools import product
from copy import copy, deepcopy

from WalshModel import WalshModel
from NeuralNet import NeuralNet

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import signal

class surrogateModel:
	def __init__(self, folder, numberOfVariables, alphabet, type):
		try:
			print('initializing surrogate models... | Params:', 'folder:', folder, 'L:', numberOfVariables, 'alpha:', alphabet, 'Type:', type)
			self.folder = folder
			self.numberOfVariables = numberOfVariables
			self.alphabet = alphabet
			self.type = type
			self.n_splits = 5
			self.modelsNames = ['WalshModel', 'NeuralNet', 'SVR', 'GaussianProcessRegressor', 'LGBMRegressor']
			self.modelsNames = ['RandomForestRegressor']

			models = {'NeuralNet':[NeuralNet() for i in range(self.n_splits)],
						   'SVR':[SVR() for i in range(self.n_splits)],
						   'GaussianProcessRegressor':[GaussianProcessRegressor() for i in range(self.n_splits)],
						   'WalshModel':[WalshModel(alphabet=alphabet) for i in range(self.n_splits)],
						   'LGBMRegressor':[LGBMRegressor() for i in range(self.n_splits)],
						   'RandomForestRegressor':[RandomForestRegressor(n_estimators=30) for i in range(self.n_splits)]}
						   
			if self.type == -1:
				self.models = {}
				for name in self.modelsNames:
					self.models[name] = models[name]
			else:
				self.models = {self.modelsNames[self.type]:models[self.modelsNames[self.type]]}
			print('models',self.models)
			# self.models = {'SVR':[SVR() for i in range(self.n_splits)],
			# 'GaussianProcessRegressor':[GaussianProcessRegressor() for i in range(self.n_splits)],
			# }#'LGBMRegressor':[LGBMRegressor() for i in range(self.n_splits)]}
			
			self.encoder = OneHotEncoder(categories=[np.arange(self.alphabet) for j in range(self.numberOfVariables)], sparse=False)		
			self.scores = []
			self.modelQuality = {'spearman':[], 'r2':[]}
			self.cv_split = None
			self.prevDataSize = -1
			self.modelTrained = {}
			self.failedModels = {}
			
		except Exception as e:
			print(e)

	def removeDuplicates(self, data):
		sorted_data = data[np.lexsort(data.T),:]
		row_mask = np.append([True],np.any(np.diff(sorted_data,axis=0),1))
		return sorted_data[row_mask]

	def loadData(self, modelId):
		#print(os.path.join(self.folder, 'populations/population_%d.txt' % modelId))
		try:
			data = None
			print('reading from folder ', self.folder)
			if os.path.exists(os.path.join(self.folder, 'populations/population_%d.txt' % modelId)):
				data = np.loadtxt(os.path.join(self.folder, 'populations/population_%d.txt' % modelId))
				if len(data.shape) == 1:
					data = data.reshape(1, -1)
				#print(data.shape)
			if os.path.exists(os.path.join(self.folder, 'populations/population_-1.txt')):
				data_random = np.loadtxt(os.path.join(self.folder, 'populations/population_-1.txt'))			
				if len(data_random.shape) == 1:
					data_random = data_random.reshape(1, -1)

				#print('random data', data_random)
				#print(data_random.shape)
				if isinstance(data, np.ndarray):
					data = np.vstack([data, data_random])
				else:
					data = np.copy(data_random)
				#print(data.shape)
			#print('before remove duplicates',data,data.shape)			
			data = self.removeDuplicates(data)
			#print('after remove duplicates',data,data.shape)

			data_X = data[:, :-1]
			data_y = data[:, -1]
			
			self.min = np.min(data_y)
			self.max = np.max(data_y)
			self.range = self.max-self.min		
			data_y -= self.min
			if self.range != 0:
				data_y /= self.range
			print('mean', np.min(data_y), 'std', np.max(data_y))

			#print('data_y', data_y)
			print('obtained data for training:', data.shape, np.min(data_y), np.max(data_y))
			if data.shape[0] >= 4:
				self.cv_split = KFold(n_splits=min(self.n_splits, data.shape[0]//2), random_state=42, shuffle=True)						
			else:
				print('not enough samples to train!')
				return None, None

		except Exception as e:
			print(e)
			return None, None

		return(data_X, data_y)

	def spearman_scorer(self, estimator, X, y):
		y_pred = estimator.predict(X)
		spearman = spearmanr(y, y_pred)[0]
		return spearman

	def spearman_metric(self, y_true, y_pred):
		#print('spearman', y_true, y_pred, )
		spearman = spearmanr(y_true, y_pred)[0]	
		#print('spearman', y_true, y_pred, spearman)	
		return ('spearman', spearman, True)

	def r2_metric(self, y_true, y_pred):
		#print('r2', y_true, y_pred, )
		r2 = r2_score(y_true, y_pred)	
		#print('spearman', y_true, y_pred, spearman)	
		return ('R2', r2, True)

	def plotModelQuality(self, modelId):
		scores = self.modelQuality['spearman']
		scores_r2 = self.modelQuality['r2']
		print('scores', scores)
		print('scores_r2', scores_r2)

		try:	
			lines = []
			for model in scores:
				p, = plt.plot(np.arange(len(model)), np.array(model))
				lines.append(p)
				plt.xlabel('number of fittings')
				plt.ylabel('spearman')
				plt.legend(lines, np.arange(len(lines)))
				plt.savefig('%s/model_quality_spearman.png' % self.folder)
				plt.close()
			lines = []
			for model in scores_r2:
				p, = plt.plot(np.arange(len(model)), np.array(model))
				lines.append(p)
				plt.xlabel('number of fittings')
				plt.ylabel(r'$R^2$')
				plt.legend(lines, np.arange(len(lines)))
				plt.savefig('%s/model_quality_R2.png' % self.folder)
				plt.close()
		except Exception as e:
			print(e)

	def crossValidation(self, model, X, y):
		y_true, y_preds = [], []
		for train_index, val_index in self.cv_split.split(X, y):
			train_X, train_y = X[train_index], y[train_index]
			val_X, val_y = X[val_index], y[val_index]
			
			if isinstance(model, LGBMRegressor) or isinstance(model, NeuralNet):
				model.fit(train_X, train_y, eval_metric=self.r2_metric, eval_set=(val_X, val_y), early_stopping_rounds=3, verbose=-1)
			else:
				model.fit(train_X, train_y)
			
			val_y = list(val_y)
			cur_preds = list(model.predict(val_X))
			#print('cv preds',val_y, cur_preds)
			y_true += val_y
			y_preds += cur_preds

		#print(y_true, y_preds)
		score = spearmanr(y_true, y_preds)[0]
		score_r2 = r2_score(y_true, y_preds)
		#print(score, score_r2)
		return score, score_r2, y_preds

	def crossValidationMultipleModels(self, models, storeData = False):
		y_true, y_preds = [], []
		modelInd = 0
		#print('crossValidationMultipleModels')
		if isinstance(models[modelInd], LGBMRegressor):
			X, y = self.data_X, self.data_y
		else:
			X,y = self.encoded_data_X, self.data_y

		#try:
		for train_index, val_index in self.cv_split.split(X, y):
			train_X, train_y = X[train_index], y[train_index]
			val_X, val_y = X[val_index], y[val_index]
			
			if isinstance(models[modelInd], LGBMRegressor) or isinstance(models[modelInd], NeuralNet):
				models[modelInd].fit(train_X, train_y, eval_metric=self.r2_metric, eval_set=(val_X, val_y), early_stopping_rounds=3, verbose=-1)
			else:
				models[modelInd].fit(train_X, train_y)
			
			val_y = list(val_y)
			cur_preds = list(models[modelInd].predict(val_X))
			
			y_true += val_y
			y_preds += cur_preds

			if storeData:
				for i in range(len(val_y)):
					tuple_x = tuple(val_X[i])
					if tuple_x not in self.all_data:
						self.all_data[tuple_x] = (val_y[i], cur_preds[i])
					else:
						print('solution already added!')

			modelInd += 1

		#except Exception as e:
		#	print(e)
		#print('crossValidationMultipleModels finished!')
		score = spearmanr(y_true, y_preds)[0]
		score_r2 = r2_score(y_true, y_preds)
		#print(score, score_r2)
		return score, score_r2, y_preds

	def train(self, model):
		#print('training surrogate model ', model)
		if isinstance(model, LGBMRegressor):
			train_X, val_X, train_y, val_y = train_test_split(self.data_X, data_y, test_size=0.1, random_state=42)
			model.fit(train_X, train_y, eval_metric=self.r2_metric, eval_set=(val_X, val_y), early_stopping_rounds=3, 
				categorical_feature=list(np.arange(train_x.shape[1])), verbose=-1)			
		elif isinstance(model, NeuralNet):
			train_X, val_X, train_y, val_y = train_test_split(self.encoded_data_X, self.data_y, test_size=0.1, random_state=42)
			model.fit(train_X, train_y, eval_metric=self.r2_metric, eval_set=(val_X, val_y), verbose=-1)					
		else:
			model.fit(self.encoded_data_X, self.data_y)
		#print('training surrogate models finished!')
		return model

	def tuneModel(self, model):
		print('tuning surrogate model ', model)

		if isinstance(model, SVR):
			params = {'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': ['scale', 'auto'], 'C':[1.0, 0.1, 10.0]}
			X, y = self.encoded_data_X, self.data_y
		elif isinstance(model, GaussianProcessRegressor):
			params = {'alpha':[1e-15, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0], 'normalize_y':[False, True]}
			X, y = self.encoded_data_X, self.data_y
		elif isinstance(model, WalshModel):
			params = {'order':[1,2], 'alpha':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}
			X, y = self.encoded_data_X, self.data_y
		elif isinstance(model, NeuralNet):
			params = {'filter_size':[1, 5], 'stride':[1], 'dilation':[1], 'filter_size2':[1], 'nb_fc_layers':[1], 'lr':[1e-4, 1e-3], 'early_stopping_rounds':[3, 30]}		
			X, y = self.encoded_data_X, self.data_y
		elif isinstance(model, LGBMRegressor):
			params = {'boosting_type':['gbdt'], 'max_depth':[-1, 5], 'learning_rate':[1e-2, 1e-3], 'n_estimators':[1000], 
			'min_child_samples':[1, 10], 'subsample':[1.0], 'colsample_bytree':[1.0, 0.5]}
			#params = {'boosting_type':['gbdt'], 'max_depth':[-1], 'learning_rate':[1e-1], 'n_estimators':[1000], 
			#'min_child_samples':[1,20], 'verbose':[-1]}
			X, y = self.data_X, self.data_y
		elif isinstance(model, RandomForestRegressor):
			params = {'max_depth':[None, 3, 10], 'max_features':['auto','sqrt','log2'], 'bootstrap':[True], 
			'min_samples_leaf':[0.01, 0.03, 0.1], 'min_samples_split':[0.01, 0.03, 0.1]}
			X, y = self.encoded_data_X, self.data_y
		else:
			print(model, type(model), 'Model not implemented!')
			#exit(0)

		print(X.shape, y.shape)
		#try:
		keys, values = zip(*params.items())
		best_params = None
		best_score = -1e+308
		for bundle in product(*values):
			cur_params = dict(zip(keys, bundle))
			model.set_params(**cur_params)
			print(cur_params)
			score, score_r2, _ = self.crossValidation(model, X, y)
			print('score', score)
			if score > best_score:
				best_score = score
				best_params = copy(cur_params)
		model.set_params(**best_params)
		print(model, 'best params=', best_params)
		#except Exception as e:
		#	print(e)		
		#print(model, best_score, best_params)
		print('tuning surrogate model finished!')
		return model

	def handler(self, signum, frame):
		print('time limit reached!')
		signal.alarm(0)
		raise Exception

	def trainModel(self, modelId, tune, timeLimitSeconds, test = False):
		print('training model with id ', modelId)
		#signal.signal(signal.SIGALRM, self.handler)
		if modelId not in self.failedModels:
			self.failedModels[modelId] = set([])

		print('failedModels:', self.failedModels)
		self.scores = {}
		self.scores_r2 = {}
		
		modelId = int(modelId)
		self.modelTrained[modelId] = False

		self.data_X, self.data_y = self.loadData(modelId)
		if not isinstance(self.data_X, np.ndarray):
			print('Data error! exiting model Training!...')
			return -1e+308, -1e+308

		if self.alphabet > 2:
			self.encoded_data_X = self.encoder.fit_transform(self.data_X)
		else:
			self.encoded_data_X = np.copy(self.data_X)
		#self.encoded_data_X = []
		#for i in range(self.data_X.shape[0]):
		#	cur = []
		#	for j in range(self.data_X.shape[1]):
		#		for k in range(j+1, self.data_X.shape[1]):
		#			cur.append(int(self.data_X[i][j]==self.data_X[i][k]))
		#	self.encoded_data_X.append(np.copy(cur))
		#	#print(cur)
		#self.encoded_data_X = np.array(self.encoded_data_X).astype(np.float32)
		#print(self.encoded_data_X.shape)
		#print(self.data_X.shape, self.data_X)
		#print(self.encoded_data_X.shape, self.encoded_data_X)
		#print(self.data_X, self.data_y)
		#tune=True
		if not isinstance(self.data_X, np.ndarray):
			print('not training...')
			return
		#print('before tuning', self.models)
		self.all_data = {}

		if self.prevDataSize<=0 or self.data_X.shape[0] - self.prevDataSize >= tune:
			self.prevDataSize = self.data_X.shape[0]		
			for model in self.models:
				if model in self.failedModels[modelId]:
					continue
				print(model, self.models[model])
				
				signal.signal(signal.SIGALRM, self.handler)
				signal.alarm(timeLimitSeconds)

				try:
					tunedModel = self.tuneModel(self.models[model][0])
					self.models[model] = [deepcopy(tunedModel) for i in range(self.n_splits)]
					print(model, tunedModel, self.models[model])
				except Exception as e:
					print(e)
					print('failed model:', model)
					self.failedModels[modelId].add(model)

				signal.alarm(0)

		cv_results = {model:self.crossValidationMultipleModels(self.models[model]) for model in self.models if model not in self.failedModels[modelId]}
		if len(cv_results) == 0:
			print ('All failed!!! Exiting!')
			return None, None

		self.scores = {x:cv_results[x][0] for x in cv_results}
		self.scores_r2 = {x:cv_results[x][1] for x in cv_results}
		
		print('all spearman scores', self.scores)
		print('all R2 scores', self.scores_r2)

		#print ('all y', self.y_preds)
		best_score=-1e+308
		for name in self.scores:
			if self.scores[name] > best_score:
				best_score = self.scores[name]
			self.bestModelName = name
		
		best_score_r2 = self.scores_r2[self.bestModelName]
		print('best model', self.bestModelName)
		
		if test:
			return

		self.crossValidationMultipleModels(self.models[self.bestModelName], storeData=True)
		#print(self.all_data)
		best_low_bound = np.min([self.all_data[x][1] for x in self.all_data])		
		print('best model score:', best_score, 'best model min prediction:', best_low_bound)
				
		best_score = np.round(best_score, 6)
		best_low_bound = np.round(best_low_bound, 6)
		try:
			if len(self.modelQuality['r2'])-1 < modelId:
				self.modelQuality['spearman'].append([])
				self.modelQuality['r2'].append([])
				
			self.modelQuality['spearman'][modelId].append(best_score)
			self.modelQuality['r2'][modelId].append(best_score_r2)
		except Exception as e:
			print(e)
		self.plotModelQuality(modelId)
		self.modelTrained[modelId] = True
		
		return float(best_score), float(best_low_bound)

	def fitness(self, solution, modelId):
		print('predicting surrogate fitness...')
		modelId = int(modelId)
		#print(modelId, self.modelTrained)
		if modelId not in self.modelTrained or self.modelTrained[modelId] == False:
			return -1e+308
		try:
			if tuple(solution) in self.all_data:
				pred = self.all_data[tuple(solution)][1]
			else:
				solution = np.array(solution).reshape(1,-1)			
				best_models = self.models[self.bestModelName]
				#print(solution, best_models)
				if not isinstance(best_models[0], LGBMRegressor):
					solution = self.encoder.transform(solution)
				#print(solution)

				preds = [best_model.predict(solution).flatten() for best_model in best_models]
				pred = np.mean(preds)
				#print(preds, pred)

			if self.range != 0:
				pred *= self.range
			pred += self.min

			print('pred corrected', pred)

		except Exception as e:
			print(e)
		return float(pred)

if __name__ == '__main__':
	model = surrogateModel('test', 50, 3, -1)
	#model.trainModel(0,1,10)
