import os
import time
import pickle
import numpy as np
import openml
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def alphabetToList(alphabet, numberOfVariables):
	if alphabet.isnumeric():
		return [int(alphabet) for i in range(numberOfVariables)]
	file = open(alphabet,'r')
	alphabetSizes = file.readline().split(' ')
	file.close()

	return [int(alphabetSizes[i]) for i in range(numberOfVariables)]

class Logger:
	def __init__(self, folder):
		self.folder = folder
		self.solutions = {}
		self.solutionsCounter = {}
		self.start_time = time.time()
		file = open('%s/optimization.txt' % self.folder,'w')
		file.write('#Evals time solution fitness\n')
		file.close()
		file = open('%s/generalization.txt' % self.folder,'w')
		file.write('#Evals time solution fitness\n')
		file.close()
		file = open('%s/collectedData.txt' % self.folder,'w')
		file.write('#Evals time solution fitness\n')
		file.close()
		
	def nEvals(self):
		return len(self.solutions)

	def elapsedTime(self):
		return time.time()-self.start_time

	def returnSolution(self, x):
		if x not in self.solutions:
			return None
		else:
			with open(self.folder+'/logger.pkl', 'wb') as file:
				pickle.dump(self, file)
			return self.solutions[x]

	def solutionToStr(self, arr):
		x = tuple([str(i) for i in arr])
		x = '_'.join(x)
		return x
		
	def write(self, x, fitness):
		file = open('%s/optimization.txt' % self.folder,'a')
		if x not in self.solutions:
			self.solutions[x] = fitness
			self.solutionsCounter[x] = 1
			with open(self.folder+'/logger.pkl', 'wb') as f:
				pickle.dump(self, f)
		else:
			self.solutionsCounter[x] += 1
	
		elapsed_time = time.time()-self.start_time
		file.write(str(len(self.solutions))+' '+'%.3f'%elapsed_time+' '+ x +' '+str(fitness)+'\n')
		file.close()

	def writeForTraining(self, x, fitness):
		filename = '%s/collectedData.txt' % (self.folder)
		if not os.path.exists(filename):
			file = open(filename,'w')
			file.close()
		file = open(filename,'a')
			
		elapsed_time = time.time()-self.start_time
		file.write(str(len(self.solutions))+' '+'%.3f'%elapsed_time+' '+ x +' '+str(fitness)+'\n')
		file.close()

	def writeGen(self, x, fitness_val, fitness_test):
		file = open('%s/generalization.txt' % self.folder,'a')
		elapsed_time = time.time()-self.start_time
		file.write(str(len(self.solutions))+' '+'%.3f'%elapsed_time+' '+ x +' '+str(fitness_val)+' '+str(fitness_test)+'\n')
		file.close()

def config_to_array(d, numberOfVariables):
	arr=[]
	for i in range(numberOfVariables):
		arr.append(d['x%d'%i])
	return np.array(arr)

def loadData(filename):
	if filename.endswith('.mat'):
		data = loadmat(filename)
	else:
		data = np.loadtxt(filename,delimiter=',').astype(np.float32)
		data={'X':data[:,:-1], 'Y':data[:,-1]}
	return data

class DummyFitnessFunction:
	def __init__(self, folder, filename, numberOfVariables, alphabet):
		self.logger = Logger(folder) 
		self.numberOfVariables = int(numberOfVariables)		
		self.alphabet = alphabetToList(alphabet, numberOfVariables)
		self.K = np.max(self.alphabet)
		self.filename = filename

	def nEvals(self):
		cnt = 0
		for solution in self.logger.solutionsCounter:
			cnt += self.logger.solutionsCounter[solution]
		return cnt

	def neg_fitness(self, x): #for SMAC
		return float(-self.fitness(x))

	def fitness(self, x):
		'''
		A dummy fitness example showing all necessary actions in a fitness function
		'''
		x = np.array(x).astype(np.int32)
		xStr = self.logger.solutionToStr(x) #convert solution to str
		find = self.logger.returnSolution(xStr) #check if it was already evaluated
		if find != None:
			return find
		score = np.random.uniform(0,1) #dummy fitness value
		
		self.logger.write(xStr, score) #write to the file '*/optimization.txt', increments solution counter			
		self.logger.writeForTraining(xStr, score) #write to the file '*/collectedData.txt' to use for surrogate model training	
 		
		return score

class Ensembling(DummyFitnessFunction):
	'''
	Ensembling of K SVMs, fitness is the ensemble accuracy.
	This function is deterministic.
	'''

	def __init__(self, folder, filename, numberOfVariables, alphabet):
		print(folder, filename, numberOfVariables, alphabet)
		self.logger = Logger(folder) 
		self.numberOfVariables = int(numberOfVariables)		
		self.alphabet = alphabetToList(alphabet, numberOfVariables)
		self.K = np.max(self.alphabet)
		self.filename = filename

		task = openml.tasks.get_task(int(filename))  
		X, y = task.get_X_and_y()
		print(X.shape,y.shape)
		np.random.seed(42)

		ind = np.random.permutation(np.arange(len(X)))		
		train_ind = ind[:numberOfVariables]
		val_test_ind = ind[-1000:]
		val_ind = val_test_ind[:len(val_test_ind)//2] #500 samples for validation
		test_ind = val_test_ind[len(val_test_ind)//2:] #500 samples for test
				
		self.train_data = [X[train_ind], y[train_ind]]
		self.val_data = [X[val_ind], y[val_ind]]
		self.test_data = [X[test_ind], y[test_ind]]

		#all data is scaled 
		scaler = MinMaxScaler()
		scaler.fit(self.train_data[0])
		self.train_data[0] = scaler.transform(self.train_data[0])
		self.val_data[0] = scaler.transform(self.val_data[0])
		self.test_data[0] = scaler.transform(self.test_data[0])
		
		print(len(X), len(val_test_ind), len(self.train_data), len(self.val_data), len(self.test_data))
		

	def fitness(self, x):
		try:
			if not isinstance(x, list) and not isinstance(x, tuple) and not isinstance(x, np.ndarray):
				x = config_to_array(x, self.numberOfVariables)
			x = np.array(x).astype(np.int32)
			xStr = self.logger.solutionToStr(x)
			
			samples_by_cluster = [[] for i in range(self.K)]
			for i in range(self.K):
				samples_by_cluster[i] = [j for j in range(self.numberOfVariables) if x[j] == i]     
			samples_by_cluster = sorted(samples_by_cluster, key=lambda x : np.min(x) if len(x) else 10**6)
			
			normed_x = []
			for sample in range(self.numberOfVariables):
				for i,cluster in enumerate(samples_by_cluster):
					if sample in cluster:
						normed_x.append(i)
						break
			
			x = np.array(x).astype(np.int32)
			normed_x = np.array(normed_x).astype(np.int32)

			xStr_normed = self.logger.solutionToStr(normed_x)
			find = self.logger.returnSolution(xStr_normed)
			if find != None:
				return find
			xStr = self.logger.solutionToStr(x)
			
			self.n_classes = np.max(self.train_data[1])+1
			all_preds = np.zeros((self.val_data[0].shape[0], self.n_classes))
			all_preds_test = np.zeros((self.test_data[0].shape[0], self.n_classes))
			all_preds_cnt = np.zeros((self.val_data[0].shape[0], self.n_classes))
			all_preds_test_cnt = np.zeros((self.test_data[0].shape[0], self.n_classes))
			
			accuracies, test_accuracies = [], []
			cnt = 0
			for cluster in samples_by_cluster:
				cluster = np.array(cluster).astype(np.int32)
				if len(cluster) == 0:
					continue
				model = SVC(probability=True)
				np.random.seed(42)
				if np.unique(self.train_data[1][cluster]).shape[0] == 1:
					continue

				model.fit(self.train_data[0][cluster], self.train_data[1][cluster])
				cur_preds = model.predict_proba(self.val_data[0])
				
				cur_preds_test = model.predict_proba(self.test_data[0])
				
				all_preds[:,model.classes_] += cur_preds
				all_preds_test[:,model.classes_] += cur_preds_test
				all_preds_cnt[:,model.classes_] += 1
				all_preds_test_cnt[:,model.classes_] += 1

				cnt += 1
				
			all_preds /= all_preds_cnt
			all_preds_test /= all_preds_test_cnt
			ensemble = np.argmax(all_preds, axis=1)
			ensemble_test = np.argmax(all_preds_test, axis=1)
			
			score = accuracy_score(self.val_data[1], ensemble)
			
			test_score = accuracy_score(self.test_data[1], ensemble_test)
			print(score, test_score)
			
			self.logger.write(xStr_normed, score)			
			self.logger.writeForTraining(xStr, score)
			self.logger.writeGen(xStr, score, test_score)

			return score
		except Exception as e:
			print(e)

	

class Ensembling2(Ensembling):
	pass

class Ensembling5(Ensembling):
	pass

class Ensembling10(Ensembling):
	pass

class EnsemblingNoisy(Ensembling):
	'''
	Ensembling of K SVMs, fitness is the ensemble accuracy.
	This function is noisy.
	'''

	def fitness(self, x):
		
		try:
			if not isinstance(x, list) and not isinstance(x, tuple) and not isinstance(x, np.ndarray):
				x = config_to_array(x, self.numberOfVariables)
			x = np.array(x).astype(np.int32)
			xStr = self.logger.solutionToStr(x)
			
			samples_by_cluster = [[] for i in range(self.K)]
			for i in range(self.K):
				samples_by_cluster[i] = [j for j in range(self.numberOfVariables) if x[j] == i]     
			
			self.n_classes = np.max(self.train_data[1])+1
			all_preds = np.zeros((self.val_data[0].shape[0], self.n_classes))
			all_preds_test = np.zeros((self.test_data[0].shape[0], self.n_classes))
			all_preds_cnt = np.zeros((self.val_data[0].shape[0], self.n_classes))
			all_preds_test_cnt = np.zeros((self.test_data[0].shape[0], self.n_classes))
			
			accuracies, test_accuracies = [], []

			cnt = 0
			for cluster in samples_by_cluster:
				cluster = np.array(cluster).astype(np.int32)
				if len(cluster) == 0:
					continue
				
				model = SVC(probability=True)
				np.random.seed((int(time.time()*len(samples_by_cluster)*(1+cnt)*self.nEvals()))% (2**32))

				if np.unique(self.train_data[1][cluster]).shape[0] == 1:
					continue
					
				model.fit(self.train_data[0][cluster], self.train_data[1][cluster])
				cur_preds = model.predict_proba(self.val_data[0])
				
				cur_preds_test = model.predict_proba(self.test_data[0])
				
				all_preds[:,model.classes_] += cur_preds
				all_preds_test[:,model.classes_] += cur_preds_test
				all_preds_cnt[:,model.classes_] += 1
				all_preds_test_cnt[:,model.classes_] += 1
				cnt += 1
				#print(all_preds)

				#print(cur_preds_test)
				
			all_preds /= all_preds_cnt
			all_preds_test /= all_preds_test_cnt
			
			ensemble = np.argmax(all_preds, axis=1)
			ensemble_test = np.argmax(all_preds_test, axis=1)
			
			score = accuracy_score(self.val_data[1], ensemble)
			test_score = accuracy_score(self.test_data[1], ensemble_test)
			print(score, test_score)
			
			self.logger.write(xStr, score)			
			self.logger.writeForTraining(xStr, score)
			self.logger.writeGen(xStr, score, test_score)

			return score

		except Exception as e:
			print(e)
	
class EnsemblingNoisy2(EnsemblingNoisy):
	pass

class EnsemblingNoisy5(EnsemblingNoisy):
	pass

class EnsemblingNoisy10(EnsemblingNoisy):
	pass


def getBaseline():
	acc_dict = {}
	task_ids = ['146822','43', '9960', '3917', '145945']
	for id in task_ids:
		task = openml.tasks.get_task(int(id))  
		X, y = task.get_X_and_y()
		print(X.shape,y.shape)
		
		np.random.seed(42)

		for numberOfVariables in [100,250,500]:
			ind = np.random.permutation(np.arange(len(X)))		
			train_ind = ind[:numberOfVariables]
			val_test_ind = ind[-1000:]
			val_ind = val_test_ind[:len(val_test_ind)//2]
			test_ind = val_test_ind[len(val_test_ind)//2:]
					
			train_data = [X[train_ind], y[train_ind]]
			val_data = [X[val_ind], y[val_ind]]
			test_data = [X[test_ind], y[test_ind]]

			scaler = MinMaxScaler()
			scaler.fit(train_data[0])
			train_data[0] = scaler.transform(train_data[0])
			val_data[0] = scaler.transform(val_data[0])
			test_data[0] = scaler.transform(test_data[0])
			
			print(len(X), X.shape, y.min() ,y.max())
			model = SVC(probability=True)
			np.random.seed(42)
			model.fit(train_data[0], train_data[1])
			preds = model.predict_proba(val_data[0])
			preds = np.argmax(preds,axis=1)
			acc = accuracy_score(val_data[1], preds)
			print('baseline acc', id, numberOfVariables, acc)
			
			preds_test = model.predict_proba(test_data[0])
			preds_test = np.argmax(preds_test,axis=1)
			acc_test = accuracy_score(test_data[1], preds_test)
			print('baseline acc test', id, numberOfVariables, acc_test)
			
			if id not in acc_dict:
				acc_dict[id] = {}
			acc_dict[id][numberOfVariables] = (acc, acc_test)
		print('\n')

	pickle.dump(acc_dict, open('baselines.pkl','wb'))

def getNoisyBaseline():
	acc_dict = {}
	task_ids = ['146822','43', '9960', '3917', '145945']
	for id in task_ids:
		task = openml.tasks.get_task(int(id))  
		X, y = task.get_X_and_y()
		print(X.shape,y.shape)
		
		np.random.seed(42)

		for numberOfVariables in [100,250,500]:
			ind = np.random.permutation(np.arange(len(X)))		
			train_ind = ind[:numberOfVariables]
			val_test_ind = ind[-1000:]
			val_ind = val_test_ind[:len(val_test_ind)//2]
			test_ind = val_test_ind[len(val_test_ind)//2:]
					
			train_data = [X[train_ind], y[train_ind]]
			val_data = [X[val_ind], y[val_ind]]
			test_data = [X[test_ind], y[test_ind]]

			scaler = MinMaxScaler()
			scaler.fit(train_data[0])
			train_data[0] = scaler.transform(train_data[0])
			val_data[0] = scaler.transform(val_data[0])
			test_data[0] = scaler.transform(test_data[0])
			
			acc = []
			acc_test = []

			print(len(X), X.shape, y.min() ,y.max())
			for k in range(10):
				model = SVC(probability=True)
				model.fit(train_data[0], train_data[1])
				preds = model.predict_proba(val_data[0])
				preds = np.argmax(preds,axis=1)
				acc.append(accuracy_score(val_data[1], preds))
				#print('baseline acc', id, numberOfVariables, acc)
				preds_test = model.predict_proba(test_data[0])
				preds_test = np.argmax(preds_test,axis=1)
				acc_test.append(accuracy_score(test_data[1], preds_test))
				#print('baseline acc test', id, numberOfVariables, acc_test)
			print(acc, 'test', acc_test)
			if id not in acc_dict:
				acc_dict[id] = {}
			acc_dict[id][numberOfVariables] = (np.mean(acc), np.mean(acc_test))
		print('\n')

	pickle.dump(acc_dict, open('baselines_noisy.pkl','wb'))



if __name__ == '__main__':
	getBaseline()
	getNoisyBaseline()
	
	