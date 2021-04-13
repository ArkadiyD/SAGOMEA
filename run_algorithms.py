import numpy as np
import os
import argparse
from copy import deepcopy
import time
import sys
import shutil
import subprocess
import pickle
import gc

sys.path.append('py_src')
import fitnessFunctions

from ConfigSpace.hyperparameters import CategoricalHyperparameter
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario

parser = argparse.ArgumentParser(description='parse parameters')
parser.add_argument('folder', metavar='folder', default='test')
parser.add_argument('timeLimit', metavar='timeLimit', type=int, default=3600)
parser.add_argument('evalsLimit', metavar='evalsLimit', type=int, default=1000)
parser.add_argument('firstRun', metavar='firstRun', type=int, default=0)
parser.add_argument('nRuns', metavar='nRuns', type=int, default=10)
parser.add_argument('nProc', metavar='nProc', type=int, default=4)

args = parser.parse_args()
print(args)

def alphabetSize(problem, dim, instance):
	if problem == 'Ensembling2':
		return '2'
	elif problem == 'Ensembling5':
		return '5'
	elif problem == 'Ensembling10':
		return '10'

	elif problem == 'EnsemblingNoisy2':
		return '2'
	elif problem == 'EnsemblingNoisy5':
		return '5'
	elif problem == 'EnsemblingNoisy10':
		return '10'
	
	else:
		print('problem not implemented!')
		exit(0)

def getInstance(problem, dim, run):
	if 'Ensembling' in problem:
		if run < 10:
			return '146822'
		elif run < 20:
			return '43'
		elif run < 30:
			return '9960'
		elif run < 40:
			return '3917'
		elif run < 50:
			return '145945'
	else:
		print('problem not implemented!')
		exit(0)

def alphabetToList(alphabet, numberOfVariables):
	if alphabet.isnumeric():
		return [int(alphabet) for i in range(numberOfVariables)]
	file = open(alphabet,'r')
	alphabetSizes = file.readline().split(' ')
	file.close()

	return [int(alphabetSizes[i]) for i in range(numberOfVariables)]

def createAndCleanFolder(folder):
	os.makedirs(folder, exist_ok=True)
	files = os.listdir(folder)
	f = 'collectedData.txt'
	if f in files:
		file = open(folder+'/'+f,'r')
		lines = file.readlines()
		print(folder, len(lines))

	for f in files:
		f = folder+'/'+f

		if os.path.isfile(f):
			os.remove(f)
		else:
			shutil.rmtree(f)

	return True


def run_RS(function, problem, L, alphabet, cur_folder, randomSeed, instanceFile=''):
 
	alphabetSizes = alphabetToList(alphabet, L)

	print(alphabetSizes)

	np.random.seed(randomSeed)
	time_start = time.time()
	while function.logger.nEvals() < args.evalsLimit and time.time()-time_start < args.timeLimit:
		print('#evals',function.logger.nEvals(), 'time', time.time()-time_start)
		np.random.seed(int(time.time())+randomSeed)
		solution = []
		for i in range(L):
			solution.append(np.random.randint(0,alphabetSizes[i]))
		solution = np.array(solution)
		print(solution)
		fitness = function.fitness(solution)
		
def run_SMAC(function, problem, L, alphabet, cur_folder, randomSeed, instanceFile=''):
	
	alphabetSizes = alphabetToList(alphabet, L)

	np.random.seed(randomSeed)

	# Build Configuration Space which defines all parameters and their ranges
	cs = ConfigurationSpace()
	hyperparameters = []
	for i in range(L):
		options = [str(j) for j in range(alphabetSizes[i])]
		hyperparameters.append(CategoricalHyperparameter("x%d"%i, options, default_value=options[np.random.randint(len(options))]))
	print(hyperparameters)
	cs.add_hyperparameters(hyperparameters)

	# Scenario object
	scenario = Scenario({'run_obj': 'quality',  # we optimize quality (alternatively runtime)
						 'runcount-limit': args.evalsLimit,  # max. number of function evaluations; for this example set to a low number
						 'wallclock_limit': args.timeLimit,
						 'cs': cs,  # configuration space
						 'deterministic': "true",
						 'output_dir': cur_folder,
						 })
	smacInstance = SMAC4HPO(scenario=scenario,
					rng=randomSeed,
					initial_design=smac.initial_design.latin_hypercube_design.LHDesign,
					tae_runner=getattr(function, 'neg_fitness'))
	smacInstance.optimize()


from hyperopt import hp
from hyperopt import fmin, tpe, space_eval

def run_TPE(function, problem, L, alphabet, cur_folder, randomSeed, instanceFile=''):
	
	alphabetSizes = alphabetToList(alphabet, L)

	np.random.seed(randomSeed)
	
	# define a search space
	space = []
	for i in range(L):
		options = [str(j) for j in range(alphabetSizes[i])]
		space.append(hp.choice('x%d'%i, options=options))

	print(space)
	best = fmin(fn=getattr(function, 'neg_fitness'), space=space, algo=tpe.suggest, max_evals=args.evalsLimit)
	print(best)

def run_GOMEA(function, problem, L, alphabetSizes, cur_folder, randomSeed, instanceFile=''):
	functionName = function.__class__.__name__
	subprocess.call(['./GOMEA', '--L=%d'%L, '--timeLimit=%d'%args.timeLimit, '--maxEvals=%d'%args.evalsLimit, 
		'--functionName=%s'%functionName, '--instance=%s'%function.filename, '--alphabet=%s'%alphabetSizes,
		'--folder=%s'%cur_folder, '--seed=%d'%randomSeed,
		'--hillClimber=0'])

def run_LS(function, problem, L, alphabetSizes, cur_folder, randomSeed, instanceFile=''):
	functionName = function.__class__.__name__
	print(functionName)
	subprocess.call(['./GOMEA', '--L=%d'%L, '--timeLimit=%d'%args.timeLimit, '--maxEvals=%d'%args.evalsLimit, 
		'--functionName=%s'%functionName, '--instance=%s'%function.filename, '--alphabet=%s'%alphabetSizes,
		'--folder=%s'%cur_folder, '--seed=%d'%randomSeed,
		'--hillClimber=3'])

def run_SAGOMEA_SVR_0999(function, problem, L, alphabetSizes, cur_folder, randomSeed, instanceFile=''):
	functionName = function.__class__.__name__
	print(functionName)
	subprocess.call(['./SAGOMEA', '--L=%d'%L, '--timeLimit=%d'%args.timeLimit, '--maxEvals=%d'%args.evalsLimit, 
		'--functionName=%s'%functionName, '--instance=%s'%function.filename, '--alphabet=%s'%alphabetSizes,
		'--folder=%s'%cur_folder, '--seed=%d'%randomSeed,
		'--hillClimber=0', 
		'--SurrogateModelClass=surrogateModelSVR', '--delta=0.999'])

def run_SAGOMEA_SVR_099(function, problem, L, alphabetSizes, cur_folder, randomSeed, instanceFile=''):
	functionName = function.__class__.__name__
	print(functionName)
	subprocess.call(['./SAGOMEA', '--L=%d'%L, '--timeLimit=%d'%args.timeLimit, '--maxEvals=%d'%args.evalsLimit, 
		'--functionName=%s'%functionName, '--instance=%s'%function.filename, '--alphabet=%s'%alphabetSizes,
		'--folder=%s'%cur_folder, '--seed=%d'%randomSeed,
		'--hillClimber=0', 
		'--SurrogateModelClass=surrogateModelSVR', '--delta=0.99'])

def run_SAGOMEA_RF_0999(function, problem, L, alphabetSizes, cur_folder, randomSeed, instanceFile=''):
	functionName = function.__class__.__name__
	print(functionName)
	subprocess.call(['./SAGOMEA', '--L=%d'%L, '--timeLimit=%d'%args.timeLimit, '--maxEvals=%d'%args.evalsLimit, 
		'--functionName=%s'%functionName, '--instance=%s'%function.filename, '--alphabet=%s'%alphabetSizes,
		'--folder=%s'%cur_folder, '--seed=%d'%randomSeed,
		'--hillClimber=0', 
		'--SurrogateModelClass=surrogateModelRF', '--delta=0.999'])

def run_SAGOMEA_RF_099(function, problem, L, alphabetSizes, cur_folder, randomSeed, instanceFile=''):
	functionName = function.__class__.__name__
	print(functionName)
	subprocess.call(['./SAGOMEA', '--L=%d'%L, '--timeLimit=%d'%args.timeLimit, '--maxEvals=%d'%args.evalsLimit, 
		'--functionName=%s'%functionName, '--instance=%s'%function.filename, '--alphabet=%s'%alphabetSizes,
		'--folder=%s'%cur_folder, '--seed=%d'%randomSeed,
		'--hillClimber=0', 
		'--SurrogateModelClass=surrogateModelRF', '--delta=0.99'])

def run_SAGOMEA_Catboost_0999(function, problem, L, alphabetSizes, cur_folder, randomSeed, instanceFile=''):
	functionName = function.__class__.__name__
	print(functionName)
	subprocess.call(['./SAGOMEA', '--L=%d'%L, '--timeLimit=%d'%args.timeLimit, '--maxEvals=%d'%args.evalsLimit, 
		'--functionName=%s'%functionName, '--instance=%s'%function.filename, '--alphabet=%s'%alphabetSizes,
		'--folder=%s'%cur_folder, '--seed=%d'%randomSeed,
		'--hillClimber=0', 
		'--SurrogateModelClass=surrogateModelCatboost', '--delta=0.999'])

def run_SAGOMEA_Catboost_099(function, problem, L, alphabetSizes, cur_folder, randomSeed, instanceFile=''):
	functionName = function.__class__.__name__
	print(functionName)
	subprocess.call(['./SAGOMEA', '--L=%d'%L, '--timeLimit=%d'%args.timeLimit, '--maxEvals=%d'%args.evalsLimit, 
		'--functionName=%s'%functionName, '--instance=%s'%function.filename, '--alphabet=%s'%alphabetSizes,
		'--folder=%s'%cur_folder, '--seed=%d'%randomSeed,
		'--hillClimber=0', 
		'--SurrogateModelClass=surrogateModelCatboost', '--delta=0.99'])

def run_SAGOMEA_NeuralNet_0999(function, problem, L, alphabetSizes, cur_folder, randomSeed, instanceFile=''):
	functionName = function.__class__.__name__
	print(functionName)
	subprocess.call(['./SAGOMEA', '--L=%d'%L, '--timeLimit=%d'%args.timeLimit, '--maxEvals=%d'%args.evalsLimit, 
		'--functionName=%s'%functionName, '--instance=%s'%function.filename, '--alphabet=%s'%alphabetSizes,
		'--folder=%s'%cur_folder, '--seed=%d'%randomSeed,
		'--hillClimber=0', 
		'--SurrogateModelClass=surrogateModelNeuralNet', '--delta=0.999'])

def run_SAGOMEA_NeuralNet_099(function, problem, L, alphabetSizes, cur_folder, randomSeed, instanceFile=''):
	functionName = function.__class__.__name__
	print(functionName)
	subprocess.call(['./SAGOMEA', '--L=%d'%L, '--timeLimit=%d'%args.timeLimit, '--maxEvals=%d'%args.evalsLimit, 
		'--functionName=%s'%functionName, '--instance=%s'%function.filename, '--alphabet=%s'%alphabetSizes,
		'--folder=%s'%cur_folder, '--seed=%d'%randomSeed,
		'--hillClimber=0', 
		'--SurrogateModelClass=surrogateModelNeuralNet', '--delta=0.99'])

import multiprocessing
from multiprocessing import Pool, Process
from types import SimpleNamespace 

def run_algorithm(cur_args):
	cur_args = SimpleNamespace(**cur_args)
	print('running', cur_args)
	basic_folder = args.folder+'/'+cur_args.problem+'_'+str(cur_args.alphabet.split('/')[-1])+'/'+str(cur_args.L)+ '/'+cur_args.algorithm
	cur_folder = basic_folder + '/'+'run%d' % cur_args.run
	toRun = createAndCleanFolder(cur_folder)
	if not toRun:
		return

	randomSeed = 42 + cur_args.run
	instance = getInstance(cur_args.problem, cur_args.L, cur_args.run)
	if cur_args.algorithm == 'GOMEA': #Standard GOMEA-P3 algorithm
		algorithmRunner = run_GOMEA
	elif cur_args.algorithm == 'LS': #Local Search algorithm
		algorithmRunner = run_LS
	elif cur_args.algorithm == 'RS': #Random Search algorithm
		algorithmRunner = run_RS
	elif cur_args.algorithm == 'SMAC': #SMAC optimizer
		algorithmRunner = run_SMAC
	elif cur_args.algorithm == 'TPE': #Hyperopt (TPE) optimizer
		algorithmRunner = run_TPE
	elif cur_args.algorithm == 'COMBO': #COMBO optimizer
		algorithmRunner = run_COMBO
	
	elif cur_args.algorithm == 'SAGOMEA_SVR0999': #SAGOMEA-P3 with SVR surrogate model, eta=0.999
		algorithmRunner = run_SAGOMEA_SVR_0999
	elif cur_args.algorithm == 'SAGOMEA_SVR099': #SAGOMEA-P3 with SVR surrogate model, eta=0.99
		algorithmRunner = run_SAGOMEA_SVR_099
	
	elif cur_args.algorithm == 'SAGOMEA_RF0999': #SAGOMEA-P3 with RF surrogate model, eta=0.999
		algorithmRunner = run_SAGOMEA_RF_0999
	elif cur_args.algorithm == 'SAGOMEA_RF099': #SAGOMEA-P3 with RF surrogate model, eta=0.99
		algorithmRunner = run_SAGOMEA_RF_099

	elif cur_args.algorithm == 'SAGOMEA_Catboost0999': #SAGOMEA-P3 with Catboost surrogate model, eta=0.999
		algorithmRunner = run_SAGOMEA_Catboost_0999
	elif cur_args.algorithm == 'SAGOMEA_Catboost099': #SAGOMEA-P3 with Catboost surrogate model, eta=0.99
		algorithmRunner = run_SAGOMEA_Catboost_099
	
	elif cur_args.algorithm == 'SAGOMEA_NeuralNet0999': #SAGOMEA-P3 with MLP surrogate model, eta=0.999
		algorithmRunner = run_SAGOMEA_NeuralNet_0999
	elif cur_args.algorithm == 'SAGOMEA_NeuralNet099': #SAGOMEA-P3 with MLP surrogate model, eta=0.99
		algorithmRunner = run_SAGOMEA_NeuralNet_099

	else:
		print('algorithm not implemented!')
		exit(0)

	function = getattr(fitnessFunctions, cur_args.problem)(cur_folder, instance, cur_args.L, cur_args.alphabet)
	print(function, instance)
	algorithmRunner(function, cur_args.problem, cur_args.L, cur_args.alphabet, cur_folder, randomSeed, instance)
	del function
	
	gc.collect()

def runInParallel(algorithms, problems, dims):
	args_tuples = []

	for algorithm in algorithms:
		for problem in problems:
			for L in dims:
				instance = getInstance(problem, L, 0)
				alphabet = alphabetSize(problem, L, instance)
				basic_folder = args.folder+'/'+problem+'_'+str(alphabet.split('/')[-1])+'/'+str(L)+ '/'+algorithm
				print(basic_folder)
				createAndCleanFolder(basic_folder)		

				for run in range(args.firstRun, args.nRuns):
					cur_args = {'algorithm':algorithm, 'problem':problem, 'L':L, 'run':run, 'alphabet':alphabet}
					args_tuples.append(cur_args)	
	print(args_tuples)
	print('total to run:',len(args_tuples), 'procs:', args.nProc)
	
	if args.nProc > 1:
		pool = Pool(processes=int(args.nProc))
		pool.map(run_algorithm, args_tuples)
		pool.close()
		pool.join()
	else:
		for args_tuple in args_tuples:
			run_algorithm(args_tuple)


dims={'Ensembling5':[250]}

algorithms=['GOMEA','LS']


for problem in list(dims.keys()):
	print(problem)
	runInParallel(algorithms, [problem], dims[problem])
