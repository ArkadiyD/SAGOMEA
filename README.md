**SAGOMEA** is an efficient surrogate-assisted Evolutionary Algorithm for combinatorial (discrete) optimization problems with expensive fitness functions.
The paper (please cite when using the algorithm): 
#### Installation
1. ```pip install -r requirements.txt```
2. Set Python versions in *Makefile_SAGOMEA* and *Makefile_GOMEA* to the Python paths of your system
3. To compile **SAGOMEA**: ```make -f Makefile_SAGOMEA```
To compile **GOMEA**: ```make -f Makefile_GOMEA```


#### SAGOMEA
- An example of how to run SAGOMEA with the default hyperparameters is specified in the function
**run_SAGOMEA** in the *run_algorithms.py* file
- You can specify a surrogate model type used by **SAGOMEA**:
1. Support Vector Regression (SVR)
2.  Random Forest (RF)
3. Gradient Boosting (Catboost Regressor)
4. Multilayer Perceptron (MLP)
- All surrogate models are defined in the file: [py_src/surrogateModel.py](py_src/surrogateModel.py)
- The recommended value of hyperparameter $\eta$ is 0.999


#### Using a custom fitness function
To use **SAGOMEA** for optimizing your own fitness function, it needs to be specified in [py_src/fitnessFunction.py](py_src/fitnessFunction.py)
- Inherit a function class from the *DummyFitnessFunction* class
- If necessary, modify the constructor
- Specify *fitness(self, x)* function. Note that a *logger* class instance can be used to save evaluated solutions.

