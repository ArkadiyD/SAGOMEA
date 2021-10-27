**SAGOMEA** is an efficient surrogate-assisted Evolutionary Algorithm for combinatorial (discrete) optimization problems with expensive fitness functions.
The paper: https://arxiv.org/abs/2104.08048

In this repository you can also find implementations of non-surrogate search algorithms, such as GOMEA, Local Search (LS), and Random Search (RS). 

#### Installation
1. ```pip install -r requirements.txt```
2. Set Python versions in *Makefile_SAGOMEA* and *Makefile_GOMEA* to the Python paths of your system
3. To compile **SAGOMEA**: ```make -f Makefile_SAGOMEA```
To compile **GOMEA**: ```make -f Makefile_GOMEA```


#### SAGOMEA
- Usage information is shown if ```./SAGOMEA --help``` is typed
- An example of how to run SAGOMEA with the default hyperparameters is specified in the function
**run_SAGOMEA** in the *run_algorithms.py* file
- You can specify a surrogate model type used by **SAGOMEA**:
1. Support Vector Regression (SVR)
2.  Random Forest (RF)
3. Gradient Boosting (Catboost Regressor)
4. Multilayer Perceptron (MLP)
- All surrogate models are defined in the file: [py_src/surrogateModel.py](py_src/surrogateModel.py)
- The recommended value of hyperparameter <img src="http://www.sciweavers.org/tex2img.php?eq=%24%5Ceta%24&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="$\eta$" width="17" height="17" /> is 0.999


#### Using a custom fitness function
To use **SAGOMEA** for optimizing your own fitness function, it needs to be specified in [py_src/fitnessFunctions.py](py_src/fitnessFunctions.py)
- Inherit a function class from the *DummyFitnessFunction* class
- If necessary, modify the constructor
- Specify *fitness(self, x)* function
- Note that a *logger* class instance should be used to save evaluated solutions (an example is shown in *DummyFitnessFunction*) .
- All obtained solutions during an optimization run along with their fitness values are stored in the file *folder/optimization.txt*

#### Running search algorithms in parallel
Parallel runs of SAGOMEA (or other search algorithms) can be done using *run_algorithms.py* script
- Specify *problems* and *algorithms* variables in *run_algorithms.py*. An example is provided in lines 329-330.
- For example, running ```python3 run_algorithms.py test 3600 5000 0 50 10``` would execute 50 runs (with ids from 0 to 49) of the specified search algorithm(s) on the specified search problem(s), with 3600 seconds time limit (per run), 5000 fitness evaluations (per run), using *test* root folder, and performing 10 runs in parallel.
