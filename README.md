**SAGOMEA** is an efficient surrogate-assisted Evolutionary Algorithms for combinatorial (discrete) optimization problems with expensive fitness functions

#### SAGOMEA


#### Using a custom fitness function
To use SAGOMEA for optimizing your own fitness function, it needs to be specified in [py_src/fitnessFunction.py](py_src/fitnessFunctions.py)
- Inherit a function class from the *DummyFitnessFunction* class
- If necessary, modify the constructor
- Specify *fitness(self, x)* function. Note that a *logger* class instance can be used to save evaluated solutions.

