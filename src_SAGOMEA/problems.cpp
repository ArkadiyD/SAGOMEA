#include "problems.hpp"

void createProblemInstance(int numberOfVariables, Config *config, Problem **problemInstance, string &instancePath)
{
    *problemInstance = new PythonFunction(config->functionName, instancePath);
    (*problemInstance)->initializeProblem(config, numberOfVariables);
}

void PythonFunction::initializeProblem(Config *config, int numberOfVariables_)
{
  numberOfVariables = numberOfVariables_;
  string pwd = filesystem::current_path().string();
  cout << "current_path: " << pwd << endl;
  char moduleName[1000];
  sprintf(moduleName, "import sys; sys.path.insert(0, \"%s/py_src\")", pwd.c_str());  
  cout << moduleName << endl;
  PyRun_SimpleString(moduleName);
  //PyRun_SimpleString ("import sys; print (sys.path)");

  PyObject* module = PyImport_ImportModule("fitnessFunctions");
  if (module == NULL) {cout << "Module import failed!\n";}

  functionClass = PyObject_GetAttrString(module, functionName.c_str());   /* fetch module.class */
  if (functionClass == NULL) {cout << "Class import failed!\n";}

  PyObject *pargs  = Py_BuildValue("(s,s,i,s)", config->folder.c_str(), instancePath.c_str(), config->numberOfVariables, config->alphabet.c_str());
  functionInstance  = PyEval_CallObject(functionClass, pargs);        /* call class(  ) */
  if (functionInstance == NULL) {cout << "Function init failed!\n";}

  fitnessFunction  = PyObject_GetAttrString(functionInstance, "fitness"); /* fetch bound method */
  if (fitnessFunction == NULL) {cout << "Fitness function retrieval failed!\n";}    

  getEvalsFunction  = PyObject_GetAttrString(functionInstance, "nEvals"); /* fetch bound method */
  if (getEvalsFunction == NULL) {cout << "Get evals function retrieval failed!\n";}   
};


double PythonFunction::calculateFitness(Individual *solution)
{
    PyObject *pySolution = PyTuple_New(numberOfVariables);
    for (Py_ssize_t i = 0; i < numberOfVariables; i++)
    {
        PyTuple_SET_ITEM(pySolution, i, PyLong_FromLong((int)solution->genotype[i]));
    }
    PyObject *arglist = Py_BuildValue("(O)", pySolution);
    PyObject *result = PyEval_CallObject(fitnessFunction, arglist);
    if (result == NULL) {cout << "Fitness calculation failed!\n";}

    solution->fitness = PyFloat_AsDouble(result);

    Py_DECREF(result);
    Py_DECREF(arglist);
    Py_DECREF(pySolution);

    return solution->fitness;
}

int PythonFunction::getEvals()
{
    PyObject *arglist = NULL;
    PyObject *result = PyEval_CallObject(getEvalsFunction, arglist);
    if (result == NULL) {cout << "Get evals function failed!\n";}

    int evals = PyLong_AsLong(result);
    //cout << "#evals:" << evals << endl;
    Py_DECREF(result);
    
    return evals;
}


