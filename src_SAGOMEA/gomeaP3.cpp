#include <iostream>
#include <fstream>
using namespace std;

#include "gomeaP3.hpp"
#include "utils.hpp"

gomeaP3::gomeaP3(Config *config_): GOMEA(config_)
{
  prepareFolder(config->folder);
  initElitistFile(config->folder);

  maximumNumberOfGOMEAs       = 1000000000;
  numberOfGOMEAs              = 0;
  
  createProblemInstance(config->numberOfVariables, config, &problemInstance, config->problemInstancePath);
  
  sharedInformationInstance = new sharedInformation(config->maxArchiveSize);
  
  initializePythonFunctions();
}

void gomeaP3::initializePythonFunctions()
{
  string pwd = filesystem::current_path().string();
  char moduleName[1000];
  sprintf(moduleName, "import sys; sys.path.insert(0, \"%s/py_src\")", pwd.c_str());  
  PyRun_SimpleString(moduleName);
  PyRun_SimpleString ("import sys; print (sys.path)");

  PyObject* module = PyImport_ImportModule("surrogateModel");
  if (module == NULL) {cout << "SurrogateModels module import failed!\n";}

  PyObject *surrogateModelClass = PyObject_GetAttrString(module, config->SurrogateModelClass.c_str());  
  if (surrogateModelClass == NULL) {cout << "Surrogate Model class import failed!\n";}

  PyObject *pargs  = Py_BuildValue("(s,i,s)", config->folder.c_str(), config->numberOfVariables, config->alphabet.c_str()); 
  PyObject *surrogateModelInstance  = PyEval_CallObject(surrogateModelClass, pargs);
  if (surrogateModelInstance == NULL) {cout << "Surrogate model constructor failed!\n";}

  sharedInformationInstance->surrogateFitnessEstimationFunction  = PyObject_GetAttrString(surrogateModelInstance, "surrogateFitness");
  if (sharedInformationInstance->surrogateFitnessEstimationFunction == NULL) {cout << "Surrogate Fitness Estimation function retrieval failed!\n";}   

  sharedInformationInstance->surrogateModelTraining  = PyObject_GetAttrString(surrogateModelInstance, "updateModel");
  if (sharedInformationInstance->surrogateModelTraining == NULL) {cout << "Surrogate Model Training function retrieval failed!\n";}   
}

gomeaP3::~gomeaP3()
{
  for (int i = 0; i < numberOfGOMEAs; ++i)
    delete GOMEAs[i];

  delete problemInstance;
  delete sharedInformationInstance;
}

void gomeaP3::run()
{
  while(!checkTermination())
  {
    double elitistFitness = -1e+308;
    if (!sharedInformationInstance->firstEvaluationEver)
     elitistFitness = sharedInformationInstance->elitist.fitness;
    
    if (numberOfGOMEAs < maximumNumberOfGOMEAs)
      initializeNewGOMEA();

    GOMEAGenerationalSteps(numberOfGOMEAs-1);

    double newElitistFitness = -1e+308;
    if (!sharedInformationInstance->firstEvaluationEver)
     newElitistFitness = sharedInformationInstance->elitist.fitness;

    if (newElitistFitness > elitistFitness)
      config->currentDelta = 1.0;
    else
      config->currentDelta *= config->delta;
    
  }
}

bool gomeaP3::checkTermination()
{
  int i;
  
  if (numberOfGOMEAs == maximumNumberOfGOMEAs)
  {
    for (i = 0; i < maximumNumberOfGOMEAs; i++)
    {
      if (!GOMEAs[i]->terminated)
        return false;
    }

    return true;
  }
  
  return false;
}

void gomeaP3::initializeNewGOMEA()
{
  PopulationP3 *newPopulation = NULL;

  int populationSize = 1; //P3 scheme
  newPopulation = new PopulationP3(config, problemInstance, sharedInformationInstance, numberOfGOMEAs, populationSize);

  GOMEAs.push_back(newPopulation);
  numberOfGOMEAs++;
}

void gomeaP3::GOMEAGenerationalSteps(int GOMEAIndex)
{
  while (true)
  {
    if(!GOMEAs[GOMEAIndex]->terminated)
    {
      GOMEAs[GOMEAIndex]->makeOffspring();

      GOMEAs[GOMEAIndex]->copyOffspringToPopulation();

      GOMEAs[GOMEAIndex]->currentPyramidLevel++;
    }
    else
      break;

    if (config->hillClimber == 3)
      break;
  }
}


