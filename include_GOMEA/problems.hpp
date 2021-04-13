#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <fstream>
#include <deque>
#include <algorithm>
#include <unordered_set>
#include <set>
#include <cassert>
#include <filesystem>
#include <Python.h>
using namespace std;

#include "Individual.hpp"
#include "utils.hpp"

class Config;
#include "Config.hpp"

class Problem
{
public:
	int numberOfVariables;
	
	Problem(){};
	virtual ~Problem(){};
	virtual void initializeProblem(Config *config, int numberOfVariables)=0;
	virtual double calculateFitness(Individual *solution)=0;
	virtual int getEvals(){return -1;};
};



class PythonFunction:public Problem
{
	string functionName, instancePath;
	PyObject *module, *functionClass, *functionInstance, *fitnessFunction, *getEvalsFunction;

public:
	PythonFunction(string functionName_, string instancePath_): functionName(functionName_), instancePath(instancePath_)
	{
		cout<<"creating Python Function " << functionName << endl;
	}
	~PythonFunction()
	{
		Py_DECREF(module);
		Py_DECREF(functionClass);
		Py_DECREF(functionInstance);
	}

	void initializeProblem(Config *config, int numberOfVariables_);
	double calculateFitness(Individual *solution);
	int getEvals();
};

void createProblemInstance(int numberOfVariables, Config *config, Problem **problemInstance, string &instancePath);

