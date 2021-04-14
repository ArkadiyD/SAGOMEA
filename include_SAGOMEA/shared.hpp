#pragma once

#include <vector>
#include <unordered_map>
using namespace std;

#include "utils.hpp"
#include "time.hpp"

struct sharedInformation
{
	double numberOfEvaluations, numberOfSurrogateEvaluations;
	long long startTimeMilliseconds;
	double elitistSolutionHittingTimeMilliseconds,
	       elitistSolutionHittingTimeEvaluations;

	Individual elitist;
	double surrogateElitistFitness, minSurrogateFitness;
	bool firstEvaluationEver;
	bool surrogateModelTrained;
	double percentileThreshold;
	vector<double> surrogateFitnesses;
	
	Pyramid *pyramid;
	PyObject *surrogateFitnessEstimationFunction, *surrogateModelTraining;
	
	unordered_set<vector<char >, hashVector > solutionsWithRealEvaluations;

	sharedInformation()
	{
		numberOfEvaluations = 0;
		startTimeMilliseconds = getCurrentTimeStampInMilliSeconds();
		firstEvaluationEver = true;
		pyramid = new Pyramid();
		surrogateModelTrained = false;
		surrogateElitistFitness = -1e+308;
		minSurrogateFitness = 1e+308;
	}

	~sharedInformation()
	{
		delete pyramid;
	}
};
