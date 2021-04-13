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
	solutionsArchive *evaluatedSolutions;
	bool firstEvaluationEver;
	double percentileThreshold;
	Pyramid *pyramid;
	
	sharedInformation(int maxArchiveSize)
	{
		numberOfEvaluations = 0;
		startTimeMilliseconds = getCurrentTimeStampInMilliSeconds();
		firstEvaluationEver = true;
		evaluatedSolutions = new solutionsArchive(maxArchiveSize);
		pyramid = new Pyramid();
		//all_graphs.resize(1);
	}

	~sharedInformation()
	{
		delete evaluatedSolutions;
		delete pyramid;
	}
};