#pragma once

#include <string>
#include <cstdlib>
#include <iostream>
#include <random>
#include <unistd.h>
#include <getopt.h>

using namespace std;

#include "problems.hpp"
#include "FOS.hpp"

extern const double INF_FITNESS;

class Config
{
	void splitString(const string &str, vector<string> &splitted, char delim);
	bool isNumber(const string &s);

public:
	int printHelp = 0;

   	double eta = 0.999;
	double currentDelta = 1.0;
	string SurrogateModelClass;
   	double vtr = 1e+308;
	size_t 
	    orderFOS = 1,
	    numberOfVariables = 1;
	int similarityMeasure = 1,
		useForcedImprovements = 0;

	string folder = "test";
	string problemName,
		   FOSName;
	string problemInstancePath = "";

	long long timelimitSeconds = 3600,
			  randomSeed = 42;
	
	string alphabet;
	vector<int> alphabetSize;
	int maximumNumberOfGOMEAs  = 100,
		populationSize = 1, 
	    maxGenerations = 200;
	int hillClimber = 0,
	    donorSearch = 1,
	    tournamentSelection = 0;
    long long maxEvaluations = 1e+12;
    string functionName="";
    mt19937 rng;

	bool parseCommandLine(int argc, char **argv);
	void printUsage();
	void printOverview();
};
