#pragma once

#include <cmath>
#include <iostream> 
#include <vector>
using namespace std;

#include "Individual.hpp"
#include "Config.hpp"
#include "shared.hpp"
#include "problems.hpp"
#include "FOS.hpp"

class PopulationGeneral
{
public:
	Config *config;
	Problem *problemInstance;
	sharedInformation *sharedInformationPointer;
	size_t populationIndex;
	size_t populationSize;

	vector<Individual*> population;
	vector<Individual*> offspringPopulation;
	vector<int> noImprovementStretches;

	FOS *populationFOS;
	bool terminated;
	double averageFitness;
	size_t numberOfGenerations;
	FOS *FOSInstance = NULL;
	vector<vector<double> > matrix;

	PopulationGeneral(Config *config_, Problem *problemInstance_, sharedInformation *sharedInformationPointer_, size_t populationIndex_, size_t populationSize_);
	virtual ~PopulationGeneral(){};

	void tournamentSelection(int k, vector<Individual*> &population, vector<Individual*> &offspringPopulation);
	void hillClimberSingle(Individual *solution);	
	void hillClimberMultiple(Individual *solution);

	void calculateAverageFitness();	
	void copyOffspringToPopulation();
	void evaluateSolution(Individual *solution);
	bool GOM(size_t offspringIndex, Individual *backup);
	bool FI(size_t offspringIndex, Individual *backup);
	void updateElitistAndCheckVTR(Individual *solution);
	void checkTimeLimit();
	int compareSolutions(Individual *x, Individual*y);
};

