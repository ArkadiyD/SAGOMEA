#pragma once

#include <iostream> 
#include <vector>
#include <random>
using namespace std;

class Individual
{
public:
	size_t numberOfVariables;
	vector<int> alphabetSize;
	vector<char> genotype;
	double fitness;
	double surrogateFitness;
	bool realFitnessCalculated;
	bool surrogateFitnessCalculated;
	
	Individual() {};

	Individual(size_t numberOfVariables_, vector<int> &alphabetSize_): numberOfVariables(numberOfVariables_), alphabetSize(alphabetSize_)
	{
		genotype.resize(numberOfVariables_);
		fill(genotype.begin(), genotype.end(), 0);
		realFitnessCalculated = false;
		surrogateFitnessCalculated = false;
		surrogateFitness = -1e+308;
	}

	Individual(vector<char> &genotype_, double fitness_): fitness(fitness_)
	{
		numberOfVariables = genotype_.size();
		genotype.resize(numberOfVariables);
		copy(genotype_.begin(), genotype_.end(), genotype.begin());
		realFitnessCalculated = false;
		surrogateFitnessCalculated = false;
		surrogateFitness = -1e+308;
	}

	void randomInit(mt19937 *rng)
	{
		for (size_t i = 0; i < numberOfVariables; ++i)
		{
			genotype[i] = (*rng)() % alphabetSize[i];
		}
	}

	friend ostream & operator << (ostream &out, const Individual &individual);

	Individual& operator=(const Individual& other)
	{
		alphabetSize = other.alphabetSize;
		numberOfVariables = other.numberOfVariables;

		genotype = other.genotype;
		
		fitness = other.fitness;
		realFitnessCalculated = other.realFitnessCalculated;

		surrogateFitnessCalculated = other.surrogateFitnessCalculated;
		surrogateFitness = other.surrogateFitness;
		
		return *this;
	}

	bool operator==(const Individual& solutionB)
	{
    	for (size_t i = 0; i < numberOfVariables; ++i)
    	{
    		if (this->genotype[i] != solutionB.genotype[i])
    			return false;
    	}
    	return true;
	}
};

