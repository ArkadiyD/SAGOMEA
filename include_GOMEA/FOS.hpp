#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <deque>
#include <random>
#include <bits/stdc++.h> 
using namespace std;

#include "Individual.hpp"
#include "utils.hpp"

class FOS
{	
public:
	vector<vector<int> > FOSStructure;	
	size_t numberOfVariables;
	vector<int> alphabetSize;
	vector<vector<int> > graph;
	
	vector<int> improvementCounters;
	vector<int> usageCounters;

	FOS(size_t numberOfVariables_, vector<int> &alphabetSize_): numberOfVariables(numberOfVariables_), alphabetSize(alphabetSize_)
	{}

	virtual ~FOS(){};

	size_t FOSSize()
	{
		return FOSStructure.size();
	}

	size_t FOSElementSize(int i)
	{
		return FOSStructure[i].size();
	}
	
	virtual void learnFOS(vector<Individual*> &population, vector<vector<int> > *VIG = NULL, mt19937 *rng = NULL) = 0;
	void shuffleFOS(vector<int> &indices, mt19937 *rng);
	void sortFOSAscendingOrder(vector<int> &indices);
	void sortFOSDescendingOrder(vector<int> &indices);
	void orderFOS(int orderingType, vector<int> &indices, mt19937 *rng);

};


class LTFOS: public FOS
{
private:
	vector<vector<double> > MI_Matrix;
	vector<vector<double> > S_Matrix;
	bool filtered;
	int similarityMeasure;
	int determineNearestNeighbour(int index, vector< vector< int > > &mpm);
	void computeMIMatrix(vector<Individual*> &population);
	void computeNMIMatrix(vector<Individual*> &population);
	void estimateParametersForSingleBinaryMarginal(vector<Individual*> &population, vector<size_t> &indices, size_t  &factorSize, vector<double> &result);

public:	
	LTFOS(size_t numberOfVariables_, vector<int> &alphabetSize_, int similarityMeasure, bool filtered=false);
	~LTFOS(){};

	void learnFOS(vector<Individual*> &population, vector<vector<int> > *VIG = NULL, mt19937 *rng = NULL);
};

void createFOSInstance(FOS **FOSInstance, size_t numberOfVariables, vector<int> &alphabetSize, int similarityMeasure);
