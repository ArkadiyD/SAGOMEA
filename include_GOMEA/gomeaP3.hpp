#pragma once

#include <vector>
#include <unordered_map>
using namespace std;

#include "Config.hpp"
#include "PopulationP3.hpp"
#include "problems.hpp"
#include "shared.hpp"
#include "gomea.hpp"

class gomeaP3: public GOMEA
{
public:
	int maximumNumberOfGOMEAs;
	int basePopulationSize, numberOfGOMEAs;

	vector<PopulationP3*> GOMEAs;

	gomeaP3(Config *config_);
	~gomeaP3();
	
	void initializeNewGOMEA();
	bool checkTermination();
	void GOMEAGenerationalSteps(int GOMEAIndex);
	void run();
};