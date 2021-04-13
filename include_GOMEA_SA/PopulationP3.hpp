#pragma once

#include <cmath>
#include <iostream> 
#include <vector>
#include <csignal>
using namespace std;

#include "Individual.hpp"
#include "Config.hpp"
#include "shared.hpp"
#include "problems.hpp"
#include "FOS.hpp"
#include "PopulationGeneral.hpp"

class PopulationP3: public PopulationGeneral
{
public:
	PopulationP3(Config *config_, Problem *problemInstance_, sharedInformation *sharedInformationPointer_, size_t GOMEAIndex_, size_t populationSize_):
		PopulationGeneral(config_, problemInstance_, sharedInformationPointer_, GOMEAIndex_, populationSize_){};	
	~PopulationP3();

	void makeOffspring();
	void generateOffspring();
	bool GOM(size_t offspringIndex, Individual *backup);
};