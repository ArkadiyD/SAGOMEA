#pragma once

#include <string> 
#include <iostream>
using namespace std;

#include "Config.hpp"
#include "Individual.hpp"
#include "shared.hpp"
#include "problems.hpp"

class GOMEA
{
public:
	Config *config;
	Problem *problemInstance = NULL;
	sharedInformation *sharedInformationInstance = NULL;
	
	GOMEA(Config *config_): config(config_){};
	virtual ~GOMEA(){};

	virtual void run() = 0;
	double readVTR();
};