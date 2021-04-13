#include <iostream>
#include <fstream>
using namespace std;

#include "gomeaP3.hpp"
#include "utils.hpp"

gomeaP3::gomeaP3(Config *config_): GOMEA(config_)
{
  prepareFolder(config->folder);
  initElitistFile(config->folder);

  maximumNumberOfGOMEAs = 1000000000;
  numberOfGOMEAs = 0;
  
  createProblemInstance(config->numberOfVariables, config, &problemInstance, config->problemInstancePath);

  sharedInformationInstance = new sharedInformation(config->maxArchiveSize);
}

gomeaP3::~gomeaP3()
{
  for (int i = 0; i < GOMEAs.size(); ++i)
    delete GOMEAs[i];

  delete problemInstance;
  delete sharedInformationInstance;
}

void gomeaP3::run()
{
  while(!checkTermination())
  {
    if (numberOfGOMEAs < maximumNumberOfGOMEAs)
      initializeNewGOMEA();

    GOMEAGenerationalSteps(numberOfGOMEAs-1);
  }
}

bool gomeaP3::checkTermination()
{
  int i;
  if (numberOfGOMEAs == maximumNumberOfGOMEAs)
  {
    for (i = 0; i < maximumNumberOfGOMEAs; i++)
    {
      if (!GOMEAs[i]->terminated)
        return false;
    }

    return true;
  }
  
  return false;
}

void gomeaP3::initializeNewGOMEA()
{
  PopulationP3 *newPopulation = NULL;
  int populationSize = 1; //P3 scheme
  newPopulation = new PopulationP3(config, problemInstance, sharedInformationInstance, numberOfGOMEAs, populationSize);

  GOMEAs.push_back(newPopulation);
  numberOfGOMEAs++;
}

void gomeaP3::GOMEAGenerationalSteps(int GOMEAIndex)
{
  while (true)
  {
    if(!GOMEAs[GOMEAIndex]->terminated)
    {
      GOMEAs[GOMEAIndex]->makeOffspring();

      GOMEAs[GOMEAIndex]->copyOffspringToPopulation();

      GOMEAs[GOMEAIndex]->currentPyramidLevel++;
    }
    else
      break;

    if (config->hillClimber == 3)
      break;
  }
}


