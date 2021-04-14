#include "PopulationP3.hpp"


PopulationP3::~PopulationP3()
{
  for (size_t i = 0; i < populationSize; ++i)
  {
    delete population[i];
    delete offspringPopulation[i];
  }

  delete FOSInstance;
}

void PopulationP3::makeOffspring()
{
  if (config->hillClimber == 3)
    return;
  
  int someSolutionsImproved = false;

  for (int i = 0; i < populationSize; ++i)
  {
    if (noImprovementStretches[i] == 0) // solution improved or zero level
    {
      sharedInformationPointer->pyramid->insertSolution(currentPyramidLevel, population[i]->genotype, population[i]->fitness);
      someSolutionsImproved = true;
    }
  }

  if (currentPyramidLevel == sharedInformationPointer->pyramid->levels.size() || someSolutionsImproved == false)
  {
    terminated = true;
    return;
  }

  vector<Individual*> populationForModelLearning(sharedInformationPointer->pyramid->levels[currentPyramidLevel].size());
  for (int i = 0; i < populationForModelLearning.size(); ++i)
  {
    populationForModelLearning[i] = new Individual(config->numberOfVariables, config->alphabetSize);
    *populationForModelLearning[i] = *sharedInformationPointer->pyramid->levels[currentPyramidLevel][i];
  }

  if (config->tournamentSelection)
    tournamentSelection(2, sharedInformationPointer->pyramid->levels[currentPyramidLevel], populationForModelLearning); //performs tournament selection and saves the winners to population array

  FOSInstance->learnFOS(populationForModelLearning, NULL, &config->rng);
  
  for (int i = 0; i < populationForModelLearning.size(); ++i)
    delete populationForModelLearning[i];

  generateOffspring();

}

void PopulationP3::generateOffspring()
{
  for(size_t i = 0; i < populationSize; i++)
  {
      Individual backup = *population[i];  
      
      bool solutionHasChanged;
      solutionHasChanged = GOM(i, &backup);
      
      /* Phase 2 (Forced Improvement): optimal mixing with elitist solution */
      if (config->useForcedImprovements)
      {
        if ((!solutionHasChanged) || (noImprovementStretches[i] > (1+(log(populationSize)/log(10)))))
        {
          FI(i, &backup);
        }    
      }

    if (compareSolutions(offspringPopulation[i], population[i]) != 1)
      noImprovementStretches[i]++;
    else
      noImprovementStretches[i] = 0;
  }
}

bool PopulationP3::GOM(size_t offspringIndex, Individual *backup)
{
  size_t donorIndex;
  bool solutionHasChanged = false;
  bool thisIsTheElitistSolution = *offspringPopulation[offspringIndex] == sharedInformationPointer->elitist;//(sharedInformationPointer->elitistSolutionpopulationIndex == populationIndex) && (sharedInformationPointer->elitistSolutionOffspringIndex == offspringIndex);
  
  *offspringPopulation[offspringIndex] = *population[offspringIndex];

  vector<int> FOSIndices;
  FOSInstance->orderFOS(config->orderFOS, FOSIndices, &config->rng); 

  vector<int> donorIndices(sharedInformationPointer->pyramid->levels[currentPyramidLevel].size());
  iota(donorIndices.begin(), donorIndices.end(), 0);

  for (size_t i = 0; i < FOSInstance->FOSSize(); i++)
  {
    int ind = FOSIndices[i];

    if (FOSInstance->FOSElementSize(ind) == 0 || FOSInstance->FOSElementSize(ind) == config->numberOfVariables)
      continue;

    bool donorEqualToOffspring = true;
    int indicesTried = 0;

    while (donorEqualToOffspring && indicesTried < donorIndices.size())
    {
      int j = config->rng() % (donorIndices.size() - indicesTried);
      swap(donorIndices[indicesTried], donorIndices[indicesTried + j]);
      donorIndex = donorIndices[indicesTried];
      indicesTried++;
      
      if (offspringPopulation[offspringIndex]->genotype == sharedInformationPointer->pyramid->levels[currentPyramidLevel][donorIndex]->genotype)
        continue;

      vector<int> touchedGenes;
      for(size_t j = 0; j < FOSInstance->FOSElementSize(ind); j++)
      {
        int variableFromFOS = FOSInstance->FOSStructure[ind][j];      
        offspringPopulation[offspringIndex]->genotype[variableFromFOS] = sharedInformationPointer->pyramid->levels[currentPyramidLevel][donorIndex]->genotype[variableFromFOS];
        touchedGenes.push_back(variableFromFOS);

        if (backup->genotype[variableFromFOS] != offspringPopulation[offspringIndex]->genotype[variableFromFOS])
          donorEqualToOffspring = false;      
      }

      if (!donorEqualToOffspring)
      {
        evaluateSolution(offspringPopulation[offspringIndex]);

        // accept the change if this solution is not the elitist and the fitness is at least equally good (allows random walk in neutral fitness landscape)
        // however, if this is the elitist solution, only accept strict improvements, to avoid convergence problems
        
        if ((!thisIsTheElitistSolution && compareSolutions(offspringPopulation[offspringIndex], backup) >= 0) || 
          (thisIsTheElitistSolution && compareSolutions(offspringPopulation[offspringIndex], backup) == 1))         
        {       
          *backup = *offspringPopulation[offspringIndex];
          solutionHasChanged = true;
        }
        else
        {
          *offspringPopulation[offspringIndex] = *backup;
        }

      }

      if (!config->donorSearch) //if not exhaustive donor search then stop searching anyway
        break;
    }
  }
  return solutionHasChanged;
}


