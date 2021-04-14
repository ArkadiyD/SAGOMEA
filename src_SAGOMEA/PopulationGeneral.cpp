#include "PopulationGeneral.hpp"

PopulationGeneral::PopulationGeneral(Config *config_, Problem *problemInstance_, sharedInformation *sharedInformationPointer_, size_t populationIndex_, size_t populationSize_): 
    config(config_), 
    problemInstance(problemInstance_),
    sharedInformationPointer(sharedInformationPointer_),
    populationIndex(populationIndex_), 
    populationSize(populationSize_)
{
    terminated = false;
    numberOfGenerations = 0;
    averageFitness = 0.0;
    
    population.resize(populationSize);
    offspringPopulation.resize(populationSize);
    noImprovementStretches.resize(populationSize);
    
    vector<int> allGenes(config->numberOfVariables);
    iota(allGenes.begin(), allGenes.end(), 0);

    if (sharedInformationPointer->surrogateModelTrained == false)
    {
      int toGenerate = config->numberOfVariables;

      cout << "Generating random solutions!\n";
      for (int i =0; i < toGenerate; ++i)
      {
        Individual *solution = new Individual(config->numberOfVariables, config->alphabetSize);
        solution->randomInit(&config->rng);
        evaluateSolution(solution);
      }
     }

    if (sharedInformationPointer->solutionsWithRealEvaluations.size() >= 10)
    {
      cout << "re-training surrogate Model\n";
      trainSurrogateModel();
      //exit(0);
      cout << "surrogate Model updated!\n";
    }

    sharedInformationPointer->surrogateElitistFitness = -1e+308;
    sharedInformationPointer->minSurrogateFitness = 1e+308;
    sharedInformationPointer->surrogateFitnesses.clear();

    for (auto it = sharedInformationPointer->solutionsWithRealEvaluations.begin(); it != sharedInformationPointer->solutionsWithRealEvaluations.end(); ++it)
    {
      vector<char> genotype = *it;
      Individual solution(genotype,-1);
      evaluateSolution(&solution, 2);
      if (solution.surrogateFitness < sharedInformationPointer->minSurrogateFitness)
        sharedInformationPointer->minSurrogateFitness = solution.surrogateFitness;
      sharedInformationPointer->surrogateFitnesses.push_back(solution.surrogateFitness);
    }    

    setThreshold();
    
    for (size_t i = 0; i < populationSize; ++i)
    {
      noImprovementStretches[i] = 0;

      population[i] = new Individual(config->numberOfVariables, config->alphabetSize);
      population[i]->randomInit(&config->rng);
      //cout << "evaluating random solution!\n";
      evaluateSolution(population[i]);

      //cout << "starting hill climbing...\n";
      if (config->hillClimber == 1)
        hillClimberSingle(population[i]);
      else if (config->hillClimber == 2)
        hillClimberMultiple(population[i]);
      else if (config->hillClimber == 3)
        hillClimberMultiple(population[i]);
	
      offspringPopulation[i] = new Individual(config->numberOfVariables, config->alphabetSize);
      *offspringPopulation[i] = *population[i];      
    }
    //cout << "finished hill climbing...\n";

    createFOSInstance(&FOSInstance, config->numberOfVariables, config->alphabetSize, config->similarityMeasure);
}

void PopulationGeneral::setThreshold()
{
  sort(sharedInformationPointer->surrogateFitnesses.begin(), sharedInformationPointer->surrogateFitnesses.end());
  
  int toSelect = int(config->currentDelta*sharedInformationPointer->surrogateFitnesses.size());   
  if (sharedInformationPointer->surrogateFitnesses.size()==0)
    sharedInformationPointer->percentileThreshold = -1e+308;
  else if (toSelect < 0)
    sharedInformationPointer->percentileThreshold = sharedInformationPointer->surrogateFitnesses[0];
  else if (toSelect >= sharedInformationPointer->surrogateFitnesses.size())
    sharedInformationPointer->percentileThreshold = sharedInformationPointer->surrogateFitnesses[sharedInformationPointer->surrogateFitnesses.size()-1];
  else
    sharedInformationPointer->percentileThreshold = sharedInformationPointer->surrogateFitnesses[toSelect];
  
  evaluateSolutionSurrogateModel(&sharedInformationPointer->elitist);
  sharedInformationPointer->percentileThreshold = config->currentDelta*(sharedInformationPointer->elitist.surrogateFitness);
}

void PopulationGeneral::calculateAverageFitness()
{
	averageFitness = 0.0;
	for (size_t i = 0; i < populationSize; ++i)
		averageFitness += population[i]->fitness;
	averageFitness /= populationSize;
}

void PopulationGeneral::copyOffspringToPopulation()
{
  for(size_t i = 0; i < populationSize; i++)
  {
  	*population[i] = *offspringPopulation[i];
  }
}

void PopulationGeneral::tournamentSelection(int k, vector<Individual*> &population, vector<Individual*> &offspringPopulation)
{
  int populationSize = population.size();

  vector<int> indices(populationSize * k);
  for (int i = 0; i < k; ++i)
  {
    for (int j = 0; j < populationSize; ++j)
      indices[populationSize*i + j] = j;

    shuffle(indices.begin() + populationSize*i, indices.begin() + populationSize*(i+1), config->rng);
  }
  for (int i = 0; i < populationSize; i++)
  {
    int winnerInd = 0;
    double winnerFitness = -1e+308;

    for (int j = 0; j < k; j++)
    {
      int challengerInd = indices[k*i+j];
      double challengerFitness = population[challengerInd]->fitness;
      //cout << i << " " << j << " " << challengerInd << endl;
      if (challengerFitness > winnerFitness)
      {
        winnerInd = challengerInd;
        winnerFitness = challengerFitness;
      }
    }

    *offspringPopulation[i] = *population[winnerInd];
  }
  // for (int i = 0; i < populationSize; i++)
  //   cout << i << " " << *population[i] << endl;

  //cout << endl;
}

void PopulationGeneral::hillClimberSingle(Individual *solution)
{
	vector<int> positions(config->numberOfVariables);
  iota(positions.begin(), positions.end(), 0);   

  shuffle(positions.begin(), positions.end(), config->rng);

  for (int j = 0; j < positions.size(); ++j)
  {
    int curPos = positions[j];
    char curValue = solution->genotype[curPos];

  	for (char k = 0; k < config->alphabetSize[curPos]; ++k)
  	{
    	if (k == curValue)
      		continue;

    	Individual backup = *solution;  
    	vector<int> touchedGenes(1, curPos);

    	solution->genotype[curPos] = k;

    	evaluateSolution(solution);

    	if (compareSolutions(solution, &backup) <= 0)
      		*solution = backup;
    }
  }
}

void PopulationGeneral::hillClimberMultiple(Individual *solution)
{
	vector<int> positions(config->numberOfVariables);
	iota(positions.begin(), positions.end(), 0);

	while (true)
	{
	  bool solutionImproved = false;

	  shuffle(positions.begin(), positions.end(), config->rng);

	  for (int j = 0; j < positions.size(); ++j)
	  {
	    int curPos = positions[j];
	    char curValue = solution->genotype[curPos];

	    for (char k = 0; k < config->alphabetSize[curPos]; ++k)
	    {
	      if (k == curValue)
	        continue;

	      Individual backup = *solution;  
	      vector<int> touchedGenes(1, curPos);

	      solution->genotype[curPos] = k;

	      evaluateSolution(solution);

        if (compareSolutions(solution, &backup) > 0)
	        solutionImproved = true;
	      else
	        *solution = backup;
	    }
	  }

	  if (!solutionImproved)
	    break;
	}
}


bool PopulationGeneral::FI(size_t offspringIndex, Individual *backup)
{
  vector<int> FOSIndices;
  FOSInstance->orderFOS(config->orderFOS, FOSIndices, &config->rng); 

  bool solutionHasChanged = 0;

  for (size_t i = 0; i < FOSInstance->FOSSize(); i++)
  {
    int ind = FOSIndices[i];

    if (FOSInstance->FOSElementSize(ind) == 0 || FOSInstance->FOSElementSize(ind) == config->numberOfVariables)
      continue;

    vector<int> touchedGenes;      
    bool donorEqualToOffspring = true;
    for(size_t j = 0; j < FOSInstance->FOSElementSize(ind); j++)
    {
      int variableFromFOS = FOSInstance->FOSStructure[ind][j];
      offspringPopulation[offspringIndex]->genotype[variableFromFOS] = sharedInformationPointer->elitist.genotype[variableFromFOS];
      touchedGenes.push_back(variableFromFOS);
      if (backup->genotype[variableFromFOS] != offspringPopulation[offspringIndex]->genotype[variableFromFOS])
        donorEqualToOffspring = false;
    }

    if (!donorEqualToOffspring)
    {
      evaluateSolution(offspringPopulation[offspringIndex]);

      if (compareSolutions(offspringPopulation[offspringIndex], backup) > 0)
      {
        *backup = *offspringPopulation[offspringIndex];
        solutionHasChanged = true;
      }
      else
      {
        *offspringPopulation[offspringIndex] = *backup;
      }
    }
    if (solutionHasChanged)
      break;
  }

  if (!solutionHasChanged)
  {
    *offspringPopulation[offspringIndex] = sharedInformationPointer->elitist;
  }

  return solutionHasChanged;
}


void PopulationGeneral::evaluateSolution(Individual *solution, int doSurrogateEvaluation)
{  
  checkTimeLimit();
  //cout << *solution << endl;
  solution->realFitnessCalculated = false;
  solution->surrogateFitnessCalculated = false;
  solution->fitness = -1e+308;
  solution->surrogateFitness = -1e+308;
    
  if (sharedInformationPointer->surrogateModelTrained)
  {
    evaluateSolutionSurrogateModel(solution);
    
      if (doSurrogateEvaluation != 3)
      {
        if (solution->surrogateFitness <= sharedInformationPointer->percentileThreshold)
        {
          //cout << "doing only surrogate fitness!\n";
          return;
        }
      }
  }
  
  if (doSurrogateEvaluation == 2) //surrogate only
    return;

  
  if (sharedInformationPointer->surrogateModelTrained && sharedInformationPointer->solutionsWithRealEvaluations.find(solution->genotype) == sharedInformationPointer->solutionsWithRealEvaluations.end())
  {
    evaluateSolution(solution,2);
    sharedInformationPointer->surrogateFitnesses.push_back(solution->surrogateFitness);
  }

  sharedInformationPointer->solutionsWithRealEvaluations.insert(solution->genotype);

  problemInstance->calculateFitness(solution);
  
  solution->realFitnessCalculated = true;
 
  sharedInformationPointer->numberOfEvaluations = problemInstance->getEvals();

  updateElitistAndCheckVTR(solution);

  if (sharedInformationPointer->numberOfEvaluations >= config->maxEvaluations)
  {
    cout << "Max evals limit reached! Terminating...\n";
    throw customException("max evals");
  }
  
}

void PopulationGeneral::checkTimeLimit()
{
  if (getMilliSecondsRunningSinceTimeStamp(sharedInformationPointer->startTimeMilliseconds) > config->timelimitSeconds*1000)
  {
    cout << "TIME LIMIT REACHED!" << endl;
    throw customException("time");
  }
}

void PopulationGeneral::updateElitistAndCheckVTR(Individual *solution)
{
  if (!solution->realFitnessCalculated)
    return;

  if (sharedInformationPointer->firstEvaluationEver || (solution->fitness > sharedInformationPointer->elitist.fitness))
  {
    sharedInformationPointer->elitistSolutionHittingTimeMilliseconds = getMilliSecondsRunningSinceTimeStamp(sharedInformationPointer->startTimeMilliseconds);
    sharedInformationPointer->elitistSolutionHittingTimeEvaluations = sharedInformationPointer->numberOfEvaluations;

    sharedInformationPointer->elitist = *solution;

    /* Check the VTR */
    if (solution->fitness >= config->vtr)
    {
      writeElitistSolutionToFile(config->folder, sharedInformationPointer->elitistSolutionHittingTimeEvaluations, sharedInformationPointer->elitistSolutionHittingTimeMilliseconds, solution);
      cout << "VTR HIT!\n";
      throw customException("vtr");
    }
  
    writeElitistSolutionToFile(config->folder, sharedInformationPointer->elitistSolutionHittingTimeEvaluations, sharedInformationPointer->elitistSolutionHittingTimeMilliseconds, solution);
    
    config->currentDelta = 1.0;
  }


  sharedInformationPointer->firstEvaluationEver = false;
}

int PopulationGeneral::compareSolutions(Individual *x, Individual*y)
{
  if (x->realFitnessCalculated && y->realFitnessCalculated)
  {
    if (x->fitness > y->fitness)
      return 1;
    if (x->fitness == y->fitness)
      return 0;
    return -1;
  }
  else
  {
    if (!x->surrogateFitnessCalculated)
      evaluateSolution(x, 2);

    if (!y->surrogateFitnessCalculated)
      evaluateSolution(y, 2);
    
    assert(x->surrogateFitnessCalculated && y->surrogateFitnessCalculated);

    if (x->surrogateFitness > y->surrogateFitness)
      return 1;
    if (x->surrogateFitness == y->surrogateFitness)
      return 0;
    return -1;
  }
}

void PopulationGeneral::trainSurrogateModel()
{
  PyObject *arglist = NULL;

  PyObject *result = PyEval_CallObject(sharedInformationPointer->surrogateModelTraining, arglist);

  if (result == NULL) {cout << "Surrogate model training failed!\n";exit(0);}
  sharedInformationPointer->surrogateModelTrained = true;
}

void PopulationGeneral::evaluateSolutionSurrogateModel(Individual *solution)
{  
  if (!sharedInformationPointer->surrogateModelTrained)
  {
    solution->surrogateFitness = 0.0;
    return;    
  }

  checkTimeLimit();
  PyObject *pySolution = PyTuple_New(config->numberOfVariables);
  for (Py_ssize_t i = 0; i < config->numberOfVariables; i++)
  {
      PyTuple_SET_ITEM(pySolution, i, PyLong_FromLong((int)solution->genotype[i]));
  }
  PyObject *arglist = Py_BuildValue("(O)", pySolution);
  PyObject *result;
  int goodFitness;
  double fitnessValue;
  result = PyEval_CallObject(sharedInformationPointer->surrogateFitnessEstimationFunction, arglist);
  if (result == NULL) {cout << "Fitness calculation failed!\n";}

  solution->surrogateFitness = PyFloat_AsDouble(result);
  solution->surrogateFitnessCalculated = true;
  
  if (solution->surrogateFitness > sharedInformationPointer->surrogateElitistFitness)
    sharedInformationPointer->surrogateElitistFitness = solution->surrogateFitness;

  Py_DECREF(result);
  Py_DECREF(arglist);
  Py_DECREF(pySolution);
}

