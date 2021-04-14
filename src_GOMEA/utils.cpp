#include "utils.hpp"

void prepareFolder(string &folder)
{
	cout << "preparing folder: " << folder << endl;
	if (filesystem::exists(folder))
	{
		filesystem::remove_all(folder);
	}
	filesystem::create_directories(folder);
	filesystem::create_directories(folder + "/fos");
	filesystem::create_directories(folder + "/populations");
}

void initElitistFile(string &folder)
{
	ofstream outFile(folder + "/elitists.txt", ofstream::out);
	if (outFile.fail())
	{
		cerr << "Problems with opening file " << folder + "/elitists.txt!\n";
		exit(0);
	}
	
	outFile << "#Evaluations " << "Time,millisec. " << "Fitness " << "Solution" << endl;
	outFile.close();
}

void writeElitistSolutionToFile(string &folder, long long numberOfEvaluations, long long time, Individual *solution)
{
	ofstream outFile(folder + "/elitists.txt", ofstream::app);
	if (outFile.fail())
	{
		cerr << "Problems with opening file " << folder + "/elitists.txt!\n";
		exit(0);
	}

	outFile << (int)numberOfEvaluations << " " << time << " " <<  fixed << setprecision(6) << solution->fitness << " ";
	for (size_t i = 0; i < solution->genotype.size(); ++i)
		outFile << +solution->genotype[i];
	outFile << endl;

	outFile.close();
}

Pyramid::~Pyramid()
{
	for (int i = 0; i < levels.size(); ++i)
	{
		for (int j = 0; j < levels[i].size(); ++j)
			delete levels[i][j];
	}
}

bool Pyramid::checkAlreadyInPyramid(vector<char> &genotype)
{
	if (find(hashset.begin(), hashset.end(), genotype) != hashset.end())
		return true;

	return false;
}

bool Pyramid::insertSolution(int level, vector<char> &genotype, double fitness)
{
	//cout << "Inserting solution " << level << endl;
	
	if (!checkAlreadyInPyramid(genotype))
	{
		if (level == levels.size())
		{
			vector<Individual*> newLevel;
			levels.push_back(newLevel);
		}
		
		Individual *newSolution = new Individual(genotype, fitness);
		levels[level].push_back(newSolution);

		hashset.insert(genotype);

		return true;
	}

	return false;
}


