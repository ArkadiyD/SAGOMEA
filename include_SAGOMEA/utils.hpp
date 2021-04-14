#pragma once

#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <exception>
using namespace std;

#include "Individual.hpp"

class customException: public exception
{
private:
    string message;

public:
    customException(string message_) : message(message_) { }
    const char * what () const throw ()
    {
        return message.c_str();
    }
};

struct archiveRecord
{
  bool isFound = false;
  double value = 0.0;
};

struct hashVector
{ 
	size_t operator()(const vector<char> &vec) const
	{ 
    	hash <char> hashChar; 
     	size_t hash_value = 0;
     	for (size_t i = 0; i < vec.size(); ++i)	
 	    	hash_value = hash_value*31 + hashChar(vec[i]); 
     	return hash_value; 
	} 
}; 

struct hash_int_vector
{ 
  size_t operator()(const vector<int> &vec) const
    { 
      size_t hash_value = 0;
        hash<int> hash_int;
        for(int i = 0; i < vec.size(); ++i)
          hash_value ^= hash_int(vec[i]);

        return hash_value; 
    } 
}; 

class solutionsArchive
{
	int maxArchiveSize;
public:
	solutionsArchive(int maxArchiveSize_): maxArchiveSize(maxArchiveSize_){};
	unordered_map<vector<char>, double, hashVector > archive;
	void checkAlreadyEvaluated(vector<char> &genotype, archiveRecord *result);
	void insertSolution(vector<char> &genotype, double fitness);
};

class Pyramid
{
public:
	vector<vector<Individual*> >levels;
	unordered_set<vector<char>, hashVector> hashset;
	bool checkAlreadyInPyramid(vector<char> &genotype);
	bool insertSolution(int level, vector<char> &genotype, double fitness);
	~Pyramid();
};

void prepareFolder(string &folder);
void initElitistFile(string &folder);
void writeElitistSolutionToFile(string &folder, long long numberOfEvaluations, long long time, Individual *solution);
void writeSolutionToFile(string &folder, string &filename, Individual &solution);
