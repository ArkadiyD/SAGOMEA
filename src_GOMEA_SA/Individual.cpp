#include "Individual.hpp"

ostream & operator << (ostream &out, const Individual &individual)
{
	for (size_t i = 0; i < individual.numberOfVariables; ++i)
		out << +individual.genotype[i];
	out << "real F calculated:" << individual.realFitnessCalculated << " | " << individual.fitness << " || surr F calculated:" << individual.surrogateFitnessCalculated << " | " << individual.surrogateFitness << endl;
	return out;
}