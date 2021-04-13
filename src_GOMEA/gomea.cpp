#include <gomea.hpp>

double GOMEA::readVTR()
{
  string filename = config->folder + "/vtr.txt";
  ifstream inFile;
  inFile.open(filename);

  string vtr_str;
  inFile >> vtr_str;
  double vtr = stod(vtr_str);

  inFile.close();

  return vtr;
}

