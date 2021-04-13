#include "Config.hpp"

const double INF_FITNESS=-1e+308;

void Config::splitString(const string &str, vector<string> &splitted, char delim)
{
    size_t current, previous = 0;
    current = str.find(delim);
    while (current != string::npos)
    {
        splitted.push_back(str.substr(previous, current - previous));
        previous = current + 1;
        current = str.find(delim, previous);
    }
    splitted.push_back(str.substr(previous, current - previous));
}

bool Config::isNumber(const string &str)
{
	return !str.empty() && all_of(str.begin(), str.end(), ::isdigit);
}

bool Config::parseCommandLine(int argc, char **argv)
{
  const struct option longopts[] =
  {
	{"help",        no_argument,         0, 'h'},    
    {"FI",          no_argument,         0, 'f'},
    {"saveEvals",   no_argument,         0, 's'},  
    {"donorSearch", no_argument,         0, 'd'},    
    {"tournamentSelection", no_argument, 0, 't'}, 
      
    {"maxEvals",    required_argument,   0, 'E'},  
    {"hillClimber", required_argument,   0, 'H'},    
	{"L",           required_argument,   0, 'L'},  
	{"seed",        required_argument,   0, 'S'},
	{"alphabet",    required_argument,   0, 'A'},
	{"instance",    required_argument,   0, 'I'},
	{"vtr",         required_argument,   0, 'V'},
	{"timeLimit",   required_argument,   0, 'T'},
	{"folder",      required_argument,   0, 'O'},
	{"orderFOS",    required_argument,   0, 'B'}, 
    {"similarityMeasure", required_argument,   0, 'Z'}, 
    {"functionName", required_argument,   0, 'N'}, 
               
    {0,             0,                   0,  0 }
  };


  int c, index;
  while ((c = getopt_long(argc, argv, "h::f::s::d::t::E::H::L::S::A::I::V::T::O::B::Z::N::", longopts, &index)) != -1)
  {
  	switch (c)
	{
		case 'f':
			useForcedImprovements = 1;
			break;
		case 's':
			saveEvaluations = 1;
			break;
		case 'h':
			printHelp = 1;
			break;
		case 'd':
			donorSearch = 1;
			break;
		case 't':
			tournamentSelection = 1;
			break;
		case 'E':
			maxEvaluations = atoll(optarg);
			break;
		case 'H':
			hillClimber = atoi(optarg);
			break;
		case 'L':
			numberOfVariables = atoi(optarg);
			break;
		case 'S':
			randomSeed = atoll(optarg);
			break;
		case 'A':
			alphabet = string(optarg);
			break;
		case 'I':
			problemInstancePath = string(optarg);
			break;
		case 'V':
			vtr = atof(optarg);
			break;
		case 'T':
			timelimitSeconds = atoi(optarg);
			break;		
		case 'O':
			folder= string(optarg);
			break;
		case 'B':
			orderFOS = atoi(optarg);
			break;
		case 'Z':
			similarityMeasure = atoi(optarg);
			break;
		case 'N':
			functionName = string(optarg);
			break;
		

		default:
			abort();
	}
  }

  alphabetSize.resize(numberOfVariables);

  if (atoi(alphabet.c_str()))
  	fill(alphabetSize.begin(), alphabetSize.end(), atoi(alphabet.c_str()));
  else
  {
  	ifstream in;
  	in.open(alphabet, ifstream::in);
  	for (int i = 0; i < numberOfVariables; ++i)
  	{
  		in >> alphabetSize[i];
  	}
  	in.close();
  }

  if (printHelp)
  {
  	printUsage();
  	exit(0);
  }
  return 1;
}

void Config::printUsage()
{
  cout << "Usage: GOMEA [-h] [-p] [-v] [-w] [-s] [-h] [-P] [-L] [-F] [-T] [-S]\n";
  cout << "   -h: Prints out this usage information.\n";
  cout << "   -f: Whether to use Forced Improvements.\n";
  cout << "   -p: Whether to use partial evaluations\n";
  cout << "   -w: Enables writing FOS contents to file\n";
  cout << "   -s: Enables saving all evaluations in archive\n";
  cout << endl;
  cout << "    P: Index of optimization problem to be solved (maximization). Default: 0\n";
  cout << "    L: Number of variables. Default: 1\n";
  cout << "    A: Alphabet size. Default: 2 (binary optimization)\n";  
  cout << "    C: GOM type. Default: 0 (LT). 1 - conditionalGOM with MI, 2 - conditionalGOM with predefined VIG\n";
  cout << "    O: Folder where GOMEA runs. Default: \"test\"\n";
  cout << "    T: timeLimit in seconds. Default: 1\n";
  cout << "    S: random seed. Default: 42\n";
}


void Config::printOverview()
{
  cout << "### Settings ######################################\n";
  cout << "#\n";
  cout << "# Use Forced Improvements : " << (useForcedImprovements ? "enabled" : "disabled")  << endl;
  cout << "# Save all evaluations : " << (saveEvaluations ? "enabled" : "disabled") << endl;

  if (hillClimber == 0)
	  cout << "# use hill climber : " << "disabled" << endl;
  else if (hillClimber == 1)
	  cout << "# use hill climber : " << "single" << endl;
  else if (hillClimber == 2)
	  cout << "# use hill climber : " << "multiple" << endl;
  else if (hillClimber == 3)
	  cout << "# use hill climber : " << "Local Search" << endl;

  cout << "# use exhaustive donor search : " << (donorSearch ? "enabled" : "disabled") << endl;
  cout << "# use tournament selection : " << (tournamentSelection ? "enabled" : "disabled") << endl;
  cout << "# similarity measure : " << (similarityMeasure ? "normalized MI" : "MI") << endl;
  cout << "# FOS ordering : " << (orderFOS ? "ascending" : "random") << endl;
  
  cout << "#\n";
  cout << "###################################################\n";
  cout << "#\n";
  cout << "# Problem                      = " << problemName << " " << functionName << endl;
  cout << "# Problem Instance Filename    = " << problemInstancePath << endl;
  cout << "# Number of variables          = " << numberOfVariables << endl;
  cout << "# Alphabet size                = ";
  for (int i = 0; i < numberOfVariables; ++i)
  	cout << alphabetSize[i] << " ";
  cout << endl;
  cout << "# Time Limit (seconds)         = " << timelimitSeconds << endl;
  cout << "# #Evals Limit                 = " << maxEvaluations << endl;
  cout << "# VTR                          = " << ((vtr < 1e+308) ? to_string(vtr) : "not set") << endl;
  cout << "# Random seed                  = " << randomSeed << endl;
  cout << "# Folder                       = " << folder << endl;
  cout << "#\n";
  cout << "### Settings ######################################\n";
}

