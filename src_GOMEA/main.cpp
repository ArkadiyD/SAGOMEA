#include "Config.hpp"
#include "gomea.hpp"
#include "gomeaP3.hpp"

int main(int argc, char **argv)
{
    Py_Initialize();

    Config *config = new Config();
    config->parseCommandLine(argc, argv);
    config->printOverview();

    config->rng.seed(config->randomSeed);

    GOMEA *gomeaInstance;
    gomeaInstance = new gomeaP3(config);
    
    try
    {
        gomeaInstance->run();
    }
    catch (customException &ex)
    {}

    delete gomeaInstance;
    delete config;
    
    Py_Finalize();
    
    return 0;
}