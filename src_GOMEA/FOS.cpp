#include "FOS.hpp"

void createFOSInstance(FOS **FOSInstance, size_t numberOfVariables, vector<int> &alphabetSize, int similarityMeasure)
{
    *FOSInstance = new LTFOS(numberOfVariables, alphabetSize, similarityMeasure, true); //filtered Linkage Tree        
}

void FOS::shuffleFOS(vector<int> &indices, mt19937 *rng)
{
    indices.resize(FOSSize());
    iota(indices.begin(), indices.end(), 0);   
    shuffle(indices.begin(), indices.end(), *rng);
}

void FOS::sortFOSAscendingOrder(vector<int> &indices)
{
    vector<pair<int, int> > fos_goodness;
    indices.resize(FOSSize());

    for (size_t i = 0; i < FOSSize(); i++)
        fos_goodness.push_back(make_pair(FOSElementSize(i), i));
    
    sort(fos_goodness.begin(), fos_goodness.end(), [](const pair<double, int>& lhs, const pair<double, int>& rhs)
    {
        return lhs.first < rhs.first;
    });

    for (size_t i = 0; i < FOSSize(); i++)
        indices[i] = fos_goodness[i].second;
}

void FOS::sortFOSDescendingOrder(vector<int> &indices)
{
    vector<pair<int, int> > fos_goodness;
    indices.resize(FOSSize());

    for (size_t i = 0; i < FOSSize(); i++)
        fos_goodness.push_back(make_pair(FOSElementSize(i), i));
    
    sort(fos_goodness.begin(), fos_goodness.end(), [](const pair<double, int>& lhs, const pair<double, int>& rhs)
    {
        return lhs.first > rhs.first;
    });

    for (size_t i = 0; i < FOSSize(); i++)
        indices[i] = fos_goodness[i].second;
}

void FOS::orderFOS(int orderingType, vector<int> &indices, mt19937 *rng)
{
    if (orderingType == 0)
        shuffleFOS(indices, rng);
    else if (orderingType == 1)
        sortFOSAscendingOrder(indices);
    // else if (orderingType == 2)
    //     sortFOSDescendingOrder(indices);       
}


/////////////////////////////////////////////////////////////////////////////////////////////////////

LTFOS::LTFOS(size_t numberOfVariables_, vector<int> &alphabetSize_, int similarityMeasure_, bool filtered_): FOS(numberOfVariables_, alphabetSize_)
{
    similarityMeasure = similarityMeasure_;
    filtered = filtered_;

    MI_Matrix.resize(numberOfVariables);        
    S_Matrix.resize(numberOfVariables);

    for (size_t i = 0; i < numberOfVariables; ++i)
    {
        MI_Matrix[i].resize(numberOfVariables);         
        S_Matrix[i].resize(numberOfVariables);
    }
}

void LTFOS::learnFOS(vector<Individual*> &population, vector<vector<int> > *VIG, mt19937 *rng)
{
    FOSStructure.clear();
    vector<int> mpmFOSMap;
    vector<int> mpmFOSMapNew;

    
    /* Compute Mutual Information matrix */
    if (similarityMeasure == 0) // MI
        computeMIMatrix(population);
    else if (similarityMeasure == 1) // normalized MI
        computeNMIMatrix(population);
    
    #ifdef DEBUG
        for (int i = 0; i < numberOfVariables; ++i)
        {
            for (int j = 0; j < numberOfVariables; ++j)
                cout << MI_Matrix[i][j] << " ";
            cout << endl;
        }
    #endif

    /* Initialize MPM to the univariate factorization */
    vector <int> order(numberOfVariables);
    iota(order.begin(), order.end(), 0);
    shuffle(order.begin(), order.end(), *rng);

    vector< vector<int> > mpm(numberOfVariables);
    vector< vector<int> > mpmNew(numberOfVariables);
    
    for (size_t i = 0; i < numberOfVariables; i++)
    {
        mpm[i].push_back(order[i]);  
    }

    /* Initialize LT to the initial MPM */
    int FOSLength = 2 * numberOfVariables - 1;
    FOSStructure.resize(FOSLength);

    vector<int> useFOSElement(FOSStructure.size(), true);

    int FOSsIndex = 0;
    for (size_t i = 0; i < numberOfVariables; i++)
    {
        FOSStructure[i] = mpm[i];
        mpmFOSMap.push_back(i);
        FOSsIndex++;
    }

    for (size_t i = 0; i < numberOfVariables; ++i)
    {
        for(size_t j = 0; j < numberOfVariables; j++ )
            S_Matrix[i][j] = MI_Matrix[mpm[i][0]][mpm[j][0]];//((*rng)()%10000)/10000.0

        S_Matrix[i][i] = 0;
    }

    vector<int> NN_chain;
    NN_chain.resize(numberOfVariables+2);
    size_t NN_chain_length = 0;
    bool done = false;
    while (!done)
    {
        if (NN_chain_length == 0)
        {
            NN_chain[NN_chain_length] = (*rng)() % mpm.size();
            //std::cout << NN_chain[NN_chain_length] << " | " << mpm.size() << std::endl;
      
            NN_chain_length++;
        }

        while (NN_chain_length < 3)
        {
            NN_chain[NN_chain_length] = determineNearestNeighbour(NN_chain[NN_chain_length-1], mpm);
            NN_chain_length++;
        }

        while (NN_chain[NN_chain_length-3] != NN_chain[NN_chain_length-1])
        {
            NN_chain[NN_chain_length] = determineNearestNeighbour(NN_chain[NN_chain_length-1], mpm);
            if( ((S_Matrix[NN_chain[NN_chain_length-1]][NN_chain[NN_chain_length]] == S_Matrix[NN_chain[NN_chain_length-1]][NN_chain[NN_chain_length-2]])) && (NN_chain[NN_chain_length] != NN_chain[NN_chain_length-2]) )
                NN_chain[NN_chain_length] = NN_chain[NN_chain_length-2];
            
            NN_chain_length++;
            if (NN_chain_length > numberOfVariables)
                break;
        }

        size_t r0 = NN_chain[NN_chain_length-2];
        size_t r1 = NN_chain[NN_chain_length-1];
        if (S_Matrix[NN_chain[NN_chain_length-1]][NN_chain[NN_chain_length-2]] >= 1-(1e-6))
        {         
            useFOSElement[mpmFOSMap[r0]] = false;
            useFOSElement[mpmFOSMap[r1]] = false;            
        }

        if (r0 > r1)
        {
            int rswap = r0;
            r0 = r1;
            r1 = rswap;
        }
        NN_chain_length -= 3;


        if (r1 < mpm.size()) 
        {
            vector<int> indices(mpm[r0].size() + mpm[r1].size());

            size_t i = 0;
            for (size_t j = 0; j < mpm[r0].size(); j++)
            {
                indices[i] = mpm[r0][j];
                i++;
            }

            for (size_t j = 0; j < mpm[r1].size(); j++)
            {
                indices[i] = mpm[r1][j];
                i++;
            }

            FOSStructure[FOSsIndex] = indices;
            FOSsIndex++;

            double mul0 = (double)mpm[r0].size() / (double)(mpm[r0].size() + mpm[r1].size());
            double mul1 = (double)mpm[r1].size() / (double)(mpm[r0].size() + mpm[r1].size());
            for (size_t i = 0; i < mpm.size(); i++)
            {
                if ((i != r0) && (i != r1))
                {
                    S_Matrix[i][r0] = mul0 * S_Matrix[i][r0] + mul1 * S_Matrix[i][r1];
                    S_Matrix[r0][i] = S_Matrix[i][r0];
                }
            }
    
            mpmNew.resize(mpm.size() - 1);
            mpmFOSMapNew.resize(mpmFOSMap.size()-1);
            for (size_t i = 0; i < mpmNew.size(); i++)
            {
                mpmNew[i] = mpm[i];
                mpmFOSMapNew[i] = mpmFOSMap[i];
            }

            mpmNew[r0] = indices;
            mpmFOSMapNew[r0] = FOSsIndex-1;

            if (r1 < mpm.size() - 1)
            {
                mpmNew[r1] = mpm[mpm.size() - 1];
                mpmFOSMapNew[r1] = mpmFOSMap[mpm.size() - 1];

                for (int i = 0; i < r1; i++)
                {
                  S_Matrix[i][r1] = S_Matrix[i][mpm.size() - 1];
                  S_Matrix[r1][i] = S_Matrix[i][r1];
                }

                for (int j = r1 + 1; j < mpmNew.size(); j++)
                {
                  S_Matrix[r1][j] = S_Matrix[j][mpm.size() - 1];
                  S_Matrix[j][r1] = S_Matrix[r1][j];
                }
            }

            for (i = 0; i < NN_chain_length; i++)
            {
                if (NN_chain[i] == mpm.size() - 1)
                {
                    NN_chain[i] = r1;
                    break;
                }
            }

            mpm = mpmNew;
            mpmFOSMap = mpmFOSMapNew;

            if (mpm.size() == 1)
                done = true;
        }
   }

    if (filtered)
    {
        for (int i = 0; i < useFOSElement.size(); ++i)
        {
            if (!useFOSElement[i])
            {
                FOSStructure[i].clear();
            }        
        }
    }

    int size = FOSStructure.size();

    for (int i = 0; i < size; ++i)
    {
        if (FOSStructure[i].size()==0)
            continue;
    }

}

/**
 * Determines nearest neighbour according to similarity values.
 */
int LTFOS::determineNearestNeighbour(int index, vector<vector< int> > &mpm)
{
    int result = 0;

    if (result == index)
        result++;

    for (size_t i = 1; i < mpm.size(); i++)
    {
        //std::cout << index << " " << i << " " << result << " " << S_Matrix[index][i] << " " << S_Matrix[index][result] << " " << mpm[i].size() << " " << mpm[result].size() << std::endl;

        if (i != index)
        {
            if ((S_Matrix[index][i] > S_Matrix[index][result]) || ((S_Matrix[index][i] == S_Matrix[index][result]) && (mpm[i].size() < mpm[result].size())))
            {
                result = i;
                //std::cout << (int)(S_Matrix[index][i] > S_Matrix[index][result])<< " result=i" << std::endl;
            }
        }
    }
    //std::cout << "nearest-neighbor " << result << endl;
    return result;
}

void LTFOS::computeMIMatrix(vector<Individual*> &population)
{
    size_t factorSize;
    double p;
    
    /* Compute joint entropy matrix */
    for (size_t i = 0; i < numberOfVariables; i++)
    {
        for (size_t j = i + 1; j < numberOfVariables; j++)
        {
            vector<size_t> indices{i, j};
            vector<double> factorProbabilities;
            estimateParametersForSingleBinaryMarginal(population, indices, factorSize, factorProbabilities);

            MI_Matrix[i][j] = 0.0;
            for(size_t k = 0; k < factorSize; k++)
            {
                p = factorProbabilities[k];
                if (p > 0)
                    MI_Matrix[i][j] += -p * log2(p);
            }
            MI_Matrix[j][i] = MI_Matrix[i][j];
        }

        vector<size_t> indices{i};
        vector<double> factorProbabilities;
        estimateParametersForSingleBinaryMarginal(population, indices, factorSize, factorProbabilities);

        MI_Matrix[i][i] = 0.0;
        for (size_t k = 0; k < factorSize; k++)
        {
            p = factorProbabilities[k];
            if (p > 0)
                MI_Matrix[i][i] += -p * log2(p);
        }

    }

    /* Then transform into mutual information matrix MI(X,Y)=H(X)+H(Y)-H(X,Y) */
    for (size_t i = 0; i < numberOfVariables; i++)
    {
        for (size_t j = i + 1; j < numberOfVariables; j++)
        {
            MI_Matrix[i][j] = MI_Matrix[i][i] + MI_Matrix[j][j] - MI_Matrix[i][j];
            MI_Matrix[j][i] = MI_Matrix[i][j];
        }
    }

}

void LTFOS::computeNMIMatrix(vector<Individual*> &population)
{
    double p;
    
    /* Compute joint entropy matrix */
    for (size_t i = 0; i < numberOfVariables; i++)
    {
        for (size_t j = i + 1; j < numberOfVariables; j++)
        {
            vector<double> factorProbabilities_joint;
            vector<double> factorProbabilities_i;
            vector<double> factorProbabilities_j;
            size_t factorSize_joint, factorSize_i, factorSize_j;

            vector<size_t> indices_joint{i, j};
            estimateParametersForSingleBinaryMarginal(population, indices_joint, factorSize_joint, factorProbabilities_joint);
            
            vector<size_t> indices_i{i};
            estimateParametersForSingleBinaryMarginal(population, indices_i, factorSize_i, factorProbabilities_i);
            
            vector<size_t> indices_j{j};
            estimateParametersForSingleBinaryMarginal(population, indices_j, factorSize_j, factorProbabilities_j);

            MI_Matrix[i][j] = 0.0;
            
            double separate = 0.0, joint = 0.0;

            for(size_t k = 0; k < factorSize_joint; k++)
            {
                p = factorProbabilities_joint[k];
                //cout << i << " " << j << " " << p << endl;
                if (p > 0)
                    joint += (-p * log2(p));
            }

            for(size_t k = 0; k < factorSize_i; k++)
            {
                p = factorProbabilities_i[k];
                if (p > 0)
                    separate += (-p * log2(p));
            }

            for(size_t k = 0; k < factorSize_j; k++)
            {
                p = factorProbabilities_j[k];
                if (p > 0)
                    separate += (-p * log2(p));
            }
            //cout << separate << " " << joint << endl;
            MI_Matrix[i][j] = 0.0;
            if (joint)
                MI_Matrix[i][j] = separate / joint - 1;
            MI_Matrix[j][i] = MI_Matrix[i][j];

        }

    }

}

/**
 * Estimates the cumulative probability distribution of a
 * single binary marginal.
 */
void LTFOS::estimateParametersForSingleBinaryMarginal(vector<Individual*> &population, vector<size_t> &indices, size_t &factorSize, vector<double> &result)
{
    size_t numberOfIndices = indices.size();

    factorSize = 1;
    for (int i = 0; i < numberOfIndices; ++i)
        factorSize *= alphabetSize[indices[i]];

    result.resize(factorSize);
    fill(result.begin(), result.end(), 0.0);

    for (size_t i = 0; i < population.size(); i++)
    {
        int index = 0;
        int power = 1;
        for (int j = numberOfIndices-1; j >= 0; j--)
        {
            int var = indices[j];
            //cout << "var:" << var << " " << power << " " << +population[i]->genotype[var] << endl;
            index += (int)population[i]->genotype[var] * power;
            power *= alphabetSize[var];
        }

        result[index] += 1.0;
        //cout << "index:" << index << " " << factorSize << endl;
    }   
    for (size_t i = 0; i < factorSize; i++)
        result[i] /= (double)population.size();
}

