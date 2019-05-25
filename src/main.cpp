#include <stdio.h>
#include <math.h>
#include "windows.h"

typedef unsigned int uint;
typedef unsigned char uint8;
typedef unsigned long long uint64;

//----------------------------------------------------------------------------------------------------
//WINDOWS UTILS
//----------------------------------------------------------------------------------------------------

void* allocMemory(uint size)
{
    void* result = VirtualAlloc(0, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    return result;
}

void freeMemory(void* mem)
{
    if (mem)
    {
        VirtualFree(mem, 0, MEM_RELEASE);
    }
}

char* readEntireFile(char* filename)
{
    char* memory = 0;
    
    HANDLE fileHandle;
    fileHandle = CreateFile(filename, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, 0, 0);
    
    if (fileHandle != INVALID_HANDLE_VALUE)
    {
        LARGE_INTEGER fileSize;
        if (GetFileSizeEx(fileHandle, &fileSize))
        {
            memory = (char*)allocMemory((uint)(fileSize.QuadPart + 1));
            if (memory)
            {
                DWORD bytesRead = 0;
                if (!ReadFile(fileHandle, memory, (DWORD)fileSize.QuadPart, &bytesRead, 0))
                {
                    freeMemory(memory);
                    memory = 0;
                }
            }
        }
        memory[fileSize.QuadPart] = 0;
        CloseHandle(fileHandle);
    }
    return memory;
}

uint64 timeMS()
{
    SYSTEMTIME SystemTime;
    GetSystemTime(&SystemTime);
    
    FILETIME FileTime;
    SystemTimeToFileTime(&SystemTime, &FileTime);
    
    uint64 result = {};
    result = result | ((uint64)FileTime.dwHighDateTime << 32);
    result = result | FileTime.dwLowDateTime;
    
    return result / 10000;
}

//----------------------------------------------------------------------------------------------------
//UTILS
//----------------------------------------------------------------------------------------------------

#define assert(x) if(!(x)){(*(int*)0) = 0;}

uint to1d(uint acrossIndex, uint downIndex, uint acrossSize)
{
    assert(acrossIndex < acrossSize);
    uint result = (downIndex * acrossSize) + acrossIndex;
    return result;
}

#define arraySize(x) ((int)(sizeof(x) / sizeof(x[0])))

#define allocStruct(type) (type*)calloc(1, sizeof(type))
#define allocList(num, type) (type*)calloc(num, sizeof(type))

#define clearList(list, count, type) memset((list), 0, (count * sizeof(type)))

//----------------------------------------------------------------------------------------------------
//GETTING THE IMAGE AND LABEL LIST
//----------------------------------------------------------------------------------------------------

struct ImageList
{
    uint width;
    uint height;
    uint numImages;
    uint8* images;
};

struct LabelList
{
    uint numLabels;
    uint8* labels;
};

uint reverseEndian(uint a)
{
    uint8* aList = (uint8*)&a;
    uint result = ((aList[0] << 24) | (aList[1] << 16) | (aList[2] << 8) | (aList[3]));
    
    return result;
}

ImageList loadImages(char* filename)
{
    char* rawFile = readEntireFile(filename);
    
    ImageList imageList = {};
    
    imageList.numImages  = reverseEndian(*(uint*)(rawFile + 4 ));
    imageList.width      = reverseEndian(*(uint*)(rawFile + 8 ));
    imageList.height     = reverseEndian(*(uint*)(rawFile + 12));
    
    imageList.images = (uint8*)(rawFile + 16);
    
    return imageList;
}

LabelList loadLabels(char* filename)
{
    char* rawFile = readEntireFile(filename);
    
    LabelList labelList = {};
    
    labelList.numLabels  = reverseEndian(*(uint*)(rawFile + 4));
    labelList.labels = (uint8*)(rawFile + 8);
    
    return labelList;
}

//----------------------------------------------------------------------------------------------------
//RANDOM
//----------------------------------------------------------------------------------------------------

struct pcgState
{
    bool init;
    uint64 state, increment;
};

pcgState rng;
uint rand32();

void seedRand(uint64 initState, uint64 initIncrement)
{
    rng.state = 0U;
    rng.increment = (initIncrement << 1u) | 1u;
    rand32();
    rng.state += initState;
    rand32();
}

uint rand32()
{
    if (!rng.init)
    {
        rng.init = true;
        seedRand(93458345ULL, 3945678ULL);
    }
    uint64 oldstate = rng.state;
    rng.state = oldstate * 6364136223846793005ULL + rng.increment;
    uint xorshifted = (uint)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((rot | (1 << 31)) & 31));
}

uint randUint(uint minNum, uint maxNum)
{
    assert(minNum < maxNum);
    uint diff = (maxNum - minNum) + 1;
    uint result = minNum + (rand32() % diff);
    return result;
}

//----------------------------------------------------------------------------------------------------
//STRING UTILS
//----------------------------------------------------------------------------------------------------

void clearString(char* string, uint stringLen)
{
    clearList(string, stringLen, char);
}

void removeChar(char* string, uint stringLen, char charRemove)
{
    //make string to copy
    assert(stringLen < 2048);
    char copy[2048] = {};
    char* copyC = (char*)copy;
    
    //copy the needed characters
    char* c = string;
    while (*c != 0)
    {
        if (*c != charRemove)
        {
            *copyC++ = *c;
        }
        ++c;
    }
    
    //copy back to the original
    clearString(string, stringLen);
    copyC = (char*)copy;
    c = string;
    while (*copyC != 0)
    {
        *c++ = *copyC++;
    }
}

void getInput(char* input, uint inputSize)
{
    clearString(input, inputSize);
    fgets(input, inputSize, stdin);
    removeChar(input, inputSize, '\n');
}

char* parseStringToChar(char*& string, char c)
{
    char* result = string;
    while (*string != 0)
    {
        if (*string == c)
        {
            *string = 0;
            ++string;
            break;
        }
        ++string;
    }
    return result;
}

//NOTE: returns 0 on error
uint stringToUint(char* string)
{
    uint result = 0;
    int asInt = atoi(string);
    if (asInt > 0)
    {
        result = (uint)asInt;
    }
    return result;
}

//NOTE: converts uintlist to null pointer in the case of an error
void stringToUintList(char* string, uint stringSize, uint* uintList, uint listSize)
{
    //remove all whitespace
    removeChar(string, stringSize, ' ');
    
    //split the string based on commas
    int listIndex = 0;
    
    while (true)
    {
        char* uintAsString = parseStringToChar(string, ',');
        if (uintAsString == 0 || listIndex == listSize)
        {
            return;
        }
        
        uint asUint = stringToUint(uintAsString);
        if (asUint == 0)
        {
            uintList = 0;
            return;
        }
        
        uintList[listIndex++] = asUint;
    }
}

void uintListToString(uint* uintList, uint listSize, char* string, uint stringSize)
{
    uint listIndex = 0;
    uint stringIndex = 0;
    
    while (true)
    {
        if (listIndex != 0)
        {
            string[stringIndex++] = ',';
            if (stringIndex == stringSize) return;
            string[stringIndex++] = ' ';
            if (stringIndex == stringSize) return;
        }
        
        int charsFilled = snprintf(string + stringIndex, stringSize - stringIndex, "%u", uintList[listIndex++]);
        
        stringIndex += charsFilled;
        if (stringIndex > stringSize || listIndex == listSize) return;
    }
}

//NOTE: returns 0 on error
float stringToFloat(char* string)
{
    float result = atof(string);
    return result;
}

//----------------------------------------------------------------------------------------------------
//NEURAL NETWORK
//----------------------------------------------------------------------------------------------------

float activation(float x)
{
    float result = 1.0f / (1.0f + exp(-x));
    return result;
}

float inverseActivation(float x)
{
    float result = log(x / (1.0f - x));
    return result;
}

float activationPrime(float x)
{
    float result = activation(x) * (1.0f - activation(x));
    return result;
}

struct NeuralNetwork
{
    uint numLayers;
    uint trainingEpochsCompleted;
    uint* numLayerNeurons;
    
    float** weights;
    float** biases;
};

uint getNumWeights(NeuralNetwork network)
{
    uint numWeights = 0;
    for (uint i = 1; i < network.numLayers; ++i)
    {
        numWeights += (network.numLayerNeurons[i] * network.numLayerNeurons[i - 1]);
    }
    return numWeights;
}

uint getNumBiases(NeuralNetwork network)
{
    uint numBiases = 0;
    for (uint i = 1; i < network.numLayers; ++i)
    {
        numBiases += network.numLayerNeurons[i];
    }
    return numBiases;
}

uint getNumNonInputNeurons(NeuralNetwork network)
{
    uint result = getNumBiases(network);
    return result;
}

uint getNumNeurons(NeuralNetwork network)
{
    uint numNeurons  = 0;
    for (uint i = 0; i < network.numLayers; ++i)
    {
        numNeurons += network.numLayerNeurons[i];
    }
    return numNeurons;
}

uint getLayerListFloatOffset(uint numLayers)
{
    //NOTE: returns the amount of floats that it would take to fill up a full list of layers
    uint result = numLayers * (sizeof(float*) / sizeof(float));
    return result;
}

void clearWeightList(float** weightList, NeuralNetwork network)
{
    uint layerListSize = getLayerListFloatOffset(network.numLayers - 1);
    
    uint numWeights = getNumWeights(network);
    float* rawWeightList = (float*)weightList + layerListSize;
    
    clearList(rawWeightList, numWeights, float);
}

void clearBiasList(float** biasList, NeuralNetwork network)
{
    uint layerListSize = getLayerListFloatOffset(network.numLayers - 1);
    
    uint numBiases = getNumBiases(network);
    float* rawBiasList = (float*)biasList + layerListSize;
    
    clearList(rawBiasList, numBiases, float);
}

void clearNeuronList(float** neuronList, NeuralNetwork network)
{
    uint layerListSize = getLayerListFloatOffset(network.numLayers);
    
    uint numNeurons = getNumNeurons(network);
    float* rawNeuronList = (float*)neuronList + layerListSize;
    
    clearList(rawNeuronList, numNeurons, float);
}

void clearNonInputNeuronList(float** list, NeuralNetwork network)
{
    clearBiasList(list, network);
}

float** allocWeightList(NeuralNetwork network)
{
    //NOTE: the weights are laid out in memory so the list of layers is at the start, and then it just has raw data layed out in a 2d list
    //      of length j and width k, where j is the output neuron and k is the input neuron that the signal is coming from
    
    uint layerListSize = getLayerListFloatOffset(network.numLayers - 1);
    
    uint numWeights = getNumWeights(network);
    float** weightList = allocList(numWeights + layerListSize, float*);
    
    uint offset = layerListSize;
    for (uint i = 0; i < (network.numLayers - 1); ++i)
    {
        weightList[i] = ((float*)weightList + offset);
        offset += (network.numLayerNeurons[i + 1] * network.numLayerNeurons[i]);
    }
    
    return weightList;
}

float** allocBiasList(NeuralNetwork network)
{
    //NOTE: the biases are listed in memory so the list of layers is at the start, and then it has the list of neurons for the layer
    
    uint layerListSize = getLayerListFloatOffset(network.numLayers - 1);
    
    uint numBiases = getNumBiases(network);
    float** biasList = allocList(numBiases + layerListSize, float*);
    
    uint offset = layerListSize;
    for (uint i = 0; i < (network.numLayers - 1); ++i)
    {
        biasList[i] = ((float*)biasList + offset);
        offset += network.numLayerNeurons[i + 1];
    }
    
    return biasList;
}

//allocates a list that contains each neuron
float** allocNeuronList(NeuralNetwork network)
{
    uint layerListSize = getLayerListFloatOffset(network.numLayers);
    
    uint numNeurons = getNumNeurons(network);
    float** neuronList = allocList(numNeurons + layerListSize, float*);
    
    uint offset = layerListSize;
    for (uint i = 0; i < network.numLayers; ++i)
    {
        neuronList[i] = ((float*)neuronList + offset);
        offset += network.numLayerNeurons[i];
    }
    
    return neuronList;
}

//allocates a list that contains each neuron but not including the input layer
float** allocNonInputNeuronList(NeuralNetwork network)
{
    float** result = allocBiasList(network);
    return result;
}

NeuralNetwork initNeuralNetwork(uint numLayers, uint* numLayerNeurons)
{
    NeuralNetwork network = {};
    network.numLayers = numLayers;
    network.numLayerNeurons = numLayerNeurons;
    network.weights = allocWeightList(network);
    network.biases = allocBiasList(network);
    return network;
}

float& getWeight(float**& weightList, uint* numLayerNeurons, uint layerFrom, uint nodeTo, uint nodeFrom)
{
    float& result = weightList[layerFrom][to1d(nodeFrom, nodeTo, numLayerNeurons[layerFrom])];
    return result;
}

float& getBias(float**& biasList, uint layer, uint neuron)
{
    float& result = biasList[layer - 1][neuron];
    return result;
}

float& getNonInputNeuron(float**& nonInputNeuronList, uint layer, uint neuron)
{
    float& result = nonInputNeuronList[layer - 1][neuron];
    return result;
}

float& getNeuron(float**& fullNeuronList, uint layer, uint neuron)
{
    float& result = fullNeuronList[layer][neuron];
    return result;
}

void feedForward(NeuralNetwork network, float** activations)
{
    assert(activations);
    
    for (uint layerIndex = 1; layerIndex < network.numLayers; ++layerIndex)
    {
        float* layerInput = activations[layerIndex - 1];
        uint numInputNeurons = network.numLayerNeurons[layerIndex - 1];
        
        float* layerOutput = activations[layerIndex];
        uint numOutputNeurons = network.numLayerNeurons[layerIndex];
        
        for (uint layerOutputIndex = 0; layerOutputIndex < numOutputNeurons; ++layerOutputIndex)
        {
            float& output = layerOutput[layerOutputIndex];
            output = 0;
            
            for (uint layerInputIndex = 0; layerInputIndex < numInputNeurons; ++layerInputIndex)
            {
                output += layerInput[layerInputIndex] * getWeight(network.weights, network.numLayerNeurons, layerIndex - 1, layerOutputIndex, layerInputIndex);
            }
            
            output += getBias(network.biases, layerIndex, layerOutputIndex);
            output = activation(output);
        }
    }
}

float randomDist(float variance, float iterations)
{
    float random = 0;
    
    for (uint i = 0; i < iterations; ++i)
    {
        float randomUnitInterval = (float)rand32() / (float)0xFFFFFFFF;
        assert(randomUnitInterval >= 0.0f && randomUnitInterval <= 1.0f);
        random += randomUnitInterval;
    }
    random /= iterations;
    
    random *= variance * 2;
    random -= variance;
    return random;
}

void randomiseNetwork(NeuralNetwork network)
{
    uint layerListOffset = getLayerListFloatOffset(network.numLayers - 1);
    
    uint numWeights = getNumWeights(network);
    float* rawWeightList = ((float*)network.weights + layerListOffset);
    
    for (uint weightIndex = 0; weightIndex < numWeights; ++weightIndex)
    {
        rawWeightList[weightIndex] = randomDist(3, 32);
    }
    
    uint numBiases = getNumBiases(network);
    float* rawBiasList = ((float*)network.biases + layerListOffset);
    
    for (uint biasIndex = 0; biasIndex < numBiases; ++biasIndex)
    {
        rawBiasList[biasIndex] = randomDist(3, 32);
    }
}

void setNetwork(NeuralNetwork network, float value)
{
    uint layerListOffset = getLayerListFloatOffset(network.numLayers - 1);
    
    uint numWeights = getNumWeights(network);
    float* rawWeightList = ((float*)network.weights + layerListOffset);
    
    for (uint weightIndex = 0; weightIndex < numWeights; ++weightIndex)
    {
        rawWeightList[weightIndex] = value;
    }
    
    uint numBiases = getNumBiases(network);
    float* rawBiasList = ((float*)network.biases + layerListOffset);
    
    for (uint biasIndex = 0; biasIndex < numBiases; ++biasIndex)
    {
        rawBiasList[biasIndex] = value;
    }
}

void intToNetworkOutput(float* labelOutput, NeuralNetwork network, uint digit)
{
    uint lastLayerSize = network.numLayerNeurons[network.numLayers - 1];
    clearList(labelOutput, lastLayerSize, float);
    labelOutput[digit] = 1.0f;
}

float sq(float x)
{
    float result = x*x;
    return result;
}

void imageToNetworkInput(uint8* image, uint imageSize, float*& networkInput)
{
    if (!networkInput)
    {
        networkInput = allocList(imageSize, float);
    }
    
    for (uint i = 0; i < imageSize; ++i)
    {
        networkInput[i] = ((float)image[i] / (float)0xFF);
    }
}

float costPrime(float networkOutput, float intendedOutput)
{
    float result = networkOutput - intendedOutput;
    return result;
}

//there is a lot of repeated code from the last layer output code (that uses the cost function to change the values)
//and the rest of the layers (that use the deltas from before that)
//i might be able to come up with a better algorithm for this
void backProp(NeuralNetwork network, float* intendedOutput, float** activationList, float** overallDeltaWeights, float** overallDeltaBiases, float** delta)
{
    uint lastLayer = network.numLayers - 1;
    uint lastLayerNeurons = network.numLayerNeurons[lastLayer];
    
    for (uint lastLayerIndex = 0; lastLayerIndex < lastLayerNeurons; ++lastLayerIndex)
    {
        float cPrime = costPrime(activationList[lastLayer][lastLayerIndex], intendedOutput[lastLayerIndex]);
        float zValue = inverseActivation(getNeuron(activationList, lastLayer, lastLayerIndex));
        float aPrime = activationPrime(zValue);
        
        float& neuronDelta = getNonInputNeuron(delta, lastLayer, lastLayerIndex);
        neuronDelta = cPrime * aPrime;
        
        uint nextLayer = lastLayer - 1;
        for (uint nextLayerIndex = 0; nextLayerIndex < network.numLayerNeurons[nextLayer]; ++nextLayerIndex)
        {
            float& deltaWeight = getWeight(overallDeltaWeights, network.numLayerNeurons, nextLayer, lastLayerIndex, nextLayerIndex);
            deltaWeight += getNeuron(activationList, nextLayer, nextLayerIndex) * neuronDelta;
        }
        
        float& deltaBias = getBias(overallDeltaBiases, lastLayer, lastLayerIndex);
        deltaBias += neuronDelta;
    }
    
    for (uint layerIndex = lastLayer - 1; layerIndex > 0; --layerIndex)
    {
        for (uint neuronIndex = 0; neuronIndex < network.numLayerNeurons[layerIndex]; ++neuronIndex)
        {
            float sumWeightsWithDeltas = 0;
            uint prevLayer = layerIndex + 1;
            for (uint prevLayerNeuronIndex = 0; prevLayerNeuronIndex < network.numLayerNeurons[prevLayer]; ++prevLayerNeuronIndex)
            {
                float prevDelta = getNonInputNeuron(delta, prevLayer, prevLayerNeuronIndex);
                float weight = getWeight(network.weights, network.numLayerNeurons, layerIndex, prevLayerNeuronIndex, neuronIndex);
                sumWeightsWithDeltas += prevDelta * weight;
            }
            
            float zValue = inverseActivation(getNeuron(activationList, layerIndex, neuronIndex));
            float aPrime = activationPrime(zValue);
            
            float& neuronDelta = getNonInputNeuron(delta, layerIndex, neuronIndex);
            neuronDelta = sumWeightsWithDeltas * aPrime;
            
            uint nextLayer = layerIndex - 1;
            for (uint nextLayerIndex = 0; nextLayerIndex < network.numLayerNeurons[nextLayer]; ++nextLayerIndex)
            {
                float& deltaWeight = getWeight(overallDeltaWeights, network.numLayerNeurons, nextLayer, neuronIndex, nextLayerIndex);
                deltaWeight += getNeuron(activationList, nextLayer, nextLayerIndex) * neuronDelta;
            }
            
            float& deltaBias = getBias(overallDeltaBiases, layerIndex, neuronIndex);
            deltaBias += neuronDelta;
        }
    }
}

void fillSequentialUints(uint* list, uint listSize)
{
    for (uint i = 0; i < listSize; ++i)
    {
        list[i] = i;
    }
}

void shuffleUintList(uint* list, uint listSize)
{
    uint numSwaps = listSize * 4;
    
    for (uint i = 0; i < numSwaps; ++i)
    {
        uint a = randUint(0, listSize - 1);
        uint b = randUint(0, listSize - 1);
        
        uint swappingValue = list[a];
        list[a] = list[b];
        list[b] = swappingValue;
    }
}

void trainNetwork(NeuralNetwork& network, ImageList imageList, LabelList labelList, float step, uint miniBatchSize)
{
    assert(imageList.numImages == labelList.numLabels);
    assert((imageList.numImages % miniBatchSize) == 0);
    
    network.trainingEpochsCompleted++;
    
    uint* miniBatchList = allocList(imageList.numImages, uint);
    uint numMiniBatches = imageList.numImages / miniBatchSize;
    
    fillSequentialUints(miniBatchList, imageList.numImages);
    shuffleUintList(miniBatchList, imageList.numImages);
    
    uint imageSize = imageList.width * imageList.height;
    uint listOffset = getLayerListFloatOffset(network.numLayers - 1);
    
    uint lastLayerSize = network.numLayerNeurons[network.numLayers - 1];
    float* labelOutput = allocList(lastLayerSize, float);
    
    float** activationList = allocNeuronList(network);
    
    float** deltaWeights = allocWeightList(network);
    float** deltaBiases = allocBiasList(network);
    float** delta = allocNonInputNeuronList(network);
    
    for (uint miniBatchIndex = 0; miniBatchIndex < numMiniBatches; ++miniBatchIndex)
    {
        clearWeightList(deltaWeights, network);
        clearBiasList(deltaBiases, network);
        
        uint imageIndexStart = to1d(0, miniBatchIndex, miniBatchSize);
        
        for (uint imageIndex = imageIndexStart; imageIndex < imageIndexStart + miniBatchSize; ++imageIndex)
        {
            uint imageNum = miniBatchList[imageIndex];
            uint8* image = imageList.images + (imageNum * imageSize);
            
            //@SPEEDUP: both of these could be precomputed, although it wouldn't help loads
            imageToNetworkInput(image, imageSize, activationList[0]);
            intToNetworkOutput(labelOutput, network, labelList.labels[imageNum]);
            
            //NOTE: these take much more time than above
            feedForward(network, activationList);
            backProp(network, labelOutput, activationList, deltaWeights, deltaBiases, delta);
        }
        
        float* networkWeightsList = ((float*)network.weights) + listOffset;
        float* deltaWeightsList = ((float*)deltaWeights) + listOffset;
        
        for (uint weightIndex = 0; weightIndex < getNumWeights(network); ++weightIndex)
        {
            networkWeightsList[weightIndex] -= (deltaWeightsList[weightIndex] / miniBatchSize) * step;
        }
        
        float* networkBiasesList = ((float*)network.biases) + listOffset;
        float* deltaBiasesList = ((float*)deltaBiases) + listOffset;
        
        for (uint biasIndex = 0; biasIndex < getNumBiases(network); ++biasIndex)
        {
            networkBiasesList[biasIndex] -= (deltaBiasesList[biasIndex] / miniBatchSize) * step;
        }
    }
    free(deltaWeights);
    free(deltaBiases);
    free(delta);
    
    free(activationList);
    
    free(labelOutput);
}

uint networkOutputToUint(NeuralNetwork network, float* networkOutput)
{
    uint bestValue = 0;
    uint lastLayer = network.numLayers - 1;
    for (uint i = 1; i < network.numLayerNeurons[lastLayer]; ++i)
    {
        if (networkOutput[i] > networkOutput[bestValue])
        {
            bestValue = i;
        }
    }
    return bestValue;
}

float testNetworkAccuracy(NeuralNetwork network, ImageList imageList, LabelList labelList)
{
    assert(imageList.numImages == labelList.numLabels);
    
    uint imageSize = imageList.width * imageList.height;
    
    float** activations = allocNeuronList(network);
    
    uint numCorrect = 0;
    for (uint inputIndex = 0; inputIndex < imageList.numImages; ++inputIndex)
    {
        uint8* image = imageList.images + (inputIndex * imageSize);
        
        imageToNetworkInput(image, imageSize, activations[0]);
        
        feedForward(network, activations);
        uint lastLayerIndex = (network.numLayers - 1);
        uint networkDigit = networkOutputToUint(network, activations[lastLayerIndex]);
        
        if (networkDigit == labelList.labels[inputIndex])
        {
            ++numCorrect;
        }
    }
    free(activations);
    
    float result = (float)numCorrect / (float)imageList.numImages;
    return result;
}

int main()
{
    printf("A Neural Network for classifying digits (see \"Manual.pdf\" for instructions)");
    
    ImageList imageList = loadImages("resources/training/images.dat");
    LabelList labelList = loadLabels("resources/training/labels.dat");
    
    ImageList testingImageList = loadImages("resources/testing/images.dat");
    LabelList testingLabelList = loadLabels("resources/testing/labels.dat");
    
    char userInput[1024] = {};
    bool validResponse = false;
    uint imageSize = imageList.width * imageList.height;
    uint numLayers = 1;
    uint layers[34] = {imageSize};
    uint runningSeconds = 0;
    float backpropStep = 0.0f;
    uint miniBatchSize = 0;
    
    while (!validResponse)
    {
        printf("\nHidden layer layout: ");
        getInput(userInput, arraySize(userInput));
        
        uint hiddenLayers[32] = {};
        stringToUintList(userInput, arraySize(userInput), hiddenLayers, arraySize(hiddenLayers));
        
        if (hiddenLayers[0] != 0)
        {
            validResponse = true;
            
            for (uint layerIndex = 1; hiddenLayers[layerIndex - 1] != 0; ++layerIndex)
            {
                layers[layerIndex] = hiddenLayers[layerIndex - 1];
                ++numLayers;
            }
            layers[numLayers++] = 10;
            }
        else
        {
            printf("ERROR: Invalid hidden layer layout\n");
    }
        }
    
    validResponse = false;
    while (!validResponse)
    {
        printf("\nBackprop step amount: ");
        getInput(userInput, arraySize(userInput));
         backpropStep = stringToFloat(userInput);
        
        if (backpropStep != 0.0f)
        {
            validResponse = true;
            }
        else
        {
        printf("ERROR: Invalid backprop step amount\n");
        }
    }
    
    validResponse = false;
    while (!validResponse)
    {
        printf("\nTraining samples per mini batch (must be divisible by 60,000): ");
        getInput(userInput, arraySize(userInput));
        
         miniBatchSize = stringToUint(userInput);
        bool isDivisible = (60000.0f / (float)miniBatchSize) == (60000 / miniBatchSize);
        
        if (miniBatchSize != 0 && isDivisible)
        {
        validResponse = true;
        }
        else
        {
        printf("ERROR: Invalid mini batch size\n");
        }
    }
    
    validResponse = false;
    while (!validResponse)
    {
        printf("\nTraining time (seconds): ");
        getInput(userInput, arraySize(userInput));
        
         runningSeconds = stringToUint(userInput);
        if (runningSeconds != 0)
        {
            validResponse = true;
        }
        else
        {
        printf("ERROR: Invalid training time\n");
        }
    }
    
    NeuralNetwork network = initNeuralNetwork(numLayers, layers);
    randomiseNetwork(network);
    
    printf("\nTraining network for %u seconds...\n", runningSeconds);
    
    uint64 startTime = timeMS();
    uint64 endTime = startTime + (runningSeconds * 1000);
    
    for (uint epochIndex = 0; endTime > timeMS(); ++epochIndex)
    {
        printf("%.2f%% ", (float)(timeMS() - startTime) / (runningSeconds * 10.0f));
        trainNetwork(network, imageList, labelList, backpropStep, miniBatchSize);
    }
    
    printf("100.00%%\n\n");
    
    char layout[256] = {};
    uintListToString(layers, numLayers, layout, arraySize(layout));
    
    float trainingAccuracy = testNetworkAccuracy(network, imageList, labelList) * 100.0f;
    float testAccuracy = testNetworkAccuracy(network, testingImageList, testingLabelList) * 100.0f;
    
    printf("=============================================\n");
    printf("Network Layers         : %s\n", layout);
    printf("Backprop Step Amount   : %.2f\n", backpropStep);
    printf("Mini-Batch Size        : %u\n", miniBatchSize);
    printf("Epochs Completed       : %u\n",  network.trainingEpochsCompleted);
    printf("Training data accuracy : %.2f%%\n", trainingAccuracy);
    printf("Test data Accuracy     : %.2f%%\n", testAccuracy);
    printf("=============================================\n");
    
    getchar();
    return 0;
}