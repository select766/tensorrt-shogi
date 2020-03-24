/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// 複数のスレッドで複数のGPUを同時に実行
// 連続的に識別をしてベンチマークする
// バッチサイズ可変 or 固定
// ファイルからテストデータを読んでチェックするモード

#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <memory>
#include <mutex>
#include <thread>
#include <random>
#include <atomic>
#include <chrono>

#include "common.h"
#include "buffers.h"

static bool useMultiProfile = true;
static int batchSizeMin = 1;
static int batchSizeMax = 16;
static int nGPU = 1;
static int nThreadPerGPU = 2;
static bool fp16 = false;
static bool fp8 = false;
static bool verifyMode = false;
static int skipSample = 0;
static std::vector<std::string> inputTensorNames;
static std::vector<std::string> outputTensorNames;
static std::vector<std::mutex *> gpuMutexes;
static std::mutex resultMutex;
static std::atomic_bool waitBenchStart;
static std::atomic_bool continueBench;
static std::atomic_int nThreadInitialized;
std::vector<int> totalCounts;
std::vector<long long> totalTimesum;
std::string serializePath("/var/tmp/multi_gpu_bench.bin");

std::string addProfileSuffix(const std::string &name, int profile)
{
    std::ostringstream oss;
    oss << name;
    if (profile > 0)
    {
        oss << " [profile " << profile << "]";
    }

    return oss.str();
}

bool checkSerializedFile()
{
    ifstream f(serializePath, ios::in | ios::binary);
    return f.is_open();
}

//! \brief  The ShogiOnnx class implements the ONNX MNIST sample
//!
//! \details It creates the network using an ONNX model
//!
class ShogiOnnx
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    ShogiOnnx()
        : mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    bool serialize();

    //!
    //! \brief Function deserialize the network engine from file
    //!
    bool load();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer(int batchSize);

private:
    nvinfer1::Dims mInputDims;        //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputPolicyDims; //!< The dimensions of the output to the network.
    nvinfer1::Dims mOutputValueDims;  //!< The dimensions of the output to the network.

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
    std::map<int, std::shared_ptr<nvinfer1::IExecutionContext>> mContextForProfile;
    std::vector<int> profileForBatchSize;

    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder> &builder,
                          SampleUniquePtr<nvinfer1::INetworkDefinition> &network, SampleUniquePtr<nvinfer1::IBuilderConfig> &config,
                          SampleUniquePtr<nvonnxparser::IParser> &parser);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager &buffers, int batchSize);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const samplesCommon::BufferManager &buffers, int batchSize);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool ShogiOnnx::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }
    if (!useMultiProfile)
    {
        auto profile = builder->createOptimizationProfile();
        profile->setDimensions(inputTensorNames[0].c_str(), OptProfileSelector::kMIN, Dims4{1, 119, 9, 9});
        profile->setDimensions(inputTensorNames[0].c_str(), OptProfileSelector::kOPT, Dims4{batchSizeMax, 119, 9, 9});
        profile->setDimensions(inputTensorNames[0].c_str(), OptProfileSelector::kMAX, Dims4{batchSizeMax, 119, 9, 9});
        int profileIdx = config->addOptimizationProfile(profile);
        profileForBatchSize.resize(batchSizeMax + 1);
        for (int b = 1; b <= batchSizeMax; b++)
        {
            profileForBatchSize[b] = profileIdx;
        }
    }
    else
    {
        int bs = 1;
        int lastbs = 0;
        profileForBatchSize.resize(batchSizeMax + 1);
        while (lastbs < batchSizeMax)
        {
            auto profile = builder->createOptimizationProfile();
            if (bs > batchSizeMax)
            {
                bs = batchSizeMax;
            }
            profile->setDimensions(inputTensorNames[0].c_str(), OptProfileSelector::kMIN, Dims4{lastbs + 1, 119, 9, 9});
            profile->setDimensions(inputTensorNames[0].c_str(), OptProfileSelector::kOPT, Dims4{bs, 119, 9, 9});
            profile->setDimensions(inputTensorNames[0].c_str(), OptProfileSelector::kMAX, Dims4{bs, 119, 9, 9});
            int profileIdx = config->addOptimizationProfile(profile);
            for (int b = lastbs + 1; b <= bs; b++)
            {
                profileForBatchSize[b] = profileIdx;
            }

            lastbs = bs;
            bs *= 4;
        }
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    // different context for each profile is needed (switching causes error on setBindingDimensions)
    for (int i = 0; i < mEngine->getNbOptimizationProfiles(); i++)
    {
        auto ctx = std::shared_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext(), samplesCommon::InferDeleter());
        if (!ctx)
        {
            return false;
        }
        ctx->setOptimizationProfile(i);
        mContextForProfile[i] = ctx;
    }

    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 4);

    assert(network->getNbOutputs() == 2);
    mOutputPolicyDims = network->getOutput(0)->getDimensions();
    assert(mOutputPolicyDims.nbDims == 2);
    mOutputValueDims = network->getOutput(1)->getDimensions();
    assert(mOutputValueDims.nbDims == 2);

    return true;
}

bool ShogiOnnx::serialize()
{
    IHostMemory *serializedModel = mEngine->serialize();

    ofstream serializedModelFile(serializePath, ios::binary);
    serializedModelFile.write((const char *)serializedModel->data(), serializedModel->size());

    return true;
}

bool ShogiOnnx::load()
{
    ifstream serializedModelFile(serializePath, ios::in | ios::binary);
    serializedModelFile.seekg(0, ios_base::end);
    size_t fsize = serializedModelFile.tellg();
    serializedModelFile.seekg(0, ios_base::beg);
    std::vector<char> fdata(fsize);
    serializedModelFile.read((char *)fdata.data(), fsize);

    auto runtime = createInferRuntime(gLogger);
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(fdata.data(), fsize, nullptr), samplesCommon::InferDeleter());

    mInputDims = Dims4{1, 119, 9, 9};
    mOutputPolicyDims = Dims2{1, 2187};
    mOutputValueDims = Dims2{1, 2};
    if (!useMultiProfile)
    {
        int profileIdx = 0;//TODO: 本来はシリアライズして保存すべき
        profileForBatchSize.resize(batchSizeMax + 1);
        for (int b = 1; b <= batchSizeMax; b++)
        {
            profileForBatchSize[b] = profileIdx;
        }
    }
    else
    {
        int bs = 1;
        int lastbs = 0;
        profileForBatchSize.resize(batchSizeMax + 1);
        int profileIdx = 0;//TODO: 本来はシリアライズして保存すべき
        while (lastbs < batchSizeMax)
        {
            if (bs > batchSizeMax)
            {
                bs = batchSizeMax;
            }
            for (int b = lastbs + 1; b <= bs; b++)
            {
                profileForBatchSize[b] = profileIdx;
            }

            lastbs = bs;
            bs *= 4;
            profileIdx++;
        }
    }

    // different context for each profile is needed (switching causes error on setBindingDimensions)
    for (int i = 0; i < mEngine->getNbOptimizationProfiles(); i++)
    {
        auto ctx = std::shared_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext(), samplesCommon::InferDeleter());
        if (!ctx)
        {
            return false;
        }
        ctx->setOptimizationProfile(i);
        mContextForProfile[i] = ctx;
    }

    return true;
}
//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool ShogiOnnx::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder> &builder,
                                 SampleUniquePtr<nvinfer1::INetworkDefinition> &network, SampleUniquePtr<nvinfer1::IBuilderConfig> &config,
                                 SampleUniquePtr<nvonnxparser::IParser> &parser)
{
    // [W] [TRT] Calling isShapeTensor before the entire network is constructed may result in an inaccurate result.
    auto parsed = parser->parseFromFile(
        "data/trt/model.onnx", static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    builder->setMaxBatchSize(batchSizeMax);
    config->setMaxWorkspaceSize(1024_MiB);

    if (fp8)
    {
        gLogInfo << "INT8 mode (scale is not correctly set!)" << std::endl;
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }
    else if (fp16)
    {
        gLogInfo << "FP16 mode" << std::endl;
        config->setFlag(BuilderFlag::kFP16);
    }
    else
    {
        gLogInfo << "FP32 mode" << std::endl;
    }

    samplesCommon::enableDLA(builder.get(), config.get(), -1);

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool ShogiOnnx::infer(int batchSize)
{
    auto mContext = mContextForProfile.at(profileForBatchSize[batchSize]);
    std::string inputBindingName = addProfileSuffix(inputTensorNames[0], profileForBatchSize[batchSize]);
    int bidx = mEngine->getBindingIndex(inputBindingName.c_str());
    mContext->setBindingDimensions(bidx, Dims4{batchSize, 119, 9, 9});
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, batchSize, mContext.get());

    // Read the input data into the managed buffers
    assert(inputTensorNames.size() == 1);
    if (!processInput(buffers, batchSize))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = mContext->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    if (!verifyOutput(buffers, batchSize))
    {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool ShogiOnnx::processInput(const samplesCommon::BufferManager &buffers, int batchSize)
{
    const int inputC = mInputDims.d[1];
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];

    std::string inputName = addProfileSuffix(inputTensorNames[0], profileForBatchSize[batchSize]);

    float *hostDataBuffer = static_cast<float *>(buffers.getHostBuffer(inputName));
    if (verifyMode)
    {
        ifstream fin("data/trt/inputs.bin", ios::in | ios::binary);
        fin.seekg(inputC * inputH * inputW * sizeof(float) * skipSample);
        fin.read((char *)hostDataBuffer, inputC * inputH * inputW * sizeof(float) * batchSize);
    }
    else
    {
        memset(hostDataBuffer, 0, inputC * inputH * inputW * sizeof(float) * batchSize);
    }

    return true;
}

bool compareResult(float *expected, float *actual, size_t size)
{
    float maxDiff = 0.0F;
    size_t maxDiffIdx = 0;
    for (size_t i = 0; i < size; i++)
    {
        float diff = abs(expected[i] - actual[i]);
        if (diff > maxDiff)
        {
            maxDiff = diff;
            maxDiffIdx = i;
        }
    }

    if (maxDiff < 1e-3)
    {
        return true;
    }
    else
    {
        gLogInfo << "max diff among " << size << " elements: [" << maxDiffIdx << "] " << expected[maxDiffIdx] << "!=" << actual[maxDiffIdx] << std::endl;
        return false;
    }
}
//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool ShogiOnnx::verifyOutput(const samplesCommon::BufferManager &buffers, int batchSize)
{
    if (!verifyMode)
    {
        return true;
    }
    const int outputPolicySize = mOutputPolicyDims.d[1];
    std::string outputPName = addProfileSuffix(outputTensorNames[0], profileForBatchSize[batchSize]);
    float *outputPolicy = static_cast<float *>(buffers.getHostBuffer(outputPName));
    const int outputValueSize = mOutputValueDims.d[1];
    std::string outputVName = addProfileSuffix(outputTensorNames[1], profileForBatchSize[batchSize]);
    float *outputValue = static_cast<float *>(buffers.getHostBuffer(outputVName));
    ifstream finPolicy("data/trt/policys.bin", ios::in | ios::binary);
    finPolicy.seekg(outputPolicySize * skipSample * sizeof(float));
    std::vector<float> expectedPolicy(outputPolicySize * batchSize);
    finPolicy.read((char *)expectedPolicy.data(), outputPolicySize * batchSize * sizeof(float));
    ifstream finValue("data/trt/values.bin", ios::in | ios::binary);
    finValue.seekg(outputValueSize * skipSample * sizeof(float));
    std::vector<float> expectedValue(outputValueSize * batchSize);
    finValue.read((char *)expectedValue.data(), outputValueSize * batchSize * sizeof(float));
    bool ok = true;
    ok &= compareResult(expectedPolicy.data(), outputPolicy, outputPolicySize * batchSize);
    ok &= compareResult(expectedValue.data(), outputValue, outputValueSize * batchSize);

    return ok;
}

std::vector<ShogiOnnx *> runnerForGPU;

void threadMain(int device, int threadInDevice)
{
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::uniform_int_distribution<> batchSizeDist(batchSizeMin, batchSizeMax); //ランダムにバッチサイズを選択
    if (cudaSetDevice(device) != cudaSuccess)
    {
        gLogError << "cudaSetDevice failed" << std::endl;
        return;
    }

    ShogiOnnx *pRunner;
    if (threadInDevice == 0)
    {
        pRunner = new ShogiOnnx();
        if (checkSerializedFile())
        {
            gLogInfo << "using serialized file" << std::endl;
            if (!pRunner->load())
            {
                gLogError << "load failed" << std::endl;
                return;
            }
        }
        else
        {
            if (!pRunner->build())
            {
                gLogError << "build failed" << std::endl;
                return;
            }
            if (device == 0)
            {
                pRunner->serialize();
            }
        }

        // dummy run
        for (int dummyBatchSize = batchSizeMin; dummyBatchSize <= batchSizeMax; dummyBatchSize *= 2)
        {
            if (!pRunner->infer(dummyBatchSize))
            {
                gLogError << "dummy infer failed" << std::endl;
                return;
            }
        }
        runnerForGPU[device] = pRunner;
    }

    gLogInfo << "dummy infer end" << std::endl;
    nThreadInitialized++;

    while (waitBenchStart)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    pRunner = runnerForGPU[device];

    std::vector<int> counts(batchSizeMax + 1);
    std::vector<long long> timesum(batchSizeMax + 1);
    while (continueBench)
    {
        int batchSize = batchSizeDist(engine);
        gpuMutexes[device]->lock();
        timespec timestart, timeend;
        clock_gettime(CLOCK_REALTIME, &timestart);
        if (!pRunner->infer(batchSize))
        {
            gLogError << "infer failed" << std::endl;
            return;
        }
        clock_gettime(CLOCK_REALTIME, &timeend);
        gpuMutexes[device]->unlock();
        long long nsec = ((long long)timeend.tv_sec * 1000000000LL + (long long)timeend.tv_nsec) - ((long long)timestart.tv_sec * 1000000000LL + (long long)timestart.tv_nsec);
        counts[batchSize]++;
        timesum[batchSize] += nsec;
    }
    resultMutex.lock();
    int allCounts = 0;
    for (int bs = 0; bs <= batchSizeMax; bs++)
    {
        allCounts += bs * counts[bs];
        totalCounts[bs] += counts[bs];
        totalTimesum[bs] += timesum[bs];
    }
    std::cout << "GPU " << device << " thread " << threadInDevice << " samples " << allCounts << std::endl;
    resultMutex.unlock();
}

int main(int argc, char **argv)
{
    if (argc != 9)
    {
        std::cerr << "usage: multi_gpu_bench nGPU nThreadPerGPU batchSizeMin batchSizeMax benchTime verify suppressStdout fpbit" << std::endl;
        return 1;
    }
    nGPU = atoi(argv[1]);
    nThreadPerGPU = atoi(argv[2]);
    batchSizeMin = atoi(argv[3]);
    batchSizeMax = atoi(argv[4]);
    int benchTime = atoi(argv[5]);
    verifyMode = atoi(argv[6]) != 0;
    bool suppressStdout = atoi(argv[7]) != 0;
    int fpbit = atoi(argv[8]);
    if (fpbit == 8)
    {
        fp8 = true;
    }
    else if (fpbit == 16)
    {
        fp16 = true;
    }
    if (suppressStdout)
    {
        // TensorRTから発生するメッセージを抑制(gLogError << "")
        setReportableSeverity(Logger::Severity::kINTERNAL_ERROR);
    }
    inputTensorNames.push_back("input");
    outputTensorNames.push_back("output_policy");
    outputTensorNames.push_back("output_value");
    continueBench = true;
    waitBenchStart = true;
    for (int device = 0; device < nGPU; device++)
    {
        gpuMutexes.push_back(new std::mutex());
    }
    totalCounts.resize(batchSizeMax + 1);
    totalTimesum.resize(batchSizeMax + 1);
    runnerForGPU.resize(nGPU);
    nThreadInitialized = 0;
    int nThreads = nGPU * nThreadPerGPU;
    std::vector<std::thread *> threads;
    gLogInfo << "Initializing " << nThreads << " threads" << std::endl;
    for (int device = 0; device < nGPU; device++)
    {
        for (int threadInDevice = 0; threadInDevice < nThreadPerGPU; threadInDevice++)
        {
            threads.push_back(new std::thread(threadMain, device, threadInDevice));
        }
    }

    while (nThreadInitialized < nThreads)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    waitBenchStart = false;
    gLogInfo << "Benchmarking..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(benchTime));
    gLogInfo << "Stopping..." << std::endl;
    continueBench = false;
    for (size_t threadIdx = 0; threadIdx < threads.size(); threadIdx++)
    {
        threads[threadIdx]->join();
    }

    int allCounts = 0;
    std::cout << "batch_size,avg_ms,nps" << std::endl;
    for (int bs = 0; bs <= batchSizeMax; bs++)
    {
        allCounts += totalCounts[bs] * bs;
        if (totalCounts[bs] > 0)
        {
            double avg_ns = (double)totalTimesum[bs] / (double)totalCounts[bs];
            int nps = (int)((double)bs * 1000000000LL / avg_ns);
            std::cout << bs << "," << avg_ns / 1000000.0 << "," << nps << std::endl;
        }
    }
    std::cout << "Total nps: " << ((double)allCounts / benchTime) << std::endl;
    return 0;
}
