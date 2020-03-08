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

#include "common.h"
#include "buffers.h"

static int batchSizeMin = 1;
static int batchSizeMax = 256;
static int batchSize = 1;
const bool fp16 = false;
const bool verifyMode = false;
static int skipSample = 0;
static std::vector<std::string> inputTensorNames;
static std::vector<std::string> outputTensorNames;

std::string addProfileSuffix(const std::string& name, int profile)
{
    std::ostringstream oss;
    oss << name;
    if (profile > 0)
    {
    oss << " [profile " << profile << "]";
    }
    
    return oss.str();
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

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

private:
    nvinfer1::Dims mInputDims;        //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputPolicyDims; //!< The dimensions of the output to the network.
    nvinfer1::Dims mOutputValueDims;  //!< The dimensions of the output to the network.

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;        //!< The TensorRT engine used to run the network
    std::map<int, std::shared_ptr<nvinfer1::IExecutionContext> > mContextForProfile;
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
    bool processInput(const samplesCommon::BufferManager &buffers);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const samplesCommon::BufferManager &buffers);
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
    if (false)
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
    if (fp16)
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
bool ShogiOnnx::infer()
{
    auto mContext = mContextForProfile.at(profileForBatchSize[batchSize]);
    std::string inputBindingName = addProfileSuffix(inputTensorNames[0], profileForBatchSize[batchSize]);
    int bidx = mEngine->getBindingIndex(inputBindingName.c_str());
    mContext->setBindingDimensions(bidx, Dims4{batchSize, 119, 9, 9});
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, batchSize, mContext.get());

    // Read the input data into the managed buffers
    assert(inputTensorNames.size() == 1);
    if (!processInput(buffers))
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
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool ShogiOnnx::processInput(const samplesCommon::BufferManager &buffers)
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
bool ShogiOnnx::verifyOutput(const samplesCommon::BufferManager &buffers)
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

int main(int argc, char **argv)
{
    inputTensorNames.push_back("input");
    outputTensorNames.push_back("output_policy");
    outputTensorNames.push_back("output_value");
    ShogiOnnx sample;

    if (!sample.build())
    {
        gLogInfo << "build failed" << std::endl;
        return 1;
    }
    if (verifyMode)
    {
        gLogInfo << "verify on" << std::endl;
    }
    else
    {
        gLogInfo << "verify off" << std::endl;
    }

    gLogInfo << "first infer" << std::endl;
    if (!sample.infer())
    {
        gLogInfo << "infer failed" << std::endl;
    }
    gLogInfo << "first infer end" << std::endl;
    std::vector<int> counts(batchSizeMax + 1);
    std::vector<long long> timesum(batchSizeMax + 1);
    for (int i = 0; i < 1000; i++)
    {
        batchSize = i % (batchSizeMax - batchSizeMin + 1) + batchSizeMin;
        skipSample = i;
        timespec timestart, timeend;
        clock_gettime(CLOCK_REALTIME, &timestart);
        if (!sample.infer())
        {
            gLogInfo << "infer failed" << std::endl;
            return 1;
        }
        clock_gettime(CLOCK_REALTIME, &timeend);
        long long nsec = ((long long)timeend.tv_sec * 1000000000LL + (long long)timeend.tv_nsec) - ((long long)timestart.tv_sec * 1000000000LL + (long long)timestart.tv_nsec);
        counts[batchSize]++;
        timesum[batchSize] += nsec;
    }
    gLogInfo << "all infer end" << std::endl;
    std::cout << "batch_size,avg_ms,nps" << std::endl;
    for (int bs = 0; bs <= batchSizeMax; bs++)
    {
        if (counts[bs] > 0)
        {
            double avg_ns = (double)timesum[bs] / (double)counts[bs];
            int nps = (int)((double)bs * 1000000000LL / avg_ns);
            std::cout << bs << "," << avg_ns / 1000000.0 << "," << nps << std::endl;
        }
    }
    return 0;
}
