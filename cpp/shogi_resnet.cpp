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

//!
//! sampleOnnxMNIST.cpp
//! This file contains the implementation of the ONNX MNIST sample. It creates the network using
//! the MNIST onnx model.
//! It can be run with the following command line:
//! Command: ./sample_onnx_mnist [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//! [--useDLACore=<int>]
//!

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

const int batchSize = 64;
const int skipSample = 0;
std::vector<std::string> inputTensorNames;
std::vector<std::string> outputTensorNames;

//! \brief  The SampleOnnxMNIST class implements the ONNX MNIST sample
//!
//! \details It creates the network using an ONNX model
//!
class SampleOnnxMNIST
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleOnnxMNIST()
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

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputPolicyDims; //!< The dimensions of the output to the network.
    nvinfer1::Dims mOutputValueDims; //!< The dimensions of the output to the network.

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool SampleOnnxMNIST::build()
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
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(inputTensorNames[0].c_str(), OptProfileSelector::kMIN, Dims4{1,119,9,9});
    profile->setDimensions(inputTensorNames[0].c_str(), OptProfileSelector::kOPT, Dims4{batchSize,119,9,9});
    profile->setDimensions(inputTensorNames[0].c_str(), OptProfileSelector::kMAX, Dims4{batchSize,119,9,9});
    config->addOptimizationProfile(profile);

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
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
bool SampleOnnxMNIST::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    // [W] [TRT] Calling isShapeTensor before the entire network is constructed may result in an inaccurate result.
    auto parsed = parser->parseFromFile(
        "data/trt/model.onnx", static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    builder->setMaxBatchSize(batchSize);//PARAM
    config->setMaxWorkspaceSize(1024_MiB);
    if (false)//PARAM FP16
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (false)//PARAM FP16
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
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
bool SampleOnnxMNIST::infer()
{
    gLogInfo << "createExecutionContext" << std::endl;
    auto context = mEngine->createExecutionContext();
    if (!context)
    {
        return false;
    }
    context->setBindingDimensions(0, Dims4{batchSize,119,9,9});
    gLogInfo << "BufferManager" << std::endl;
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, batchSize, context);

    gLogInfo << "processInput" << std::endl;
    // Read the input data into the managed buffers
    assert(inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        return false;
    }

    gLogInfo << "copyInputToDevice" << std::endl;
    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    gLogInfo << "executeV2" << std::endl;
    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    gLogInfo << "copyOutputToHost" << std::endl;
    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    gLogInfo << "verifyOutput" << std::endl;
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
bool SampleOnnxMNIST::processInput(const samplesCommon::BufferManager& buffers)
{
    const int inputC = mInputDims.d[1];
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];

    ifstream fin("data/trt/inputs.bin", ios::in|ios::binary);
    fin.seekg(inputC*inputH*inputW*sizeof(float)*skipSample);

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(inputTensorNames[0]));
    fin.read((char*)hostDataBuffer, inputC*inputH*inputW*sizeof(float)*batchSize);

    return true;
}

bool compareResult(float* expected, float* actual, size_t size) {
    float maxDiff = 0.0F;
    size_t maxDiffIdx = 0;
    for (size_t i = 0; i < size; i++) {
        float diff = abs(expected[i]-actual[i]);
        if (diff > maxDiff) {
            maxDiff = diff;
            maxDiffIdx = i;
        }
    }

    gLogInfo << "max diff among " << size << " elements: [" << maxDiffIdx << "] " << expected[maxDiffIdx] << "!=" << actual[maxDiffIdx] << std::endl;
    return maxDiff < 1e-3;
}
//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool SampleOnnxMNIST::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const int outputPolicySize = mOutputPolicyDims.d[1];
    gLogInfo << "policysize " << mOutputPolicyDims.d[0] << "," << mOutputPolicyDims.d[1] << std::endl;
    float* outputPolicy = static_cast<float*>(buffers.getHostBuffer(outputTensorNames[0]));
    const int outputValueSize = mOutputValueDims.d[1];
    float* outputValue = static_cast<float*>(buffers.getHostBuffer(outputTensorNames[1]));
    gLogInfo << "valuesize " << mOutputValueDims.d[0] << "," << mOutputValueDims.d[1] << std::endl;
    ifstream finPolicy("data/trt/policys.bin", ios::in|ios::binary);
    finPolicy.seekg(outputPolicySize*skipSample*sizeof(float));
    std::vector<float> expectedPolicy(outputPolicySize*batchSize);
    finPolicy.read((char*)expectedPolicy.data(), outputPolicySize*batchSize*sizeof(float));
    ifstream finValue("data/trt/values.bin", ios::in|ios::binary);
    finValue.seekg(outputValueSize*skipSample*sizeof(float));
    std::vector<float> expectedValue(outputValueSize*batchSize);
    finValue.read((char*)expectedValue.data(), outputValueSize*batchSize*sizeof(float));
    for (int i = 0; i < 10; i++) {
        gLogInfo << "policy " << i << " " << outputPolicy[i] << "," << expectedPolicy[i] << std::endl;
    }
    bool ok = true;
    ok &= compareResult(expectedPolicy.data(), outputPolicy, outputPolicySize*batchSize);
    for (int i = 0; i < 2; i++) {
        gLogInfo << "value " << i << " " << outputValue[i] << "," << expectedValue[i] << std::endl;
    }
    ok &= compareResult(expectedValue.data(), outputValue, outputValueSize*batchSize);


    return ok;
}

int main(int argc, char** argv)
{
    inputTensorNames.push_back("input");
    outputTensorNames.push_back("output_policy");
    outputTensorNames.push_back("output_value");
    SampleOnnxMNIST sample;


    if (!sample.build())
    {
        gLogInfo << "build failed" << std::endl;
    }
    if (!sample.infer())
    {
        gLogInfo << "infer failed" << std::endl;
    }

}
