#include <iostream>
#include <vector>
#include <fstream>
#include <iterator>
#include <numeric>
#include <onnxruntime_c_api.h>
#include <opencv2/opencv.hpp>  // For image loading and preprocessing

// Define a function to read an image and preprocess it for the model
std::vector<float> preprocessImage(const std::string& imagePath) {
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(28, 28));  // Resize to model input size

    // Normalize and flatten the image
    std::vector<float> input;
    for (int i = 0; i < resizedImage.rows; ++i) {
        for (int j = 0; j < resizedImage.cols; ++j) {
            input.push_back((resizedImage.at<uchar>(i, j) / 255.0f - 0.5f) / 0.5f);  // Normalize
        }
    }

    return input;
}

int main() {
    OrtEnv* env = nullptr;
    OrtStatus* status = nullptr;

    // Initialize the ONNX Runtime environment
    status = OrtCreateEnv(ORT_LOGGING_LEVEL_WARNING, "myEnv", &env);
    if (status != nullptr) {
        std::cerr << "Error creating environment: " << OrtGetErrorMessage(status) << std::endl;
        OrtReleaseStatus(status);
        return 1;
    }

    // Path to the ONNX model file
    const char* modelPath = "mnist_model.onnx";

    // Create a session options object if needed
    OrtSessionOptions* sessionOptions = nullptr;
    OrtCreateSessionOptions(&sessionOptions);

    // Create a session and load the ONNX model
    OrtSession* session = nullptr;
    status = OrtCreateSession(env, modelPath, sessionOptions, &session);
    if (status != nullptr) {
        std::cerr << "Error creating session: " << OrtGetErrorMessage(status) << std::endl;
        OrtReleaseStatus(status);
        OrtReleaseEnv(env);
        return 1;
    }

    // Read and preprocess the input image
    std::string imagePath = "sample.jpg";
    std::vector<float> inputTensor = preprocessImage(imagePath);

    // Set up input tensor with the preprocessed image data
    size_t inputTensorSize = inputTensor.size();
    std::vector<const char*> inputNodeNames = {"input_name"};  // Name of the input node in the ONNX model
    std::vector<int64_t> inputNodeDims = {1, 1, 28, 28};  // Assuming input shape is (1, 1, 28, 28)

    OrtValue* inputTensorValue = nullptr;
    OrtCreateTensorWithDataAsOrtValue(inputNodeDims.data(), inputNodeDims.size(), ORT_FLOAT,
                                      inputTensor.data(), inputTensorSize * sizeof(float),
                                      OrtAllocatorType::OrtDeviceAllocator, &inputTensorValue);

    // Run inference
    std::vector<const char*> outputNodeNames = {"output_name"};  // Name of the output node in the ONNX model
    OrtValue* outputTensorValue = nullptr;

    status = OrtRun(session, nullptr, inputNodeNames.data(), &inputTensorValue, 1,
                     outputNodeNames.data(), 1, &outputTensorValue);
    if (status != nullptr) {
        std::cerr << "Error running inference: " << OrtGetErrorMessage(status) << std::endl;
        OrtReleaseStatus(status);
        OrtReleaseEnv(env);
        return 1;
    }

    // Get the output tensor data
    float* outputData = nullptr;
    OrtGetTensorMutableData(outputTensorValue, reinterpret_cast<void**>(&outputData));

    // Display the model's output (assuming it's a single float value for classification)
    std::cout << "Model output: " << *outputData << std::endl;

    // Clean up
    OrtReleaseValue(inputTensorValue);
    OrtReleaseValue(outputTensorValue);
    OrtReleaseSession(session);
    OrtReleaseSessionOptions(sessionOptions);
    OrtReleaseEnv(env);

    return 0;
}
