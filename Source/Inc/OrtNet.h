#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cassert>
#include "onnxruntime_cxx_api.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <QImage>

class OrtNet
{
public:
	OrtNet();
	~OrtNet();


    void Init(const char* model_path);

	// Ort::Value getInputTensor(Mat blob);
    void setInputTensor(const cv::Mat& frame, bool side);
    void forward(bool side);
    QImage getProcessedFrame(bool side);

    void postprocess(cv::Mat& frame, bool side);

private:
	// Ort Environment
    Ort::Env env = Ort::Env(nullptr);
	Ort::Session session = Ort::Session(nullptr);
	Ort::SessionOptions session_options;
	Ort::Allocator allocator = Ort::Allocator::CreateDefault();
	Ort::AllocatorInfo allocator_info = Ort::AllocatorInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);


    // Model
	// Inputs
	std::vector<const char*> input_node_names = std::vector<const char*>();
	std::vector<size_t> input_node_sizes = std::vector<size_t>();
	std::vector<std::vector<int64_t>> input_node_dims = std::vector<std::vector<int64_t>>();
    // Ort::Value input_tensor = Ort::Value(nullptr);
    Ort::Value input_tensor_right = Ort::Value(nullptr);
    Ort::Value input_tensor_left = Ort::Value(nullptr);

	// Outputs
	std::vector<const char*>output_node_names = std::vector<const char*>();
	std::vector<size_t> output_node_sizes = std::vector<size_t>();
	std::vector<std::vector<int64_t>> output_node_dims = std::vector<std::vector<int64_t>>();
    // std::vector<Ort::Value> output_tensor = std::vector<Ort::Value>();
    std::vector<Ort::Value> output_tensor_right = std::vector<Ort::Value>();
    std::vector<Ort::Value> output_tensor_left = std::vector<Ort::Value>();

    // float* scores = NULL;
    // float* boxes = NULL;
    // std::pair<float*, float*> outs = std::pair<float*, float*>();
    // cv::Mat frame;
    cv::Mat frame_right;
    cv::Mat frame_left;


    unsigned int input_width = 320;
    unsigned int input_height = 320;
    float confThreshold = 0.5f;
    float nmsThreshold = 0.4f;
    std::vector<std::string> classes;

    void drawPred(cv::Mat& frame, int classId, float conf, int left, int top, int right, int bottom);
};
