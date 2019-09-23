#include "OrtNet.h"
#include <QtCore/QFile>
#include <QtCore/QJsonParseError>
#include <QtCore/QJsonArray>
#include <QtCore/QJsonObject>
#include <QDebug>

OrtNet::OrtNet() {}
OrtNet::~OrtNet() {}

// extern Ort::Env ortEnv;

void OrtNet::Init(const char* model_path) {

    env = Ort::Env(ORT_LOGGING_LEVEL_FATAL, "OrtEnv");
    session_options.SetThreadPoolSize(4);
    session_options.SetGraphOptimizationLevel(2);
    // session_options.EnableSequentialExecution();

    // std::cout << "Using Onnxruntime C++ API" << std::endl;
    session = Ort::Session(env, model_path, session_options);

	// print number of model input nodes
    int num_input_nodes = session.GetInputCount();
	input_node_names = std::vector<const char*>(num_input_nodes);
    // std::cout << "Number of inputs = " << num_input_nodes << std::endl;

	// iterate over all input nodes
    for (int i = 0; i < num_input_nodes; i++) {
		// print input node names
		char* input_name = session.GetInputName(i, allocator);
        // printf("Input %d : name=%s\n", i, input_name);
		input_node_names[i] = input_name;

		// print input node types
		Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
		input_node_sizes.push_back(tensor_info.GetElementCount());

        // ONNXTensorElementDataType type = tensor_info.GetElementType();
        // printf("Input %d : type=%d\n", i, type);

		// print input shapes/dims
		input_node_dims.push_back(tensor_info.GetShape());
        // printf("Input %d : num_dims=%zu\n", i, input_node_dims[i].size());
        // for (size_t j = 0; j < input_node_dims[i].size(); j++)
        //	printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[i][j]);
	}

	// print number of model output nodes
    int num_output_nodes = session.GetOutputCount();
	output_node_names = std::vector<const char*>(num_output_nodes);
    // std::cout << "Number of outputs = " << num_output_nodes << std::endl;

	// iterate over all input nodes
	for (int i = 0; i < num_output_nodes; i++) {
		// print output node names
		char* output_name = session.GetOutputName(i, allocator);
        // printf("Output %d : name=%s\n", i, output_name);
		output_node_names[i] = output_name;

		// print output node types
		Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
		output_node_sizes.push_back(tensor_info.GetElementCount());

        // ONNXTensorElementDataType type = tensor_info.GetElementType();
        // printf("Output %d : type=%d\n", i, type);

		// print output shapes/dims
		output_node_dims.push_back(tensor_info.GetShape());
        // printf("Output %d : num_dims=%zu\n", i, output_node_dims[i].size());
        // for (size_t j = 0; j < output_node_dims[i].size(); j++)
        //	printf("Output %d : dim %d=%jd\n", i, j, output_node_dims[i][j]);
	}

    QFile inFile("/home/ierturk/Work/REPOs/ssd/yoloData/clk/train.json");
     inFile.open(QIODevice::ReadOnly|QIODevice::Text);
     QByteArray data = inFile.readAll();
     inFile.close();

     QJsonParseError errorPtr{};
     QJsonDocument doc = QJsonDocument::fromJson(data, &errorPtr);
     if (doc.isNull()) {
         qDebug() << "Parse failed";

     }

     QJsonObject rootObj = doc.object();
     QJsonArray ptsArray = rootObj.value("categories").toArray();

     for(auto && i : ptsArray) {
         classes.emplace_back(i.toObject().value("name").toString().toUtf8().constData());
     }
}


void OrtNet::setInputTensor(const cv::Mat& frame, bool side)
{
	static cv::Mat blob;
    blob = cv::dnn::blobFromImage(
        frame,
        1.0,
        cv::Size(320,320),
        cv::Scalar(123, 117, 104),
        true,
        false,
        CV_32F);

    if(side) {

        iosRight->frame = frame;
        iosRight->input_tensor = Ort::Value::CreateTensor<float>(
            allocator_info,
            blob.ptr<float>(),
            input_node_sizes[0],
            input_node_dims[0].data(),
            input_node_dims[0].size());
        assert(iosRight->input_tensor.IsTensor());

    } else {
        iosLeft->frame = frame;

        iosLeft->input_tensor = Ort::Value::CreateTensor<float>(
            allocator_info,
            blob.ptr<float>(),
            input_node_sizes[0],
            input_node_dims[0].data(),
            input_node_dims[0].size());
        assert(iosLeft->input_tensor.IsTensor());
    }
}

void OrtNet::forward(bool side)
{
    auto start = std::chrono::high_resolution_clock::now();

    if(side) {
        iosRight->output_tensor = session.Run(
            Ort::RunOptions{ nullptr },
            input_node_names.data(),
            &(iosRight->input_tensor),
            input_node_names.size(),
            output_node_names.data(),
            output_node_names.size());
    } else {
        iosLeft->output_tensor = session.Run(
            Ort::RunOptions{ nullptr },
            input_node_names.data(),
            &(iosLeft->input_tensor),
            input_node_names.size(),
            output_node_names.data(),
            output_node_names.size());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Inference time : " << duration.count() << " ms" << '\n';

    // scores = output_tensor[0].GetTensorMutableData<float>();
    // boxes = output_tensor[1].GetTensorMutableData<float>();
    // outs = std::make_pair(scores, boxes);
}

QImage OrtNet::getProcessedFrame(bool side) {

    static cv::Mat frame;
    if(side)
        frame = iosRight->frame;
    else
        frame = iosLeft->frame;

    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    postprocess(frame, side);
    return QImage(
            frame.data,
            frame.cols,
            frame.rows,
            frame.step,
            QImage::Format_RGB888);
}

void OrtNet::postprocess(cv::Mat& frame, bool side)
{
    std::vector<int> p_classIds;
    std::vector<float> p_confidences;
    std::vector<cv::Rect> p_boxes;
    float* scores;
    float* boxes;

    if(side) {
        scores = iosRight->output_tensor[0].GetTensorMutableData<float>();
        boxes = iosRight->output_tensor[1].GetTensorMutableData<float>();

    } else {
        scores = iosLeft->output_tensor[0].GetTensorMutableData<float>();
        boxes = iosLeft->output_tensor[1].GetTensorMutableData<float>();
    }

    // CV_Assert(scores[0] > 0);

    for (size_t i = 0; i < 3234; i++) {
        for (size_t j = 1; j < 78; j++) {
            float confidence = scores[78 * i + j];
            if (confidence > confThreshold)
            {
                int left = (int)(boxes[4 * i] * frame.cols);
                int top = (int)(boxes[4 * i + 1] * frame.rows);
                int right = (int)(boxes[4 * i + 2] * frame.cols);
                int bottom = (int)(boxes[4 * i + 3] * frame.rows);
                int width = right - left + 1;
                int height = bottom - top + 1;

                p_classIds.emplace_back((int)j - 1);;
                p_confidences.emplace_back((float)confidence);
                p_boxes.emplace_back(cv::Rect(left, top, width, height));
            }
        }
    }

    std::vector<int> p_indices;
    cv::dnn::NMSBoxes(p_boxes, p_confidences, confThreshold, nmsThreshold, p_indices);
    for (size_t i = 0; i < p_indices.size(); ++i)
    {
        int idx = p_indices[i];
        cv::Rect box = p_boxes[idx];
        drawPred(frame, p_classIds[idx], p_confidences[idx], box.x, box.y,
            box.x + box.width, box.y + box.height);
    }
}

void OrtNet::drawPred(cv::Mat& frame, int classId, float conf, int left, int top, int right, int bottom)
{
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0));

    std::string label = cv::format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ": " + label;
    }

    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = cv::max(top, labelSize.height);

    cv::rectangle(frame,
                  cv::Point(left, top - labelSize.height),
                  cv::Point(left + labelSize.width, top + baseLine),
                  cv::Scalar::all(255),
                  cv::FILLED);

    cv::putText(frame,
                label,
                cv::Point(left, top),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar());
}
