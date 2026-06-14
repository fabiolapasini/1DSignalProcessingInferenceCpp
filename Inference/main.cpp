#include <NvInfer.h>
#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>


// check memory for errors
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


// logger
class Logger : public nvinfer1::ILogger
{
public:
	void log(Severity severity, const char* msg) noexcept override
	{
		if (severity <= Severity::kINFO)
		{
			std::cout << "[TensorRT] " << msg << std::endl;
		}
	}
};
Logger g_logger;


std::vector<char> LoadEngine(const std::string& path)
{
	std::ifstream file(path, std::ios::binary);
	if (!file) throw std::runtime_error("Cannot open engine");

	file.seekg(0, std::ios::end);
	size_t size = file.tellg();
	file.seekg(0);

	std::vector<char> data(size);
	file.read(data.data(), size);
	return data;
}


size_t GetTensorSize(nvinfer1::IExecutionContext& ctx, const char* name)
{
	nvinfer1::Dims dims = ctx.getTensorShape(name);

	size_t total = 1;
	for (int i = 0; i < dims.nbDims; i++)
		total *= dims.d[i];

	return total * sizeof(float);
}


void PrintEngineInfo(nvinfer1::ICudaEngine* engine)
{
	int nb_tensors = engine->getNbIOTensors();

	std::cout << "\n========== TensorRT Engine Info ==========\n";
	std::cout << "Total IO tensors: " << nb_tensors << "\n\n";

	for (int i = 0; i < nb_tensors; i++) {
		const char* name = engine->getIOTensorName(i);
		auto mode = engine->getTensorIOMode(name);
		auto dims = engine->getTensorShape(name);

		std::cout << "Tensor #" << i << "\n";
		std::cout << "  Name : " << name << "\n";
		std::cout << "  Type : " << (mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT") << "\n";
		std::cout << "  Shape: [";

		for (int d = 0; d < dims.nbDims; d++) {
			std::cout << dims.d[d];
			if (d < dims.nbDims - 1) std::cout << ", ";
		}

		std::cout << "]\n\n";
	}

	std::cout << "=========================================\n\n";
}


void GenerateFakeSignal(std::vector<float>& signal)
{
	signal.resize(1024, 0.0f);

	struct Peak {
		int center;
		float height;
		float width;
	};

	std::vector<Peak> peaks = {
		{200, 1.0f, 20.0f},
		{500, 0.6f, 40.0f},
		{800, 0.9f, 30.0f}
	};

	for (const auto& p : peaks) {
		for (int i = 0; i < 1024; i++) {
			float x = static_cast<float>(i);
			float diff = (x - p.center) / p.width;
			float value = p.height * std::exp(-0.5f * diff * diff);
			signal[i] += value;
		}
	}
}


void LoadTestSignal(std::vector<float>& signal)
{
	signal.resize(1024);
	std::ifstream file("Assets/test_signal.raw", std::ios::binary);
	if (!file) {
		throw std::runtime_error("Cannot open test_signal.raw");
	}

	file.read(
		reinterpret_cast<char*>(signal.data()),
		1024 * sizeof(float)
	);

	if (!file) {
		throw std::runtime_error("Failed to read full signal");
	}
}


void DrawPeaks(
	const std::vector<float>& signal, const std::vector<float>& pt_positions, cv::Mat& canvas, int row)
{
	int width = 1024;
	int height = 300;
	float threshold = 0.005f;
	size_t n = 3;

	cv::Mat plot(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

	// rescaling
	float min_v = *std::min_element(signal.begin(), signal.end());
	float max_v = *std::max_element(signal.begin(), signal.end());
	auto toY = [&](float v) {
		return height - (int)((v - min_v) / (max_v - min_v + 1e-6f) * (height - 20)) - 10;
	};

	// draw signal
	for (int i = 1; i < std::min((int)signal.size(), width); i++) {
		cv::line(plot, cv::Point(i - 1, toY(signal[i - 1])), cv::Point(i, toY(signal[i])),
			cv::Scalar(150, 110, 80), 1, cv::LINE_AA);
	}

	// best peaks
	std::vector<int> pt_peaks;
	for (int i = 0; i < std::min((int)pt_positions.size(), width); ++i) {
		if (pt_positions[i] > threshold) {
			pt_peaks.push_back(i);
		}
	}

	std::sort(pt_peaks.begin(), pt_peaks.end(), [&](int a, int b) {
		return pt_positions[a] > pt_positions[b];
		});

	if (pt_peaks.size() > n) {
		pt_peaks.resize(n);
	}

	for (int p : pt_peaks) {
		if (p < 0 || p >= (int)signal.size()) continue;

		cv::circle(
			plot,
			cv::Point(p, toY(signal[p])),
			4,
			cv::Scalar(100, 180, 240),
			-1,
			cv::LINE_AA
		);
	}

	plot.copyTo(canvas(cv::Rect(0, row * height, width, height)));
}


// inference function
void RunInference(nvinfer1::ICudaEngine* engine, nvinfer1::IExecutionContext* context, const std::vector<float>& input_signal,
	std::vector<float>& positions, std::vector<float>& heights, std::vector<float>& widths)
{
	const char* input_name = engine->getIOTensorName(0);
	const char* pos_name = engine->getIOTensorName(1);
	const char* h_name = engine->getIOTensorName(2);
	const char* w_name = engine->getIOTensorName(3);

	void* input_buffer = nullptr;
	void* positions_buffer = nullptr;
	void* heights_buffer = nullptr;
	void* widths_buffer = nullptr;

	size_t input_size = GetTensorSize(*context, input_name);
	size_t positions_size = GetTensorSize(*context, pos_name);
	size_t heights_size = GetTensorSize(*context, h_name);
	size_t widths_size = GetTensorSize(*context, w_name);

	// input gpu tensor
	gpuErrChk(cudaMalloc(&input_buffer, input_size));
	gpuErrChk(cudaMemset(input_buffer, 0, input_size));
	gpuErrChk(cudaMemcpy(input_buffer, input_signal.data(), input_size, cudaMemcpyHostToDevice));
	context->setTensorAddress(input_name, input_buffer);

	// output gpu tensors
	gpuErrChk(cudaMalloc(&positions_buffer, positions_size));
	gpuErrChk(cudaMalloc(&heights_buffer, heights_size));
	gpuErrChk(cudaMalloc(&widths_buffer, widths_size));

	gpuErrChk(cudaMemset(positions_buffer, 0, positions_size));
	gpuErrChk(cudaMemset(heights_buffer, 0, heights_size));
	gpuErrChk(cudaMemset(widths_buffer, 0, widths_size));

	context->setTensorAddress(pos_name, positions_buffer);
	context->setTensorAddress(h_name, heights_buffer);
	context->setTensorAddress(w_name, widths_buffer);

	// stream CUDA
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// inference
	if (!context->enqueueV3(stream)) {
		throw std::runtime_error("enqueueV3 failed");
	}

	// sync
	cudaStreamSynchronize(stream);

	// copy output
	positions.resize(positions_size / sizeof(float));
	heights.resize(heights_size / sizeof(float));
	widths.resize(widths_size / sizeof(float));

	gpuErrChk(cudaMemcpy(positions.data(), positions_buffer, positions_size, cudaMemcpyDeviceToHost));
	gpuErrChk(cudaMemcpy(heights.data(), heights_buffer, heights_size, cudaMemcpyDeviceToHost));
	gpuErrChk(cudaMemcpy(widths.data(), widths_buffer, widths_size, cudaMemcpyDeviceToHost));

	// free
	cudaFree(input_buffer);
	cudaFree(positions_buffer);
	cudaFree(heights_buffer);
	cudaFree(widths_buffer);

	cudaStreamDestroy(stream);
}



int main()
{
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(g_logger);
	if (!runtime) {
		std::cerr << "Failed to create runtime\n";
		return -1;
	}

	std::vector<char> engine_data = LoadEngine("Assets/net.trt");

	nvinfer1::ICudaEngine* engine =
		runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());

	if (!engine) {
		std::cerr << "Engine load failed\n";
		return -1;
	}

	std::cout << "Engine loaded\n";

	nvinfer1::IExecutionContext* context = engine->createExecutionContext();
	if (!context) {
		std::cerr << "Context creation failed\n";
		engine->destroy();
		runtime->destroy();
		return -1;
	}

	PrintEngineInfo(engine);

	// fake signal
	// std::vector<float> input_signal;
	// GenerateFakeSignal(input_signal);

	std::vector<float> input_signal;
	LoadTestSignal(input_signal);

	std::vector<float> positions;
	std::vector<float> heights;
	std::vector<float> widths;

	// Inference
	RunInference(engine, context, input_signal, positions, heights, widths);
	size_t top_class = 0;
	for (size_t i = 1; i < heights.size(); ++i) {
		if (heights[i] > heights[top_class])
			top_class = i;
	}

	std::cout << "Positions size: " << positions.size() << "\n";
	std::cout << "Heights size: " << heights.size() << "\n";
	std::cout << "Widths size: " << widths.size() << "\n";

	std::cout << "Top peak index: " << top_class << "\n";
	std::cout << "Top peak height: " << heights[top_class] << "\n";
	std::cout << "Top peak position: " << positions[top_class] << "\n";
	std::cout << "Top peak width: " << widths[top_class] << "\n";


	cv::Mat canvas(300, 1024, CV_8UC3, cv::Scalar(255, 255, 255));
	DrawPeaks(input_signal, positions, canvas, 0);
	cv::imshow("tensorRT_pred", canvas);
	cv::imwrite("tensorRT_pred.png", canvas);

	// cleanup
	context->destroy();
	engine->destroy();
	runtime->destroy();

	return 0;
}