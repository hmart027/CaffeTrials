//============================================================================
// Name        : CaffeTrials.cpp
// Author      : Harold Martin
// Version     :
// Copyright   : Your copyright notice
// Description : I copied most of this code from somewhere on the internet but cant remember where.
//============================================================================

#include <iostream>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <memory>

#include <algorithm>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;

Blob<float>* input_layer;
Blob<float>* output_layer;

vector<int> in_shape;
vector<int> out_shape;

shared_ptr<Net<float> > net_;
cv::Size input_geometry_;
int num_channels_;
cv::Mat mean_;
std::vector<string> labels_;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

void SetMean(const string& mean_file);

std::vector<float> Predict(const cv::Mat& img);

void WrapInputLayer(std::vector<cv::Mat>* input_channels);

void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
}

/* Load the mean file in binaryproto format. */
void SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}


int main() {
	std::cout << "!!!Hello World!!!" << std::endl; // prints !!!Hello World!!!

	string model_file   = "/media/DATA/INS/caffe-master/models/bvlc_reference_caffenet/deploy.prototxt";
	string trained_file = "/media/DATA/INS/caffe-master/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel";
	string mean_file    = "/media/DATA/INS/caffe-master/data/ilsvrc12/imagenet_mean.binaryproto";
	string label_file   = "/media/DATA/INS/caffe-master/data/ilsvrc12/synset_words.txt";

	::google::InitGoogleLogging("CaffeTrials");

	Caffe::set_mode(Caffe::GPU);
	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));

	std::cout << "Loading Trained Model" << std::endl;
	net_->CopyTrainedLayersFrom(trained_file);
	std::cout << "Model Loaded" << std::endl;

	input_layer = net_->input_blobs()[0];
	output_layer = net_->output_blobs()[0];

	in_shape = input_layer->shape();
	out_shape = output_layer->shape();

	num_channels_ = in_shape[1];
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	std::cout << input_layer->shape_string() << std::endl;
	std::cout << output_layer->shape_string() << std::endl;

	/* Load the binaryproto mean file. */
	SetMean(mean_file);

	/* Load labels. */
	std::ifstream labels(label_file.c_str());
	CHECK(labels) << "Unable to open labels file " << label_file;
	string line;
	while (std::getline(labels, line))
		labels_.push_back(string(line));

	std::cout << labels_.size() << " labels loaded" << std::endl;

	string file = "/home/harold/Pictures/Wallpapers/cat.jpg";
//	string file = "/media/DATA/INS/caffe-master/examples/images/cat.jpg";

	cv::Mat img = cv::imread(file, CV_LOAD_IMAGE_COLOR);
	CHECK(!img.empty()) << "Unable to decode image " << file;
	std::vector<Prediction> predictions = Classify(img);

	/* Print the top N predictions. */
	for (size_t i = 0; i < predictions.size(); ++i) {
		Prediction p = predictions[i];
		std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
				<< p.first << "\"" << std::endl;
	}

	cv::VideoCapture cap;
	if(!cap.open(0)){
		std::cout<<"No video source"<<std::endl;
		return 0;
	}
	for(;;){
		cv::Mat frame;
		cap>>frame;
		if(frame.empty()) break;
		imshow("Video", frame);

		std::vector<Prediction> predictions = Classify(frame);
		std::cout<<std::endl;
		/* Print the top N predictions. */
		for (size_t i = 0; i < predictions.size(); ++i) {
			Prediction p = predictions[i];
			std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
					<< p.first << "\"" << std::endl;
		}

		if(cv::waitKey(1)==32) break;
	}

	return 0;
}
