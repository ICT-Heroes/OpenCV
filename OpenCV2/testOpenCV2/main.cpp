/*

#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <opencv2\nonfree\nonfree.hpp>
#include <opencv2\ml\ml.hpp>
#include <opencv2\core\core.hpp>


#define NTRAINING_SAMPLES	100
#define FRAC_LINEAR_SEP		0.9f

using namespace std;
using namespace cv;

const int DIM_VECTOR = 128;


void writeSURF(const char* filename, CvSeq* imageKeypoints, CvSeq* imageDescriptors) {
	fstream fout;
	fout.open(filename, ios::out);
	if (!fout.is_open()) {
		cerr << "cannot open file:" << filename << endl;
		return;
	}

	//1행은 키 포인트의 수와 특징량의 차원수를 기록
	fout << imageKeypoints->total << ' ' << DIM_VECTOR << endl;
	
	//2행부터는 키포인트 정보와 특징벡터를 기록
	for (int i = 0; i < imageKeypoints->total; i++) {
		CvSURFPoint* point = (CvSURFPoint*)cvGetSeqElem(imageKeypoints, i);
		float* descriptor = (float*)cvGetSeqElem(imageDescriptors, i);
		//키 포인트 정보 기록( x좌표, y좌표, 사이즈, 라플라시안)
		fout << point->pt.x << ' ' << point->pt.y << ' ' << point->size << ' ' << point->laplacian << '\n';
		//특징벡터 정보기록
		for (int j = 0; j < DIM_VECTOR; j++) {
			fout << descriptor[j] << '\n';
		}
		fout << endl;
	}

	fout.close();
}

int main(int argc, char** argv) {

	// Data for visual representation
	const int WIDTH = 512, HEIGHT = 512;
	Mat canvas = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);

	//--------------------- 1. Set up training data randomly ---------------------------------------


	Mat trainData = Mat(2 * NTRAINING_SAMPLES, 2, CV_32FC1);
	Mat labels = Mat(2 * NTRAINING_SAMPLES, 1, CV_32FC1);

	RNG rng = RNG(100); // Random value generation class

	// Set up the linearly separable part of the training data
	int nLinearSamples = (int)(FRAC_LINEAR_SEP * NTRAINING_SAMPLES);

	// Generate random points for the class 1111111111111111111111111111111111111111111111
	Mat trainClass = trainData.rowRange(0, nLinearSamples);
	// The x coordinate of the points is in [0, 0.4)
	Mat c = trainClass.colRange(0, 1);
	rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(0.4 * WIDTH));
	// The y coordinate of the points is in [0, 1)
	c = trainClass.colRange(1, 2);
	rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(HEIGHT));

	// Generate random points for the class 22222222222222222222222222222222222222222222222
	trainClass = trainData.rowRange(2 * NTRAINING_SAMPLES - nLinearSamples, 2 * NTRAINING_SAMPLES);
	// The x coordinate of the points is in [0.6, 1]
	c = trainClass.colRange(0, 1);
	rng.fill(c, RNG::UNIFORM, Scalar(0.6*WIDTH), Scalar(WIDTH));
	// The y coordinate of the points is in [0, 1)
	c = trainClass.colRange(1, 2);
	rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(HEIGHT));

	//------------------ Set up the non-linearly separable part of the training data ---------------

	// Generate random points for the classes 1 and 2
	trainClass = trainData.rowRange(nLinearSamples, 2 * NTRAINING_SAMPLES - nLinearSamples);
	// The x coordinate of the points is in [0.4, 0.6)
	c = trainClass.colRange(0, 1);
	rng.fill(c, RNG::UNIFORM, Scalar(0.4*WIDTH), Scalar(0.6*WIDTH));
	// The y coordinate of the points is in [0, 1)
	c = trainClass.colRange(1, 2);
	rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(HEIGHT));

	//------------------------- Set up the labels for the classes ---------------------------------
	labels.rowRange(0, NTRAINING_SAMPLES).setTo(1);  // Class 1
	labels.rowRange(NTRAINING_SAMPLES, 2 * NTRAINING_SAMPLES).setTo(2);  // Class 2

	//------------------------ 2. Set up the support vector machines parameters --------------------
	CvSVMParams params;
	/*
	params.svm_type = SVM::C_SVC;
	params.C = 0.1;
	params.kernel_type = SVM::LINEAR;
	params.term_crit = TermCriteria(CV_TERMCRIT_ITER, (int)1e7, 1e-6);
	

	params.svm_type = SVM::C_SVC;
	params.kernel_type = SVM::POLY;
	params.gamma = 20;
	params.degree = 3;
	params.coef0 = 2000;

	params.C = 7;
	params.nu = 0.0;
	params.p = 0.0;

	params.class_weights = NULL;
	params.term_crit.type = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS;
	params.term_crit.max_iter = 1000;
	params.term_crit.epsilon = 1e-6;

	//------------------------ 3. Train the svm ----------------------------------------------------
	cout << "Starting training process" << endl;
	CvSVM svm;
	svm.train(trainData, labels, Mat(), Mat(), params);
	cout << "Finished training process" << endl;

	//------------------------ 4. Show the decision regions ----------------------------------------
	Vec3b green(0, 100, 0), blue(100, 0, 0);
	for (int i = 0; i < canvas.rows; ++i)
		for (int j = 0; j < canvas.cols; ++j)
		{
			Mat sampleMat = (Mat_<float>(1, 2) << i, j);
			float response = svm.predict(sampleMat);

			if (response == 1)    canvas.at<Vec3b>(j, i) = green;
			else if (response == 2)    canvas.at<Vec3b>(j, i) = blue;
		}

	//----------------------- 5. Show the training data --------------------------------------------
	int thick = -1;
	int lineType = 8;
	float px, py;
	// Class 1
	for (int i = 0; i < NTRAINING_SAMPLES; ++i)
	{
		px = trainData.at<float>(i, 0);
		py = trainData.at<float>(i, 1);
		circle(canvas, Point((int)px, (int)py), 3, Scalar(0, 255, 0), thick, lineType);
	}
	// Class 2
	for (int i = NTRAINING_SAMPLES; i <2 * NTRAINING_SAMPLES; ++i)
	{
		px = trainData.at<float>(i, 0);
		py = trainData.at<float>(i, 1);
		circle(canvas, Point((int)px, (int)py), 3, Scalar(255, 0, 0), thick, lineType);
	}

	//------------------------- 6. Show support vectors --------------------------------------------
	thick = 2;
	lineType = 8;
	int x = svm.get_support_vector_count();

	for (int i = 0; i < x; ++i)
	{
		const float* v = svm.get_support_vector(i);
		circle(canvas, Point((int)v[0], (int)v[1]), 6, Scalar(128, 128, 128), thick, lineType);
	}

	imwrite("result.png", canvas);                      // save the Image
	imshow("SVM for Non-Linear Training Data", canvas); // show it to the user
	waitKey(0);

	return 0;










	/*



	cv::initModule_nonfree();
	//const char* imageFile = argc == 3 ? argv[1] : "C:\work\beaber.jpeg";
	//const char* surfFile = argc == 3 ? argv[2] : "C:\work\beaber.surf";
	char* imageFile = "C:/work/car.jpg";
	char* surfFile = "C:/work/car.surf";


	//SURF 추출을 위한 입력 영상을 그레이스케일로 읽음
	IplImage* grayImage = cvLoadImage(imageFile, CV_LOAD_IMAGE_GRAYSCALE);
	if (!grayImage) {
		cerr << "cannot find image file:" << imageFile << endl;
		return -1;
	}
	CvSVM svm;
	//결과에 키 포인트를 표현하기 위해 컬러로도 읽음
	IplImage* colorImage = cvLoadImage(imageFile, CV_LOAD_IMAGE_COLOR);
	if (!colorImage) {
		cerr << "cannot find image file:" << imageFile << endl;
		return -1;
	}
	
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* imageKeypoint = 0;
	CvSeq* imageDescriptors = 0;
	CvSURFParams params = cvSURFParams(500, 1);

	//CvMSERParams mserParams = cvMSERParams();

	//영상으로부터 SURF 특징 추출
	//cvExtractMSER(grayImage, 0, &imageKeypoint, storage, mserParams);
	cvExtractSURF(grayImage, 0, &imageKeypoint, &imageDescriptors, storage, params);
	cout << "image Descriptors:" << imageDescriptors->total << endl;
	
	//SURF 정보 파일에 기록
	writeSURF(surfFile, imageKeypoint, imageDescriptors);

	//영상에 키포인트 표현
	for (int i = 0; i < imageKeypoint->total; i++) {
		CvSURFPoint* point = (CvSURFPoint*)cvGetSeqElem(imageKeypoint, i);
		CvPoint center, way;
		int radius;
		center.x = cvRound(point->pt.x);
		center.y = cvRound(point->pt.y);
		way.x = center.x - cos(point->dir) * point->size;
		way.y = center.y + sin(point->dir) * point->size;
		radius = cvRound(point->size * 12 / 9.0*2.0);

		cvCircle(colorImage, center, 1, cvScalar(0, 255, 255), 2, 8, 0);
		cvLine(colorImage, center, way, cvScalar(0, 0, 255), 1, 8, 0);
		//cvCircle(colorImage, center, radius*0.1f, cvScalar(0, 0, 255), 1, 8, 0);
	}

	cvNamedWindow("SURF");
	cvShowImage("SURF", colorImage);
	cvWaitKey(0);

	//후처리 - 메모리 해제 등
	cvReleaseImage(&grayImage);
	cvReleaseImage(&colorImage);
	cvClearSeq(imageKeypoint);
	cvClearSeq(imageDescriptors);
	cvReleaseMemStorage(&storage);
	cvDestroyAllWindows();

	return 0;
}

*/


#include <iostream>
#include <math.h>
#include <string>
#include "cv.h"
#include "ml.h"
#include "highgui.h"

using namespace cv;
using namespace std;

bool plotSupportVectors = true;
int numTrainingPoints = 2000;
int numTestPoints = 2000;
int size = 300;
int eq = 0;

// accuracy
float evaluate(Mat& predicted, Mat& actual) {
	assert(predicted.rows == actual.rows);
	int t = 0;
	int f = 0;
	for (int i = 0; i < actual.rows; i++) {
		float p = predicted.at<float>(i, 0);
		float a = actual.at<float>(i, 0);
		if ((p >= 0.0 && a >= 0.0) || (p <= 0.0 &&  a <= 0.0)) {
			t++;
		}
		else {
			f++;
		}
	}
	return (t * 1.0) / (t + f);
}

// plot data and class
void PrintWindow(Mat& pointData, Mat& classes, string windowName) {
	Mat canvas(size, size, CV_8UC3);
	canvas.setTo(Scalar(255.0, 255.0, 255.0));
	for (int i = 0; i < pointData.rows; i++) {

		float x = pointData.at<float>(i, 0) * size;
		float y = pointData.at<float>(i, 1) * size;

		if (classes.at<float>(i, 0) > 0) {
			circle(canvas, Point(x, y), 2, CV_RGB(255, 0, 0), 1);
		}
		else {
			circle(canvas, Point(x, y), 2, CV_RGB(0, 255, 0), 1);
		}
	}
	imshow(windowName, canvas);
}

// function to learn
int labelfunction(float x, float y, int equation) {
	switch (equation) {
	case 0:
		return y > sin(x * 10) ? -1 : 1;
		break;
	case 1:
		return y > cos(x * 10) ? -1 : 1;
		break;
	case 2:
		return y > 2 * x ? -1 : 1;
		break;
	case 3:
		return y > tan(x * 10) ? -1 : 1;
		break;
	default:
		return y > cos(x * 10) ? -1 : 1;
	}
}

// label data with equation
Mat labelData(Mat points, int labelNum) {
	Mat labels(points.rows, 1, CV_32FC1);
	for (int i = 0; i < points.rows; i++) {
		float x = points.at<float>(i, 0);
		float y = points.at<float>(i, 1);
		labels.at<float>(i, 0) = labelfunction(x, y, labelNum);
	}
	return labels;
}

void svm(Mat& trainingData, Mat& trainingClasses, Mat& testData, Mat& testClasses) {
	CvSVMParams param = CvSVMParams();

	param.svm_type = CvSVM::C_SVC;
	param.kernel_type = CvSVM::RBF; //CvSVM::RBF, CvSVM::LINEAR ...
	param.degree = 0; // for poly
	param.gamma = 20; // for poly/rbf/sigmoid
	param.coef0 = 0; // for poly/sigmoid

	param.C = 7; // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
	param.nu = 0.0; // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
	param.p = 0.0; // for CV_SVM_EPS_SVR

	param.class_weights = NULL; // for CV_SVM_C_SVC
	param.term_crit.type = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS;
	param.term_crit.max_iter = 1000;
	param.term_crit.epsilon = 1e-6;

	// SVM training (use train auto for OpenCV>=2.0)
	CvSVM svm(trainingData, trainingClasses, Mat(), Mat(), param);

	Mat predicted(testClasses.rows, 1, CV_32F);

	for (int i = 0; i < testData.rows; i++) {
		Mat sample = testData.row(i);

		float x = sample.at<float>(0, 0);
		float y = sample.at<float>(0, 1);

		predicted.at<float>(i, 0) = svm.predict(sample);
	}

	cout << "Accuracy_{SVM} = " << evaluate(predicted, testClasses) << endl;
	PrintWindow(testData, predicted, "Predictions SVM");

	// plot support vectors
	if (plotSupportVectors) {
		Mat plot_sv(size, size, CV_8UC3);
		plot_sv.setTo(Scalar(255.0, 255.0, 255.0));

		int svec_count = svm.get_support_vector_count();
		for (int vecNum = 0; vecNum < svec_count; vecNum++) {
			const float* vec = svm.get_support_vector(vecNum);
			circle(plot_sv, Point(vec[0] * size, vec[1] * size), 3, CV_RGB(0, 0, 0));
		}
		imshow("Support Vectors", plot_sv);
	}
}

void mlp(Mat& trainingData, Mat& trainingClasses, Mat& testData, Mat& testClasses) {

	Mat layers = Mat(4, 1, CV_32SC1);

	layers.row(0) = Scalar(2);
	layers.row(1) = Scalar(10);
	layers.row(2) = Scalar(15);
	layers.row(3) = Scalar(1);

	CvANN_MLP mlp;
	CvANN_MLP_TrainParams params;
	CvTermCriteria criteria;
	criteria.max_iter = 100;
	criteria.epsilon = 0.00001f;
	criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
	params.train_method = CvANN_MLP_TrainParams::BACKPROP;
	params.bp_dw_scale = 0.05f;
	params.bp_moment_scale = 0.05f;
	params.term_crit = criteria;

	mlp.create(layers);

	// train
	mlp.train(trainingData, trainingClasses, Mat(), Mat(), params);

	Mat response(1, 1, CV_32FC1);
	Mat predicted(testClasses.rows, 1, CV_32F);
	for (int i = 0; i < testData.rows; i++) {
		Mat response(1, 1, CV_32FC1);
		Mat sample = testData.row(i);

		mlp.predict(sample, response);
		predicted.at<float>(i, 0) = response.at<float>(0, 0);

	}

	cout << "Accuracy_{MLP} = " << evaluate(predicted, testClasses) << endl;
	PrintWindow(testData, predicted, "Predictions Backpropagation");
}

void knn(Mat& trainingData, Mat& trainingClasses, Mat& testData, Mat& testClasses, int K) {

	CvKNearest knn(trainingData, trainingClasses, Mat(), false, K);
	Mat predicted(testClasses.rows, 1, CV_32F);
	for (int i = 0; i < testData.rows; i++) {
		const Mat sample = testData.row(i);
		predicted.at<float>(i, 0) = knn.find_nearest(sample, K);
	}

	cout << "Accuracy_{KNN} = " << evaluate(predicted, testClasses) << endl;
	PrintWindow(testData, predicted, "Predictions KNN");

}

void bayes(Mat& trainingData, Mat& trainingClasses, Mat& testData, Mat& testClasses) {

	CvNormalBayesClassifier bayes(trainingData, trainingClasses);
	Mat predicted(testClasses.rows, 1, CV_32F);
	for (int i = 0; i < testData.rows; i++) {
		const Mat sample = testData.row(i);
		predicted.at<float>(i, 0) = bayes.predict(sample);
	}

	cout << "Accuracy_{BAYES} = " << evaluate(predicted, testClasses) << endl;
	PrintWindow(testData, predicted, "Predictions Bayes");

}

void decisiontree(Mat& trainingData, Mat& trainingClasses, Mat& testData, Mat& testClasses) {

	CvDTree dtree;
	Mat var_type(3, 1, CV_8U);

	// define attributes as numerical
	var_type.at<unsigned int>(0, 0) = CV_VAR_NUMERICAL;
	var_type.at<unsigned int>(0, 1) = CV_VAR_NUMERICAL;
	// define output node as numerical
	var_type.at<unsigned int>(0, 2) = CV_VAR_NUMERICAL;

	dtree.train(trainingData, CV_ROW_SAMPLE, trainingClasses, Mat(), Mat(), var_type, Mat(), CvDTreeParams());
	Mat predicted(testClasses.rows, 1, CV_32F);
	for (int i = 0; i < testData.rows; i++) {
		const Mat sample = testData.row(i);
		CvDTreeNode* prediction = dtree.predict(sample);
		predicted.at<float>(i, 0) = prediction->value;
	}

	cout << "Accuracy_{TREE} = " << evaluate(predicted, testClasses) << endl;
	PrintWindow(testData, predicted, "Predictions tree");

}


int main() {

	Mat trainingData(numTrainingPoints, 2, CV_32FC1);
	Mat testData(numTestPoints, 2, CV_32FC1);

	randu(trainingData, 0, 1);
	randu(testData, 0, 1);

	Mat trainingClasses = labelData(trainingData, eq);
	Mat testClasses = labelData(testData, eq);

	PrintWindow(trainingData, trainingClasses, "Training Data");
	PrintWindow(testData, testClasses, "Test Data");

	svm(trainingData, trainingClasses, testData, testClasses);
	mlp(trainingData, trainingClasses, testData, testClasses);
	knn(trainingData, trainingClasses, testData, testClasses, 3);
	bayes(trainingData, trainingClasses, testData, testClasses);
	//decisiontree(trainingData, trainingClasses, testData, testClasses);

	waitKey(0);

	return 0;
}


