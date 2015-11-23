
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

//1���� Ű ����Ʈ�� ���� Ư¡���� �������� ���
fout << imageKeypoints->total << ' ' << DIM_VECTOR << endl;

//2����ʹ� Ű����Ʈ ������ Ư¡���͸� ���
for (int i = 0; i < imageKeypoints->total; i++) {
CvSURFPoint* point = (CvSURFPoint*)cvGetSeqElem(imageKeypoints, i);
float* descriptor = (float*)cvGetSeqElem(imageDescriptors, i);
//Ű ����Ʈ ���� ���( x��ǥ, y��ǥ, ������, ���ö�þ�)
fout << point->pt.x << ' ' << point->pt.y << ' ' << point->size << ' ' << point->laplacian << '\n';
//Ư¡���� �������
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
	for (int i = NTRAINING_SAMPLES; i < 2 * NTRAINING_SAMPLES; ++i)
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














	cv::initModule_nonfree();
	//const char* imageFile = argc == 3 ? argv[1] : "C:\work\beaber.jpeg";
	//const char* surfFile = argc == 3 ? argv[2] : "C:\work\beaber.surf";
	char* imageFile = "C:/work/car.jpg";
	char* surfFile = "C:/work/car.surf";


	//SURF ������ ���� �Է� ������ �׷��̽����Ϸ� ����
	IplImage* grayImage = cvLoadImage(imageFile, CV_LOAD_IMAGE_GRAYSCALE);
	if (!grayImage) {
		cerr << "cannot find image file:" << imageFile << endl;
		return -1;
	}
	CvSVM svm;
	//����� Ű ����Ʈ�� ǥ���ϱ� ���� �÷��ε� ����
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

	//�������κ��� SURF Ư¡ ����
	//cvExtractMSER(grayImage, 0, &imageKeypoint, storage, mserParams);
	cvExtractSURF(grayImage, 0, &imageKeypoint, &imageDescriptors, storage, params);
	cout << "image Descriptors:" << imageDescriptors->total << endl;

	//SURF ���� ���Ͽ� ���
	writeSURF(surfFile, imageKeypoint, imageDescriptors);

	//���� Ű����Ʈ ǥ��
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

	//��ó�� - �޸� ���� ��
	cvReleaseImage(&grayImage);
	cvReleaseImage(&colorImage);
	cvClearSeq(imageKeypoint);
	cvClearSeq(imageDescriptors);
	cvReleaseMemStorage(&storage);
	cvDestroyAllWindows();

	return 0;
}


//*/
