

#include <iostream>
#include <math.h>
#include <string>
#include "cv.h"
#include "ml.h"
#include "highgui.h"

#include "SVM.h"
#include "ImageProcess.h"

using namespace cv;
using namespace std;



int main() {
	
	char* imageFile = "C:/work/lena.jpg";
	Mat edge;
	Mat grayImage = imread(imageFile, CV_LOAD_IMAGE_GRAYSCALE);
	Mat colorImage = imread(imageFile, CV_LOAD_IMAGE_COLOR);
	
	edge = EdgeDetection(grayImage, 2);
	imshow("Original_Grayfdsafdsa", edge);
	imshow("Original_Gray", grayImage);
	imshow("Original_Color", colorImage);

	Mat trainingData(getNumberTrainingPoints(), 2, CV_32FC1);
	Mat testData(getNumberTestPoints(), 2, CV_32FC1);

	randu(trainingData, 0, 1);
	randu(testData, 0, 1);

	Mat trainingClasses = labelData(trainingData, getEq());
	Mat testClasses = labelData(testData, getEq());

	//PrintWindow(trainingData, trainingClasses, "Training Data");
	//PrintWindow(testData, testClasses, "Test Data");

	//svm(trainingData, trainingClasses, testData, testClasses);
	//mlp(trainingData, trainingClasses, testData, testClasses);
	//knn(trainingData, trainingClasses, testData, testClasses, 3);
	//bayes(trainingData, trainingClasses, testData, testClasses);
	//decisiontree(trainingData, trainingClasses, testData, testClasses);

	waitKey(0);
	
	return 0;
}


