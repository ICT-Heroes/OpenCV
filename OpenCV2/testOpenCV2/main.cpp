

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
	
	char* imageFile = "C:/work/lena_std.jpg";
	IplImage* edge, *pressEdge, *extendEdge;
	IplImage* grayImage = cvLoadImage(imageFile, CV_LOAD_IMAGE_GRAYSCALE);
	//Mat colorImage = imread(imageFile, CV_LOAD_IMAGE_COLOR);
	//Mat edgeTotal = Mat(grayImage.rows, grayImage.cols , CV_8UC1);

	edge = EdgeDetection(grayImage, 3);
	pressEdge = PressEdges(EdgeDetection(grayImage, 3), 30);
	extendEdge = ExpendImage(PressEdges(EdgeDetection(grayImage, 0), 30), 17, 0);
	ExpendImage(extendEdge, PressEdges(EdgeDetection(grayImage, 1), 30), 17, 1);
	ExpendImage(extendEdge, PressEdges(EdgeDetection(grayImage, 2), 30), 17, 2);
	ExpendImage(extendEdge, PressEdges(EdgeDetection(grayImage, 3), 30), 17, 3);
	cvShowImage("EdgeDetection", edge);
	cvShowImage("PressedEdge", pressEdge);
	cvShowImage("ExtendPressedEdge", extendEdge);
	cvShowImage("Original", grayImage);
	cvWaitKey(0);

	cvDestroyWindow("EdgeDetection");
	cvDestroyWindow("PressedEdge");
	cvDestroyWindow("ExtendPressedEdge");
	cvDestroyWindow("Original");

	cvReleaseImage(&edge);
	cvReleaseImage(&pressEdge);
	cvReleaseImage(&extendEdge);
	cvReleaseImage(&grayImage);
	/*
	cvShowImage("sobelx:", sobelX);
	cvShowImage("sobely:", sobelY);

	cvWaitKey(0);

	cvDestroyWindow("Image:");
	cvDestroyWindow("Gray:");
	cvDestroyWindow("sobelx:");
	cvDestroyWindow("sobely:");

	cvReleaseImage(&img);
	cvReleaseImage(&gray);
	cvReleaseImage(&sobelX);
	cvReleaseImage(&sobelY);

	/*
	for (int x = 0; x < edgeTotal.rows; x++){
		for (int y = 0; y < edgeTotal.cols; y++){
			for (int i = 0; i < 4; i++){
				edgeTotal.at<char>(x, y) += edge[i].at<char>(x, y)/4;
			}
		}
	}
	*/

	//imshow("Original_EdgeDetection0", edge);
	//imshow("Original_Gray", grayImage);
	//imshow("Original_Color", colorImage);

	/*
	Mat trainingData(getNumberTrainingPoints(), 2, CV_32FC1);
	Mat testData(getNumberTestPoints(), 2, CV_32FC1);

	randu(trainingData, 0, 1);
	randu(testData, 0, 1);

	Mat trainingClasses = labelData(trainingData, getEq());
	Mat testClasses = labelData(testData, getEq());
	*/

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


