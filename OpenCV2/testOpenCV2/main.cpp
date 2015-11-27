
#include <fstream>
//#include <iostream>
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
	/*	mode : 
		0 : ImageProcessing Mode
		1 : Make data.dat
		2 : Make specify data -> test.dat
	*/
	int mode = 2;

	char* lena = "lena_std.jpg";
	char non_vehicles[] = "car/nov/col2/image0000.png";
	char vehicles[] = "car/yes/col2/image0000.png";
	IplImage* edge, *pressEdge, *extendEdge;
	IplImage* grayImage = cvLoadImage(lena, CV_LOAD_IMAGE_GRAYSCALE);

	if (mode == 0){		//imageProcessing mode
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
		cvReleaseImage(&edge);
		cvReleaseImage(&pressEdge);
		cvReleaseImage(&extendEdge);
		cvReleaseImage(&grayImage);
	}
	else if (mode == 1){
		ofstream fout;
		int buff = 0;

		fout.open("data.dat");
		for (int i = 0; i < 85; i++){
			non_vehicles[21] = (i % 10 + 48);
			non_vehicles[20] = (i / 10) % 100 + 48;
			non_vehicles[19] = (i / 100) % 1000 + 48;
			grayImage = cvLoadImage(non_vehicles, CV_LOAD_IMAGE_GRAYSCALE);
			fout << "-1 ";
			for (int k = 0; k < 4; k++){
				pressEdge = PressEdges(EdgeDetection(grayImage, k), 30);
				for (int y = 0; y < 30; y++){
					for (int x = 0; x < 30; x++){
						float val = cvGet2D(pressEdge, x, y).val[0];
						val *= 0.003921;
						fout << k * 900 + x + y * 30 + 1 << ":" << val << " ";
					}
				}
			}
			cout << i << endl;
			fout << "\n";
		}
		for (int i = 0; i < 85; i++){
			non_vehicles[21] = (buff % 10 + 48);
			non_vehicles[20] = (buff / 10) % 100 + 48;
			non_vehicles[19] = (buff / 100) % 1000 + 48;
			grayImage = cvLoadImage(non_vehicles, CV_LOAD_IMAGE_GRAYSCALE);
			fout << "-1 ";
			for (int k = 0; k < 4; k++){
				pressEdge = PressEdges(EdgeDetection(grayImage, k), 30);
				for (int y = 0; y < 30; y++){
					for (int x = 0; x < 30; x++){
						float val = cvGet2D(pressEdge, x, y).val[0];
						val *= 0.003921;
						fout << k * 900 + x + y * 30 + 1 << ":" << val << " ";
					}
				}
			}
			cout << i << endl;
			fout << "\n";
		}
		cvWaitKey(0);
		fout.close();
		cvReleaseImage(&grayImage);
	
	} else if (mode == 2){
		ofstream fout;
		int buff = 0;
		fout.open("test.dat");
		grayImage = cvLoadImage("car/yes/col2/image0100.png", CV_LOAD_IMAGE_GRAYSCALE);
		for (int k = 0; k < 4; k++){
			pressEdge = PressEdges(EdgeDetection(grayImage, k), 30);
			for (int y = 0; y < 30; y++){
				for (int x = 0; x < 30; x++){
					float val = cvGet2D(pressEdge, x, y).val[0];
					val *= 0.003921;
					fout << k * 900 + x + y * 30 + 1 << ":" << val << " ";
				}
			}
		}
		fout << "\n";
		cvWaitKey(0);
		fout.close();
		cvReleaseImage(&grayImage);
	}

	/*

	Mat trainingData(getNumberTrainingPoints(), 2, CV_32FC1);
	Mat testData(getNumberTestPoints(), 2, CV_32FC1);

	randu(trainingData, 0, 1);
	randu(testData, 0, 1);

	Mat trainingClasses = labelData(trainingData, getEq());
	Mat testClasses = labelData(testData, getEq());

	PrintWindow(trainingData, trainingClasses, "Training Data");
	PrintWindow(testData, testClasses, "Test Data");

	svm(trainingData, trainingClasses, testData, testClasses);
	mlp(trainingData, trainingClasses, testData, testClasses);
	knn(trainingData, trainingClasses, testData, testClasses, 3);
	bayes(trainingData, trainingClasses, testData, testClasses);
	decisiontree(trainingData, trainingClasses, testData, testClasses);

	//*/

	waitKey(0);
	
	return 0;



}


