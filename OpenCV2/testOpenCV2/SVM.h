#pragma once


#include <iostream>
#include <math.h>
#include <string>
#include "cv.h"
#include "ml.h"
#include "highgui.h"

using namespace cv;
using namespace std;


int getNumberTrainingPoints();
int getNumberTestPoints();
int getWindowSize();
int getEq();
bool getPlotSupportVectors();

float evaluate(Mat& predicted, Mat& actual);

void PrintWindow(Mat& pointData, Mat& classes, string windowName);

int labelfunction(float x, float y, int equation);

Mat labelData(Mat points, int labelNum);

void svm(Mat& trainingData, Mat& trainingClasses, Mat& testData, Mat& testClasses);

void mlp(Mat& trainingData, Mat& trainingClasses, Mat& testData, Mat& testClasses);

void knn(Mat& trainingData, Mat& trainingClasses, Mat& testData, Mat& testClasses, int K);

void bayes(Mat& trainingData, Mat& trainingClasses, Mat& testData, Mat& testClasses);

void decisiontree(Mat& trainingData, Mat& trainingClasses, Mat& testData, Mat& testClasses);