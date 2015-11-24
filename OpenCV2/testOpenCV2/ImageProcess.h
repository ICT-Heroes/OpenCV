#pragma once

#include <iostream>
#include <math.h>
#include <string>
#include "cv.h"
#include "ml.h"
#include "highgui.h"

using namespace cv;
using namespace std;

IplImage* EdgeDetection(IplImage* grayImage, int mask);

IplImage* PressEdges(IplImage* edge, int size);

IplImage* ExpendImage(IplImage* img, int scale);

IplImage* ExpendImage(IplImage* src, int scale, int way);

void ExpendImage(IplImage* dst, IplImage* src, int scale, int way);