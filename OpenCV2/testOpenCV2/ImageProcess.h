#pragma once

#include <iostream>
#include <math.h>
#include <string>
#include "cv.h"
#include "ml.h"
#include "highgui.h"

using namespace cv;
using namespace std;

Mat EdgeDetection(Mat& grayImage, int mask);