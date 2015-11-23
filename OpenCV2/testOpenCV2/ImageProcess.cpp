
#include "ImageProcess.h"


int EdgeMask[4][3][3] = {
	{ {1, 2, 1}, {0, 0, 0}, {-1, -2, -1} },
	{ {1, 0, -1}, {2, 0, -2}, {1, 0, -1} },
	{ {-2, -1, 0}, {-1, 0, 1}, {0, 1, 2} },
	{ {0, 1, 2}, {-1, 0, 1}, {-2, -1, 0} },
};


int GetEdgeOfPixel(Mat& img, int x, int y, int mask) {
	int color = 0;
	int sum = 0;

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			color = img.at<char>(x + i -1, y + j -1);
			sum += color * EdgeMask[mask][i][j];
		}
	}
	return sum;
}

Mat EdgeDetection(Mat& grayImage, int mask){
	int width = grayImage.cols;
	int height = grayImage.rows;

	Mat ret = Mat(width, height, CV_8UC1);

	for (int x = 1; x < width - 1; x++) {
		for (int y = 1; y < height - 1; y++) {
			int sum = GetEdgeOfPixel(grayImage, x, y, mask);
			//sum /= 4;
			if (sum < 0) sum *= -1;
			//if (sum == 0) sum = 1;
			if (255 < sum){
				sum = 255;
			}
			ret.at<char>(x, y) = sum;
		}
	}
	return ret;
}
