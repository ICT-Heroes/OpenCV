
#include "ImageProcess.h"






int GetEdgeOfPixel(IplImage* img, int x, int y, int mask) {
	int sum = 0;

	int EdgeMask[4][3][3] = {
		{ { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } },
		{ { 1, 0, -1 }, { 2, 0, -2 }, { 1, 0, -1 } },
		{ { -2, -1, 0 }, { -1, 0, 1 }, { 0, 1, 2 } },
		{ { 0, 1, 2 }, { -1, 0, 1 }, { -2, -1, 0 } }
	};
	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			sum += cvGet2D(img, x + i, y + j).val[0] * EdgeMask[mask][i + 1][j + 1];
		}
	}
	return sum;
}

IplImage* EdgeDetection(IplImage* grayImage, int mask){
	int width = grayImage->width;
	int height = grayImage->height;

	IplImage* ret = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);

	CvScalar color;
	int edge;
	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			edge = GetEdgeOfPixel(grayImage, x, y, mask);
			if (edge < 0)edge *= -1;
			if (255 < edge)
				edge = 255;
			color.val[0] = color.val[1] = color.val[2] = edge;
			cvSet2D(ret, x, y, color);
		}
	}

	return ret;
}

IplImage* PressEdges(IplImage* edge, int size){
	int width = edge->width;
	int height = edge->height;

	int cellWidth = width/size;
	int cellHeight = height/size;

	IplImage* ret = cvCreateImage(cvSize(size, size), IPL_DEPTH_8U, 3);

	CvScalar color;
	double sum;
	for (int cellx = 0; cellx < size; cellx++) {
		for (int celly = 0; celly < size; celly++) {
			sum = 0;
			for (int x = 0; x < cellWidth; x++) {
				for (int y = 0; y < cellHeight; y++) {
					sum += cvGet2D(edge, cellx*cellWidth + x, celly*cellHeight + y).val[0];
				}
			}
			//sum *= 255;
			sum /= (cellWidth) * (cellHeight);
			color.val[0] = color.val[1] = color.val[2] = sum;
			cvSet2D(ret, cellx, celly, color);
		}
	}

	return ret;
}

IplImage* ExpendImage(IplImage* img, int scale){
	int cellsize = img->height;
	int size = img->height * scale;
	IplImage* ret = cvCreateImage(cvSize(size, size), IPL_DEPTH_8U, 3);
	CvScalar color;

	for (int cellx = 0; cellx < cellsize; cellx++) {
		for (int celly = 0; celly < cellsize; celly++) {
			color = cvGet2D(img, cellx, celly);
			for (int x = 0; x < scale; x++) {
				for (int y = 0; y < scale; y++) {
					cvSet2D(ret, cellx * scale + x, celly * scale + y, color);
				}
			}
		}
	}

	return ret;
}

bool isGradient(int gradient, int width, int height, int x, int y) {
	float dashPower = 1.4;		//this is root 2
	if (gradient == 0)		//수평
		if (3 * width / 7 < x && x < 4 * width / 7)	return true;
	if (gradient == 1)		//수직
		if (3 * height / 7 < y && y < 4 * height / 7)	return true;
	if (gradient == 2)		//슬래쉬
		if (dashPower - height >-(height / width)*x - y && dashPower + height > y + (height / width)*x)	return true;
	if (gradient == 3)		//백슬래쉬
		if (dashPower >(height / width)*x - y && dashPower > y - (height / width)*x)	return true;
	return false;
}

IplImage* ExpendImage(IplImage* src, int scale, int way){
	int cellsize = src->height;
	int size = src->height * scale;
	IplImage* ret = cvCreateImage(cvSize(size, size), IPL_DEPTH_8U, 3);
	CvScalar color, darkColor;
	darkColor.val[0] = darkColor.val[1] = darkColor.val[2] = 0;

	for (int cellx = 0; cellx < cellsize; cellx++) {
		for (int celly = 0; celly < cellsize; celly++) {
			color = cvGet2D(src, cellx, celly);
			for (int x = 0; x < scale; x++) {
				for (int y = 0; y < scale; y++) {
					if (isGradient(way, scale, scale, x, y))
						cvSet2D(ret, cellx * scale + x, celly * scale + y, color);
					else
						cvSet2D(ret, cellx * scale + x, celly * scale + y, darkColor);
				}
			}
		}
	}

	return ret;
}

void ExpendImage(IplImage* dst, IplImage* src, int scale, int way){
	int cellsize = src->height;
	int size = src->height * scale;
	CvScalar color, darkColor, localColor;
	darkColor.val[0] = darkColor.val[1] = darkColor.val[2] = 0;

	for (int cellx = 0; cellx < cellsize; cellx++) {
		for (int celly = 0; celly < cellsize; celly++) {
			color = cvGet2D(src, cellx, celly);
			for (int x = 0; x < scale; x++) {
				for (int y = 0; y < scale; y++) {
					if (isGradient(way, scale, scale, x, y)){
						localColor = cvGet2D(dst, cellx * scale + x, celly * scale + y);
						localColor.val[0] = localColor.val[1] = localColor.val[2] = color.val[0] + localColor.val[0];
						cvSet2D(dst, cellx * scale + x, celly * scale + y, localColor);
					}
				}
			}
		}
	}
}

