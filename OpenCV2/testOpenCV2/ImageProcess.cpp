
#include "ImageProcess.h"






int GetEdgeOfPixel_Mask(IplImage* img, int x, int y, int mask) {
	int sum = 0;

	int EdgeMask[4][3][3] = {
		{ { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } },
		{ { 1, 0, -1 }, { 2, 0, -2 }, { 1, 0, -1 } },
		{ { -2, -1, 0 }, { -1, 0, 1 }, { 0, 1, 2 } },
		{ { 0, 1, 2 }, { -1, 0, 1 }, { -2, -1, 0 } }
	};
	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			sum += (int)(cvGet2D(img, x + i, y + j).val[0] * EdgeMask[mask][i + 1][j + 1]);
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
			edge = GetEdgeOfPixel_Mask(grayImage, x, y, mask);
			if (edge < 0)edge *= -1;
			if (255 < edge)
				edge = 255;
			color.val[0] = color.val[1] = color.val[2] = edge;
			cvSet2D(ret, x, y, color);
		}
	}

	return ret;
}

int GetGradient_Degree(int edge){
	if (0 <= edge && edge < 20 )		return 0;
	else if ((20 <= edge && edge < 40))	return 1;
	else if ((40 <= edge && edge < 60))	return 2;
	else if ((60 <= edge && edge < 80))	return 3;
	else if ((80 <= edge && edge < 100) || (-100 <= edge && edge < -80))	return 4;
	else if ((100 <= edge && edge < 120) || (-80 <= edge && edge < -60))	return 5;
	else if ((120 <= edge && edge < 140) || (-60 <= edge && edge < -40))	return 6;
	else if ((140 <= edge && edge < 160) || (-40 <= edge && edge < -20))	return 7;
	else if ((160 <= edge && edge <= 180) || (-20 <= edge && edge < 0))	return 8;
	else	return -1;
}

int GetEdgeOfPixel_Degree(IplImage* img, int x, int y) {
	float subX = 0, subY = 0;

	subX = (float)(cvGet2D(img, x + 1, y).val[0] - cvGet2D(img, x - 1, y).val[0]);
	subY = (float)(cvGet2D(img, x, y + 1).val[0] - cvGet2D(img, x, y - 1).val[0]);

	//if (subX < 0)	subX *= -1;
	//if (subY < 0)	subY *= -1;
	if (subY == 0)	subY += 0.00001f;
	float ang = atan(subX / subY) * 180.0f / 3.141592f;
	return (int)ang;
}

IplImage* EdgeDetection_Degree(IplImage* grayImage){
	int width = grayImage->width;
	int height = grayImage->height;
	IplImage* ret = cvCreateImage(cvSize(width-2, height-2), IPL_DEPTH_8U, 1);

	CvScalar color;
	int edge;
	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			edge = GetEdgeOfPixel_Degree(grayImage, x, y);
			color.val[0] = color.val[1] = color.val[2] = GetGradient_Degree(edge);
			cvSet2D(ret, x-1, y-1, color);
		}
	}
	return ret;
}

int DrawImage_Degree(int x, int y, int size, int gradient, float bright){
	float val = 0;
	int temperature = 400;
	float thickness = 0.5;

	if (gradient == 0)		//수평
		val = (distanceLineToPoint(1 / 10, size, x, y) / ((float)size * thickness)) * temperature;
	if (gradient == 1)
		val = (distanceLineToPoint(1 / 3, size, x, y) / ((float)size * thickness)) * temperature;
	if (gradient == 2)
		val = (distanceLineToPoint(1, size, x, y) / ((float)size * thickness)) * temperature;
	if (gradient == 3)
		val = (distanceLineToPoint(3, size, x, y) / ((float)size * thickness)) * temperature;
	if (gradient == 4)		//수평
		val = (distanceLineToPoint(10000, size, x, y) / ((float)size * thickness)) * temperature;
	if (gradient == 5)
		val = (distanceLineToPoint(-3, size, x, y) / ((float)size * thickness)) * temperature;
	if (gradient == 6)
		val = (distanceLineToPoint(-1, size, x, y) / ((float)size * thickness)) * temperature;
	if (gradient == 7)
		val = (distanceLineToPoint(-1 / 3, size, x, y) / ((float)size * thickness)) * temperature;
	if (gradient == 8)
		val = (distanceLineToPoint(-1 / 10, size, x, y) / ((float)size * thickness)) * temperature;
	if (255 < val)	val = 255;
	val *= -1;
	val += 255;
	return (int)(val * bright);
}

float distanceLineToPoint(float gradient, int size, int x, int y){
	float val = y - (gradient * x) - (size / 2) + (gradient * size / 2);
	if (val < 0)	val *= -1;
	return (val) / (sqrt(1 + gradient*gradient));
}

IplImage* PrintEdge_Degree_Original(IplImage* grayImage){
	IplImage* ret = cvCreateImage(cvSize(grayImage->width, grayImage->height), IPL_DEPTH_8U, 1);
	CvScalar color;
	int val = 0;
	for (int stepY = 0; stepY < grayImage->height; stepY++) {
		for (int stepX = 0; stepX < grayImage->width; stepX++) {
			val = (int)cvGet2D(grayImage, stepX, stepY).val[0];
			color.val[0] = color.val[1] = color.val[2] = val * 30;
			cvSet2D(ret, stepX, stepY, color);
		}
	}
	return ret;
}

IplImage* PrintEdge_Degree(IplImage* edgeImage, int size){
	int width, height, widthStep, heightStep;
	width = edgeImage->width;
	height = edgeImage->height;
	IplImage* ret = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
	CvScalar color;
	widthStep = width / size;
	heightStep = height / size;
	int his[9];
	int max;
	int maxIndex;
	for (int stepY = 0; stepY < heightStep; stepY++) {
		for (int stepX = 0; stepX < widthStep; stepX++) {
			for (int h = 0; h < 9; h++)	his[h] = 0;
			max = maxIndex = 0;
			for (int y = 0; y < size; y++) {
				for (int x = 0; x < size; x++) {
					int h = (int)cvGet2D(edgeImage, stepX*size + x, stepY*size + y).val[0];
					if (0<=h && h < 9)		his[h]++;
				}
			}
			for (int h = 0; h < 9; h++){
				if (max < his[h]){
					max = his[h];
					maxIndex = h;
				}
			}
			for (int y = 0; y < size; y++) {
				for (int x = 0; x < size; x++) {
					color.val[0] = DrawImage_Degree(size - x - 1, y, size, maxIndex, ((float)max)/((float)(size*size)));
					cvSet2D(ret, stepX*size + x, stepY*size + y, color);
				}
			}
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
	float dashPower = 1.4f;		//this is root 2
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

/*
그레이스케일된 이미지가 들어왔을 때
이를 저장할 데이터로 바꾸어준다.
한줄의 배열을 리턴한다.
*/
float* GetData(IplImage* grayImage){

	int width = grayImage->width;
	int height = grayImage->height;

	int oneCellWidth = width / 10;
	int oneCellHeight = height / 10;

	//엣지 생성
	IplImage* edgeImage = EdgeDetection_Degree(grayImage);

	//셀데이터 생성 ([x][y][edgeDegree]) 및 초기화
	int cell[10][10][9];
	for (int x = 0; x < 10; x++)
		for (int y = 0; y < 10; y++)
			for (int edge = 0; edge < 9; edge++)
				cell[x][y][edge] = 0;	//초기화

	//블록데이터 생성 ([x][y][cell][edgeDegree]) 및 초기화
	float block[9][9][4][9];
	for (int x = 0; x < 9; x++)
		for (int y = 0; y < 9; y++)
			for (int c = 0; c < 4; c++)
				for (int edge = 0; edge < 9; edge++)
					block[x][y][c][edge] = 0;	//초기화

	//셀에 값 채워넣기
	int edge = 0;
	float cellSum = 0;
	for (int xCell = 0; xCell < 10; xCell++){
		for (int yCell = 0; yCell < 10; yCell++){
			//값 채우기
			for (int x = 0; x < oneCellWidth; x++){
				for (int y = 0; y < oneCellHeight; y++){
					edge = (int)cvGet2D(edgeImage, xCell*oneCellWidth + x, yCell*oneCellHeight + y).val[0];
					cell[xCell][yCell][edge]++;
				}
			}
			//정규화 (셀 크기에 영향을 받지 않기위해 일정하게 만든다. 기본적으로 한 셀당 100x100 으로 만들어버린다.)
			for (int e = 0; e < 9; e++){
				cellSum = (float)cell[xCell][yCell][e];
				cellSum *= 10000.0f / (((float)(oneCellWidth))*((float)(oneCellHeight)));
				cell[xCell][yCell][e] = (int)(cellSum + 0.5f);	//반올림
			}
		}
	}

	//블록 데이터 cell 데이터로 채워넣기
	int cellX, cellY;
	for (int x = 0; x < 9; x++)
		for (int y = 0; y < 9; y++)
			for (int c = 0; c < 4; c++)
				for (int edge = 0; edge < 9; edge++){
					cellX = cellY = 0;
					if (c == 1)	cellX = 1;
					if (c == 2)	cellY = 1;
					if (c == 3) cellX = cellY = 1;
					block[x][y][c][edge] = (float)cell[x + cellX][y + cellY][edge];
				}

	//블록 노멀라이즈 하기
	long normSum = 0;
	double norm = 0;
	for (int x = 0; x < 9; x++){
		for (int y = 0; y < 9; y++){
			normSum = 0;	//sum 초기화
			for (int c = 0; c < 4; c++)
				for (int edge = 0; edge < 9; edge++)
					normSum += (long)(block[x][y][c][edge] * block[x][y][c][edge]);	//각각의 제곱을 더함
			norm = sqrt(normSum);		//다 더했으면 제곱근을 구한다. 
			for (int c = 0; c < 4; c++)
				for (int edge = 0; edge < 9; edge++)
					block[x][y][c][edge] /= (float)norm;		//각각에다가 제곱근을 다시 나눔으로써 노멀라이즈를 마친다.
		}
	}

	//한줄로 나타내기
	float* ret = (float*)malloc(sizeof(float)*2916);
	for (int x = 0; x < 9; x++)
		for (int y = 0; y < 9; y++)
			for (int c = 0; c < 4; c++)
				for (int e = 0; e < 9; e++)
					ret[x * 324 + y * 36 + c * 9 + e] = block[x][y][c][e];

	return ret;
}

