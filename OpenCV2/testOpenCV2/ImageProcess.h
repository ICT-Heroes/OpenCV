#pragma once

#include <iostream>
#include <math.h>
#include <string>
#include "cv.h"
#include "ml.h"
#include "highgui.h"
#include <math.h>

using namespace cv;
using namespace std;

bool isGradient(int gradient, int width, int height, int x, int y);

IplImage* EdgeDetection(IplImage* grayImage, int mask);

IplImage* PressEdges(IplImage* edge, int size);

IplImage* ExpendImage(IplImage* img, int scale);

IplImage* ExpendImage(IplImage* src, int scale, int way);

int GetEdgeOfPixel_Mask(IplImage* img, int x, int y, int mask);

void ExpendImage(IplImage* dst, IplImage* src, int scale, int way);





/*
	-90 <= edge <= 90
	의 edge 라는 기울기가 들어올 때
	이를 0 에서 9 사이의 가상의 값으로 바꾼다
	0 <= 리턴값 < 0
*/
int GetGradient_Degree(int edge);

/*
	한 이미지에서 어떤 x와 y의 위치의 점의 기울기를 리턴한다.
	-90 < 리턴값 < 90
*/
int GetEdgeOfPixel_Degree(IplImage* img, int x, int y);

/*
	size 단위로 자른 한 이미지 크기 안에서 x와 y위치에 있는 점을 색칠하는데,
	gradient 기울기 일 때, bright 크기만큼의 색을 칠한다.
	0 <= gradient < 9
	0 <= bright <= 1
*/
int DrawImage_Degree(int x, int y, int size, int gradient, float bright);

/*
	점과 선 사이의 거리를 리턴한다.
*/
float distanceLineToPoint(float gradient, int size, int x, int y);

/*
	그래일스케일의 이미지를 받아서 엣지디텍션을 실행한 후 그 이미지를 반환한다.
*/
IplImage* EdgeDetection_Degree(IplImage* grayImage);

/*
	엣지디텍션을 한 이미지의 픽셀 하나하나를 프린트 해볼 수 있다.
*/
IplImage* PrintEdge_Degree_Original(IplImage* edgeImage);

/*
	엣지디텍션을 한 이미지의 픽셀을 size 단위로 잘라서 프린트 해볼 수 있다.
*/
IplImage* PrintEdge_Degree(IplImage* edgeImage, int size);

/*
	그레이스케일된 이미지가 들어왔을 때
	이를 저장할 데이터로 바꾸어준다.
	9장의 이미지를 리턴한다.
	각각 3채널의 색을 쓰며
	각 색깔의 범위는 0 ~ 99까지이며,
	var[0]의 값이 99를 넘어갔을 때,
	var[1]++ 를 하며, var[0] 의 값은 0이 된다.
	따라서 한 이미지의 픽셀이 나타낼 수 있는 최대 도표의 수는 100 00 00 개이다.
	따라서 한 셀이 가로세로가 1000 을 넘지 않는다면 표현 가능하다.
*/
float* GetData(IplImage* grayImage);