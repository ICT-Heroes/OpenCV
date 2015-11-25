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
	�� edge ��� ���Ⱑ ���� ��
	�̸� 0 ���� 9 ������ ������ ������ �ٲ۴�
	0 <= ���ϰ� < 0
*/
int GetGradient_Degree(int edge);

/*
	�� �̹������� � x�� y�� ��ġ�� ���� ���⸦ �����Ѵ�.
	-90 < ���ϰ� < 90
*/
int GetEdgeOfPixel_Degree(IplImage* img, int x, int y);

/*
	size ������ �ڸ� �� �̹��� ũ�� �ȿ��� x�� y��ġ�� �ִ� ���� ��ĥ�ϴµ�,
	gradient ���� �� ��, bright ũ�⸸ŭ�� ���� ĥ�Ѵ�.
	0 <= gradient < 9
	0 <= bright <= 1
*/
int DrawImage_Degree(int x, int y, int size, int gradient, float bright);

/*
	���� �� ������ �Ÿ��� �����Ѵ�.
*/
float distanceLineToPoint(float gradient, int size, int x, int y);

/*
	�׷��Ͻ������� �̹����� �޾Ƽ� �������ؼ��� ������ �� �� �̹����� ��ȯ�Ѵ�.
*/
IplImage* EdgeDetection_Degree(IplImage* grayImage);

/*
	�������ؼ��� �� �̹����� �ȼ� �ϳ��ϳ��� ����Ʈ �غ� �� �ִ�.
*/
IplImage* PrintEdge_Degree_Original(IplImage* edgeImage);

/*
	�������ؼ��� �� �̹����� �ȼ��� size ������ �߶� ����Ʈ �غ� �� �ִ�.
*/
IplImage* PrintEdge_Degree(IplImage* edgeImage, int size);

/*
	�׷��̽����ϵ� �̹����� ������ ��
	�̸� ������ �����ͷ� �ٲپ��ش�.
	9���� �̹����� �����Ѵ�.
	���� 3ä���� ���� ����
	�� ������ ������ 0 ~ 99�����̸�,
	var[0]�� ���� 99�� �Ѿ�� ��,
	var[1]++ �� �ϸ�, var[0] �� ���� 0�� �ȴ�.
	���� �� �̹����� �ȼ��� ��Ÿ�� �� �ִ� �ִ� ��ǥ�� ���� 100 00 00 ���̴�.
	���� �� ���� ���μ��ΰ� 1000 �� ���� �ʴ´ٸ� ǥ�� �����ϴ�.
*/
float* GetData(IplImage* grayImage);