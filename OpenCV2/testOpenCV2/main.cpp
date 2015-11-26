

#include "svm_match.h"
#include "svm_makeTestData.h"
#include "svm_makeLearnData.h"
#include "data_makeImageData.h"


//한 이미지를 가로세로 10개의 묶음으로 자르면 100개의 셀이 생긴다
//블록은 네개의 셀을 묶은 것이 한 블록이다.
//가로 10개의 셀, 세로 10개의 셀이므로,
//가로 9개의 블록, 세로 9개의 블록이라고 할 수 있다.
//따라서 한 이미지에는 9 * 9 = 81 개의 블록이 있다.
//한 블록에는 네개의 셀이 있다.
//각 셀은 수많은 픽셀을 갖고 있는데, 픽셀은 각 기울기를 갖고 있다.
//기울기는 180도를 20도씩 잘라서 9개의 기울기로 나뉘는데. 각 0부터 8까지로 임의로 나눴다.
//각 셀은 기울기도표 하나씩을 갖고 있는데
//기울기도표는 9개의 그래프를 갖고 있다.
//9개의 그래프는 각 기울기를 나타낸다 (180도를 20도씩 나눈 숫자 0부터 8까지)
//한 블록은 그래서 4개의 기울기도표를 갖고있다고 할 수 있다.
//각 기울기도표마다 9개의 기울기 그래프를 갖고 있으니
//한 블록은 9 * 4 = 36개의 기울기 그래프를 갖고 있는 것이다.
//36개의 수치를 벡터로 생각하여
//한 블록은 한 벡터가 있는데 이 벡터는 36차원에서 표현되는 벡터라고 생각할 수 있다.
//이 벡터를 길이가 1인 벡터로 Nomalize 시킨다.
//노멀라이즈 시킨 값 36개를 차례로 다시 그림에 저장한다.
//각 블록마다 노멀라이즈 시킨 벡터의 수치값 36개를 갖고 있다
//블록은 81개이고 수치값은 36개이므로
//전체수치는 81 * 36 = 2916 개 이다.
//그림 하나는 2916 개의 수치로 표현 할 수 있다.
//이를 학습시킬 데이터로 data.dat 에 차례로 저장한다.


int main() {

	/*	mode :
		0 : ImageProcessing Mode
		1 : Make data.dat
		2 : Make specify data -> test.dat
		3 : Lean width data.dat
		*/
	//*
	int mode = 0;

	cout << "What are you Want?" << endl;
	cout << "(0 : ImageProcessing Mode)" << endl;
	cout << "(1 : Make data.dat)" << endl;
	cout << "(2 : Lean width data.dat, return learning.dat)" << endl;
	cout << "(3 : Make specify data -> test.dat)" << endl;
	cout << "(4 : Match test.dat and learning.dat)" << endl;

	cin >> mode;

	char* lena = "lena_std.jpg";

	if (mode == 0){		//imageProcessing mode
		IplImage* edge;
		IplImage* grayImage = cvLoadImage(lena, CV_LOAD_IMAGE_GRAYSCALE);
		edge = PrintEdge_Degree(EdgeDetection_Degree(grayImage), 5);
		cvShowImage("Original", grayImage);
		cvShowImage("EdgeDetection", edge);
		cvWaitKey(0);
		cvReleaseImage(&edge);
		cvReleaseImage(&grayImage);
	}
	else if (mode == 1){
		MakeImageData();
	}
	else if (mode == 2){
		MakeLearnData();
	}
	else if (mode == 3){
		MakeTestData();
	}
	else if (mode == 4){
		MatchData();
	}

	waitKey(0);

	int inputAnyKey;
	cin >> inputAnyKey;

	return 0;
}



