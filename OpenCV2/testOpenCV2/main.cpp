

#include "svm_match.h"
#include "svm_makeTestData.h"
#include "svm_makeLearnData.h"
#include "data_makeImageData.h"


//�� �̹����� ���μ��� 10���� �������� �ڸ��� 100���� ���� �����
//����� �װ��� ���� ���� ���� �� ����̴�.
//���� 10���� ��, ���� 10���� ���̹Ƿ�,
//���� 9���� ���, ���� 9���� ����̶�� �� �� �ִ�.
//���� �� �̹������� 9 * 9 = 81 ���� ����� �ִ�.
//�� ��Ͽ��� �װ��� ���� �ִ�.
//�� ���� ������ �ȼ��� ���� �ִµ�, �ȼ��� �� ���⸦ ���� �ִ�.
//����� 180���� 20���� �߶� 9���� ����� �����µ�. �� 0���� 8������ ���Ƿ� ������.
//�� ���� ���⵵ǥ �ϳ����� ���� �ִµ�
//���⵵ǥ�� 9���� �׷����� ���� �ִ�.
//9���� �׷����� �� ���⸦ ��Ÿ���� (180���� 20���� ���� ���� 0���� 8����)
//�� ����� �׷��� 4���� ���⵵ǥ�� �����ִٰ� �� �� �ִ�.
//�� ���⵵ǥ���� 9���� ���� �׷����� ���� ������
//�� ����� 9 * 4 = 36���� ���� �׷����� ���� �ִ� ���̴�.
//36���� ��ġ�� ���ͷ� �����Ͽ�
//�� ����� �� ���Ͱ� �ִµ� �� ���ʹ� 36�������� ǥ���Ǵ� ���Ͷ�� ������ �� �ִ�.
//�� ���͸� ���̰� 1�� ���ͷ� Nomalize ��Ų��.
//��ֶ����� ��Ų �� 36���� ���ʷ� �ٽ� �׸��� �����Ѵ�.
//�� ��ϸ��� ��ֶ����� ��Ų ������ ��ġ�� 36���� ���� �ִ�
//����� 81���̰� ��ġ���� 36���̹Ƿ�
//��ü��ġ�� 81 * 36 = 2916 �� �̴�.
//�׸� �ϳ��� 2916 ���� ��ġ�� ǥ�� �� �� �ִ�.
//�̸� �н���ų �����ͷ� data.dat �� ���ʷ� �����Ѵ�.


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



