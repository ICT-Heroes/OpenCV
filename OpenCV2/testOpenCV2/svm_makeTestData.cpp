
#include "svm_makeTestData.h"

void MakeTestData(){
	IplImage* grayImage;
	float* data;
	ofstream fout;
	int buff = 0;
	fout.open("test.dat");
	grayImage = cvLoadImage("car/nov/col2/image0100.png", CV_LOAD_IMAGE_GRAYSCALE);
	data = GetData(grayImage);
	for (int i = 0; i < 2916; i++){
		fout << i + 1 << ":" << data[i] << " ";
	}
	fout << "\n";
	cvWaitKey(0);
	fout.close();
	cvReleaseImage(&grayImage);
}