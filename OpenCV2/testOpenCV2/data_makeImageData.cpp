

#include "data_makeImageData.h"

void MakeImageData(){

	char non_vehicles[] = "car/nov/col2/image0000.png";
	char vehicles[] = "car/yes/col2/image0000.png";
	IplImage* grayImage;
	float* data;
	ofstream fout;
	int buff = 0;

	fout.open("data.dat");
	for (int i = 0; i < 975; i++){
		non_vehicles[21] = (i % 10 + 48);
		non_vehicles[20] = (i / 10) % 10 + 48;
		non_vehicles[19] = (i / 100) % 10 + 48;
		grayImage = cvLoadImage(non_vehicles, CV_LOAD_IMAGE_GRAYSCALE);
		data = GetData(grayImage);
		fout << "-1 ";
		for (int i = 0; i < 2916; i++){
			fout << i + 1 << ":" << data[i] << " ";
		}
		cout << non_vehicles << "   " << i << endl;
		fout << "\n";
		IplImage* trash = grayImage;
		cvReleaseImage(&trash);
	}
	for (int i = 0; i < 500; i++){
		vehicles[21] = (i % 10 + 48);
		vehicles[20] = (i / 10) % 10 + 48;
		vehicles[19] = (i / 100) % 10 + 48;
		grayImage = cvLoadImage(vehicles, CV_LOAD_IMAGE_GRAYSCALE);
		data = GetData(grayImage);
		fout << "1 ";
		for (int i = 0; i < 2916; i++){
			fout << i + 1 << ":" << data[i] << " ";
		}
		cout << vehicles << "   " << i << endl;
		fout << "\n";
		IplImage* trash = grayImage;
		cvReleaseImage(&trash);
	}
	cvWaitKey(0);
	fout.close();
	//cvReleaseImage(&grayImage);
}