
#include <fstream>
//#include <iostream>
#include <math.h>
#include <string>
#include "cv.h"
#include "ml.h"
#include "highgui.h"
#include "SVM.h"
#include "ImageProcess.h"

#ifdef __cplusplus
extern "C" {
#endif
#include "../testOpenCV2/svm_rank/svm_light/svm_common.h"
#include "../testOpenCV2/svm_rank/svm_light/svm_learn.h"
#ifdef __cplusplus
}
#endif

using namespace cv;
using namespace std;

void read_input_parameters(long *verbosity, LEARN_PARM *learn_parm, KERNEL_PARM *kernel_parm);

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
	int mode = 2;

	char* lena = "lena_std.jpg";
	char non_vehicles[] = "car/nov/col2/image0000.png";
	char vehicles[] = "car/yes/col2/image0000.png";


	if (mode == 0){		//imageProcessing mode
		IplImage* edge;
		IplImage* grayImage = cvLoadImage(lena, CV_LOAD_IMAGE_GRAYSCALE);

		edge = PrintEdge_Degree(EdgeDetection_Degree(grayImage),5);
		cvShowImage("Original", grayImage);
		cvShowImage("EdgeDetection", edge);
		cvWaitKey(0);
		cvReleaseImage(&edge);
		cvReleaseImage(&grayImage);
	}
	else if (mode == 1){
		IplImage* grayImage;
		float* data;
		ofstream fout;
		int buff = 0;

		fout.open("data.dat");
		for (int i = 0; i < 85; i++){
			non_vehicles[21] = (i % 10 + 48);
			non_vehicles[20] = (i / 10) % 100 + 48;
			non_vehicles[19] = (i / 100) % 1000 + 48;
			grayImage = cvLoadImage(non_vehicles, CV_LOAD_IMAGE_GRAYSCALE);
			data = GetData(grayImage);
			fout << "-1 ";
			for (int i = 0; i < 2916; i++){
				fout << i + 1 << ":" << data[i] << " ";	
			}
			cout << i << endl;
			fout << "\n";
		}
		for (int i = 0; i < 85; i++){
			vehicles[21] = (i % 10 + 48);
			vehicles[20] = (i / 10) % 100 + 48;
			vehicles[19] = (i / 100) % 1000 + 48;
			grayImage = cvLoadImage(vehicles, CV_LOAD_IMAGE_GRAYSCALE);
			data = GetData(grayImage);
			fout << "1 ";
			for (int i = 0; i < 2916; i++){
				fout << i + 1 << ":" << data[i] << " ";
			}
			cout << i << endl;
			fout << "\n";
		}
		cvWaitKey(0);
		fout.close();
		cvReleaseImage(&grayImage);
	
	} else if (mode == 2){
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
	else if (mode == 3){
		DOC **docs;
		long totwords, totdoc, i;
		double *target;
		double *alpha_in = NULL;
		KERNEL_CACHE *kernel_cache;
		LEARN_PARM learn_parm;
		KERNEL_PARM kernel_parm;
		MODEL *model = (MODEL *)my_malloc(sizeof(MODEL));
		char* restartfile = "";
		char* trainningData = "data";		//docfile
		char* outputfile = "learnData";		//modelfile
		read_input_parameters(&verbosity,&learn_parm, &kernel_parm);
		read_documents(trainningData, &docs, &target, &totwords, &totdoc);

		if (restartfile[0])
			alpha_in = read_alphas(restartfile, totdoc);

		if (kernel_parm.kernel_type == LINEAR)		kernel_cache = NULL;
		else           kernel_cache = kernel_cache_init(totdoc, learn_parm.kernel_cache_size);

		svm_learn_classification(docs, target, totdoc, totwords, &learn_parm,&kernel_parm, kernel_cache, model, alpha_in);

		if (kernel_cache) {
			/* Free the memory used for the cache. */
			kernel_cache_cleanup(kernel_cache);
		}
		write_model(outputfile, model);

		free(alpha_in);
		//free_model(model, 0);
		for (i = 0; i < totdoc; i++){}
			//free_example(docs[i], 1);
		free(docs);
		free(target);
	}

	waitKey(0);
	
	return 0;



}

void read_input_parameters(long *verbosity, LEARN_PARM *learn_parm, KERNEL_PARM *kernel_parm)
{
	set_learning_defaults(learn_parm, kernel_parm);
	(*verbosity) = 1;
	if (learn_parm->svm_iter_to_shrink == -9999) {
		if (kernel_parm->kernel_type == LINEAR)
			learn_parm->svm_iter_to_shrink = 2;
		else
			learn_parm->svm_iter_to_shrink = 100;
	}
	learn_parm->type = CLASSIFICATION;
}


