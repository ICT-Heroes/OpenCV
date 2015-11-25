
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


