#include "Test.h"
#include <highgui.h>
#include <cv.h>


Test::Test()
{
}



int main(int argc, char** argv)
{
	IplImage* capture = cvLoadImage(argv[1]);
	cvNamedWindow("Example", CV_WINDOW_AUTOSIZE);
	cvShowImage("Example", capture);

	cvWaitKey(0);

	cvReleaseImage(&capture);
	cvDestroyWindow("Example");
	return 0;
}
Test::~Test()
{
}
