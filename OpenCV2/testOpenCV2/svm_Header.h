# pragma warning (disable:4996)
#include <fstream>
#include "cv.h"
#include "ml.h"
#include "highgui.h"
#include "ImageProcess.h"

using namespace cv;
using namespace std;

#ifdef __cplusplus
extern "C" {
#endif

#include "svm_common.h"
#include "svm_learn.h"


#ifdef __cplusplus
}
#endif