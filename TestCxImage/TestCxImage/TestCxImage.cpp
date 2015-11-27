// TestCxImage.cpp : 콘솔 응용 프로그램에 대한 진입점을 정의합니다.
//

#include "stdafx.h"
#include "cximage/ximage.h"


int _tmain(int argc, _TCHAR* argv[])
{
	CxImage  image;
	// bmp -> jpg


	image.Load(_T("image.bmp"), CXIMAGE_FORMAT_BMP);
	if (image.IsValid()){
		if(!image.IsGrayScale()) image.IncreaseBpp(24);
		image.SetJpegQuality(80);
		image.Save(_T("image.jpg"),CXIMAGE_FORMAT_JPG);
	}
	// png -> tif


	image.Load(_T("image.png"), CXIMAGE_FORMAT_PNG);
	if (image.IsValid()){
		image.Save(_T("image.tif"),CXIMAGE_FORMAT_TIF);
	}

	return 0;
}

