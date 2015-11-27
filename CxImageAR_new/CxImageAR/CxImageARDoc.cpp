
/* CxImageARDoc.cpp : implement a Augmented Reality */
#include "stdafx.h"
#include "CxImageARDoc.h"



#ifdef _DEBUG
#define new DEBUG_NEW
#endif


CCxImageARDoc::CCxImageARDoc(){
	m_pImage = NULL;
}

int CCxImageARDoc::FindType(const CString& ext)
{
	return CxImage::GetTypeIdFromName(ext);
}

CxImage* CCxImageARDoc::initImage(int width, int height){
	CxImage* temp = new CxImage;
	temp->Create(width, height, 24, CXIMAGE_FORMAT_BMP);

	for(int y = 0; y < height; y++){
		for(int x = 0; x < width; x++){
			RGBQUAD color;
			color.rgbRed = 0;
			color.rgbGreen = 0;
			color.rgbBlue = 0;
			temp->SetPixelColor(x, y, color);
		}
	}
	return temp;
}



/* File Open */
BOOL CCxImageARDoc::OnOpenImage(LPCTSTR lpszPathName, int idx){
	if(idx == 0){
		m_pImage = new CxImage;
		m_pImage->Load(lpszPathName, FindType(lpszPathName));
	}
	else{
		m_pRender = new CxImage;
		m_pRender->Load(lpszPathName, FindType(lpszPathName));
	}
	return TRUE;
}



CxImage* CCxImageARDoc::Grayscale(CxImage* m_pImage){

	int width = m_pImage->GetWidth();
	int height = m_pImage->GetHeight();

	CxImage* dst = new CxImage;
	dst->Create(width, height, 24, CXIMAGE_FORMAT_BMP);

	RGBQUAD color;
	RGBQUAD newcolor;

	for(int y = 0; y < height; y++){
		for(int x = 0; x < width; x++){
			color = m_pImage->GetPixelColor(x, y);
			
			newcolor.rgbBlue = (color.rgbBlue + color.rgbGreen + color.rgbRed)/3;
			newcolor.rgbGreen = (color.rgbBlue + color.rgbGreen + color.rgbRed)/3;
			newcolor.rgbRed = (color.rgbBlue + color.rgbGreen + color.rgbRed)/3;

			dst->SetPixelColor(x, y, newcolor);
		}
	}

	return dst;
}
