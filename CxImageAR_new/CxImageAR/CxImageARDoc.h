

#pragma once

#include "stdafx.h"

class CCxImageARDoc{
private :
	CxImage* m_pImage;
	CxImage* m_pRender;

public :
	inline CxImage *GetImage() { return m_pImage; }

private:
	
	CxImage* initImage(int width, int height);

public:
	CCxImageARDoc();
	int FindType(const CString& ext);
	//file open 
	BOOL OnOpenImage(LPCTSTR lpszPathName, int idx);

	CxImage* Grayscale(CxImage* m_pImage);

};