
// CxImageARDlg.cpp : 구현 파일
//

#include "stdafx.h"
#include "CxImageAR.h"
#include "CxImageARDlg.h"
#include "afxdialogex.h"
#include "CxImageARDoc.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CCxImageARDlg 대화 상자




CCxImageARDlg::CCxImageARDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CCxImageARDlg::IDD, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CCxImageARDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_PIC_ORIGINAL, m_pic_original);
	DDX_Control(pDX, IDC_PIC_RESULT, m_pic_result);
}

BEGIN_MESSAGE_MAP(CCxImageARDlg, CDialogEx)
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BTN_OPEN, &CCxImageARDlg::OnBnClickedBtnOpen)
	ON_WM_TIMER()
	ON_WM_CREATE()
	ON_BN_CLICKED(IDC_BTN_GRAY, &CCxImageARDlg::OnBnClickedBtnGray)
END_MESSAGE_MAP()


// CCxImageARDlg 메시지 처리기

BOOL CCxImageARDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 이 대화 상자의 아이콘을 설정합니다. 응용 프로그램의 주 창이 대화 상자가 아닐 경우에는
	//  프레임워크가 이 작업을 자동으로 수행합니다.
	SetIcon(m_hIcon, TRUE);			// 큰 아이콘을 설정합니다.
	SetIcon(m_hIcon, FALSE);		// 작은 아이콘을 설정합니다.

	// TODO: 여기에 추가 초기화 작업을 추가합니다.
	cxImageARDoc = CCxImageARDoc();
	result_option = OPTION_NO;

	return TRUE;  // 포커스를 컨트롤에 설정하지 않으면 TRUE를 반환합니다.
}

// 대화 상자에 최소화 단추를 추가할 경우 아이콘을 그리려면
//  아래 코드가 필요합니다. 문서/뷰 모델을 사용하는 MFC 응용 프로그램의 경우에는
//  프레임워크에서 이 작업을 자동으로 수행합니다.

void CCxImageARDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 그리기를 위한 디바이스 컨텍스트입니다.

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 클라이언트 사각형에서 아이콘을 가운데에 맞춥니다.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 아이콘을 그립니다.
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// 사용자가 최소화된 창을 끄는 동안에 커서가 표시되도록 시스템에서
//  이 함수를 호출합니다.
HCURSOR CCxImageARDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CCxImageARDlg::OnBnClickedBtnOpen()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	static TCHAR BASED_CODE szFilter[] = 
		_T("Supported Files(*.bmp;*.gif;*.jpg;*.jpeg;*.png;*.ico;*.tif;*.tiff;*.tga;*.pcx;*.wbmp;*.j2k;*.jp2;;*.j2c;*.jpc;*.pgx;*.pnm;*.pgm;*.ppm;*.ras;*.mng;*.jng;*.ska;*.nef;*.crw;*.cr2;*.mrw;*.raf;*.erf;*.3fr;*.dcr;*.raw;*.dng;*.pef;*.x3f;*.arw;*.sr2;*.mef;*.orf)|*.bmp;*.gif;*.jpg;*.jpeg;*.png;*.ico;*.tif;*.tiff;*.tga;*.pcx;*.wbmp;*.j2k;*.jp2;;*.j2c;*.jpc;*.pgx;*.pnm;*.pgm;*.ppm;*.ras;*.mng;*.jng;*.ska;*.nef;*.crw;*.cr2;*.mrw;*.raf;*.erf;*.3fr;*.dcr;*.raw;*.dng;*.pef;*.x3f;*.arw;*.sr2;*.mef;*.orf|")
		_T("BMP Files(*.bmp)|*.bmp|")
		_T("GIF Files(*.gif)|*.gif|")
		_T("JPG Files(*.jpg;*.jpeg)|*.jpg;*.jpeg|")
		_T("PNG Files(*.png)|*.png|")
		_T("MNG Files(*.mng;*.jng;*.png)|*.mng;*.jng;*.png|")
		_T("ICO CUR Files(*.ico;*.cur)|*.ico;*.cur|")
		_T("TIF Files(*.tif;*.tiff)|*.tif;*.tiff|")
		_T("TGA Files(*.tga)|*.tga|")
		_T("PCX Files(*.pcx)|*.pcx|")
		_T("WBMP Files(*.wbmp)|*.wbmp|")
		_T("JP2 Files(*.j2k;*.jp2)|*.j2k;*.jp2|")
		_T("JPC Files(*.j2c;*.jpc)|*.j2c;*.jpc|")
		_T("PGX Files(*.pgx)|*.pgx|")
		_T("RAS Files(*.ras)|*.ras|")
		_T("PNM Files(*.pnm;*.pgm;*.ppm)|*.pnm;*.pgm;*.ppm|")
		_T("SKA Files(*.ska)|*.ska|")
		_T("RAW Files(*.nef;*.crw;*.cr2;*.mrw;*.raf;*.erf;*.3fr;*.dcr;*.raw;*.dng;*.pef;*.x3f;*.arw;*.sr2;*.mef;*.orf)|*.nef;*.crw;*.cr2;*.mrw;*.raf;*.erf;*.3fr;*.dcr;*.raw;*.dng;*.pef;*.x3f;*.arw;*.sr2;*.mef;*.orf|")
		_T("All Files(*.*)|*.*||");

	CFileDialog fileDlg(TRUE, NULL, NULL, OFN_HIDEREADONLY, szFilter);
	
	if(fileDlg.DoModal() == IDOK){
		/* OK butten clicked */
		cxImageARDoc.OnOpenImage(fileDlg.GetPathName(), 0);
		original = cxImageARDoc.GetImage();
		result = cxImageARDoc.GetImage();
	}
}


void CCxImageARDlg::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.
	if(cxImageARDoc.GetImage() != NULL){
		
		CDC* pDC1, *pDC2;
		CRect rect1, rect2;
			
		pDC1 = m_pic_original.GetDC();
		m_pic_original.GetClientRect(&rect1);

		original->Draw(pDC1->m_hDC, rect1);

		pDC2 = m_pic_result.GetDC();
		m_pic_result.GetClientRect(&rect2);

		result->Draw(pDC2->m_hDC, rect2);


		ReleaseDC(pDC1);
		ReleaseDC(pDC2);
	}

	CDialogEx::OnTimer(nIDEvent);
}


int CCxImageARDlg::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CDialogEx::OnCreate(lpCreateStruct) == -1)
		return -1;

	// TODO:  여기에 특수화된 작성 코드를 추가합니다.
	this->SetTimer(0, 33, NULL);

	return 0;
}


void CCxImageARDlg::OnBnClickedBtnGray()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	result_option = OPTION_GRAY;
	if(cxImageARDoc.GetImage() != NULL){
		result = cxImageARDoc.Grayscale(cxImageARDoc.GetImage());
	}
}





