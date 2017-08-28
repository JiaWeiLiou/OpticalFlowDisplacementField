// OpticalFlowDisplacementField.cpp : 定義主控台應用程式的進入點。
//

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <time.h>
#include <fstream>
#include <cmath>

using namespace std;
using namespace cv;

#define UNKNOWN_FLOW_THRESH 1e9

void bgraMergeToBGR(Mat &Image, Mat colorwheelImage, int x, int y);

void makecolorwheel(vector<Scalar> &colorwheel);

void colorwheel(Mat &colorwheelImage, int dim);

void drawMunsellColorSystem(Mat flow, Mat &color, double maxrad);

void drawSamplingVectorField(const Mat& flow, Mat& cflowmap, int step, double mfactor,const Scalar& color);

//void drawSamplingVectorField(const Mat& oldFlow, Mat& flow, Mat& cflowmap, int step, const Scalar& color);

int main()
{
	std::cout << "Please enter video path : ";
	string infile;
	cin >> infile;

	/*確認檔案是否存在*/
	VideoCapture video(infile); // open the default camera
	if (!video.isOpened())  // check if we succeeded
	{
		std::cout << "Error opening Video !" << endl;
		system("pause");
		return -1;
	}

	std::cout << "Whether to use the default output ((0) N (1) Y ) : ";
	bool default;
	cin >> default;

	int form = 1;		
	int diameter = 100;			//設定顯示的孟賽爾色環直徑大小(像素)
	int radius = 30;			//設定孟賽爾色環半徑(像素)
	double sampling = 20;		//設定採樣向量場間隔
	double mfactor = 5;			//設定長度放大倍數

	if (!default)
	{
		std::cout << "Please enter output form of displacement field ((1) Munsell Color System & Sampling Vector Field (2) Only Munsell Color System (3) Only Sampling Vector Field ) : ";
		cin >> form;
		if (form > 3 || form < 1)
		{
			std::cout << "Error Input !" << endl;
			std::cout << "Default output form of displacement field with Munsell Color System & Sampling Vector Field.";
			form = 1;
		}

		/*設定顯示的孟賽爾色環直徑大小*/
		if (form == 1 || form == 2)
		{
			std::cout << "Please enter output of Munsell Color System diameter (ex:100) : ";
			cin >> diameter;
			if (diameter < 0)
			{
				std::cout << "Error Input !" << endl;
				std::cout << "Default output of Munsell Color System diameter = 100";
				diameter = 100;
			}
		}

		/*設定孟賽爾色環半徑*/
		if (form == 1 || form == 2)
		{
			std::cout << "Please enter Munsell Color System radius (ex:30) : ";
			cin >> radius;
			if (radius < 0)
			{
				std::cout << "Error Input !" << endl;
				std::cout << "Default Munsell Color System radius = 30";
				radius = 30;
			}
		}

		/*設定採樣向量場間隔及長度放大倍數*/
		if (form == 1 || form == 3)
		{
			std::cout << "Please enter Sampling Vector Field interval (ex:20) : ";
			cin >> sampling;
			if (sampling < 0 || sampling > 150)
			{
				std::cout << "Error Input !" << endl;
				std::cout << "Default Sampling Vector Field interval = 20";
				radius = 20;
			}

			std::cout << "Please enter the magnification factor of the vector length (ex:5) : ";
			cin >> mfactor;
			if (mfactor < 0 || mfactor > 20)
			{
				std::cout << "Error Input !" << endl;
				std::cout << "Default Sampling Vector Field interval = 5";
				mfactor = 5;
			}
		}
	}

	/*計算程式執行時間*/
	double timeStart, timeEnd;
	timeStart = clock();

	/*設定輸出文件名*/
	string outfile;
	int pos1 = infile.find_last_of('/\\');
	int pos2 = infile.find_last_of('.');
	string filepath(infile.substr(0, pos1));
	string infile_name(infile.substr(pos1 + 1, pos2 - pos1 - 1));
	outfile = filepath + "\\" + infile_name + "_DisplacementField.avi";

	/*獲取第一幀影像*/
	Mat newFrame;
	video >> newFrame;

	/*將彩色影像轉換為灰階並設定為第一幀*/
	Mat newGray, prevGray;
	cv::cvtColor(newFrame, newGray, CV_BGR2GRAY);
	prevGray = newGray.clone();

	/*設定Farneback光流法參數*/
	double pyr_scale = 0.5;
	int levels = 3;
	int winsize = 9;
	int iterations = 7;
	int poly_n = 5;
	double poly_sigma = 1.1;
	int flags = OPTFLOW_USE_INITIAL_FLOW;
	double fps = video.get(CV_CAP_PROP_FPS);   //獲取影片幀率

	/*輸出影片基本資訊*/
	std::cout << "Video's frame width  : " << video.get(CV_CAP_PROP_FRAME_WIDTH)  << " pixel"  << endl;
	std::cout << "Video's frame height : " << video.get(CV_CAP_PROP_FRAME_HEIGHT) << " pixel"  << endl;
	std::cout << "Video's total frames : " << video.get(CV_CAP_PROP_FRAME_COUNT)  << " frames" << endl;
	std::cout << "Video's frame rate   : " << fps								  << " FPS"    << endl;

	std::cout << endl << "Start calculate ..." << endl;

	int frameNum = 1;
	cout << endl << "No. " << frameNum << "\t";

	/*創建輸出影片物件*/
	VideoWriter writer;
	if (form == 1)
		writer = VideoWriter(outfile, CV_FOURCC('D', 'I', 'V', 'X'), fps, Size(newFrame.cols * 2, newFrame.rows));
	else
		writer = VideoWriter(outfile, CV_FOURCC('D', 'I', 'V', 'X'), fps, Size(newFrame.size()));

	/*光流場*/
	Mat flow = Mat(newGray.size(), CV_32FC2);

	/*創建孟塞爾色環*/
	Mat colorwheelImg;
	colorwheel(colorwheelImg, diameter);

	while (1)
	{
		video >> newFrame;
		if (newFrame.empty()) break;
		cv::cvtColor(newFrame, newGray, CV_BGR2GRAY);

		frameNum++;
		cout << frameNum << "\t";

		/*Farneback光流法計算*/
		calcOpticalFlowFarneback(prevGray, newGray, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags);

		if (form == 1)
		{
			Mat MunsellColorSystem;
			Mat SamplingVectorField = newFrame;

			drawMunsellColorSystem(flow, MunsellColorSystem, radius);
			drawSamplingVectorField(flow, SamplingVectorField, sampling, mfactor, CV_RGB(255, 0, 0));

			/*將孟塞爾色環放置於孟塞爾流場影像的左上方*/
			bgraMergeToBGR(MunsellColorSystem, colorwheelImg, 0, 0);

			Mat ImageCombine(newFrame.rows, newFrame.cols + newFrame.cols, MunsellColorSystem.type());
			Mat LeftImage = ImageCombine(Rect(0, 0, newFrame.cols, newFrame.rows));
			Mat RightImage = ImageCombine(Rect(newFrame.cols, 0, newFrame.cols, newFrame.rows));
			MunsellColorSystem.copyTo(LeftImage);
			SamplingVectorField.copyTo(RightImage);

			writer.write(ImageCombine);
		}
		else if (form == 2)
		{
			Mat MunsellColorSystem;
			drawMunsellColorSystem(flow, MunsellColorSystem, radius);

			/*將孟塞爾色環放置於孟塞爾流場影像的左上方*/
			bgraMergeToBGR(MunsellColorSystem, colorwheelImg, 0, 0);

			writer.write(MunsellColorSystem);
		}
		else
		{
			Mat SamplingVectorField = newFrame;
			drawSamplingVectorField(flow, SamplingVectorField, sampling, mfactor, CV_RGB(255, 0, 0));
			writer.write(SamplingVectorField);
		}

		prevGray = newGray.clone();
	}

	timeEnd = clock();
	std::cout << endl << "total time = " << (timeEnd - timeStart) / CLOCKS_PER_SEC << " s" << endl;

	return 0;
}

void bgraMergeToBGR(Mat &Image, Mat colorwheelImage, int x, int y)
{
	for (int i = 0; i < colorwheelImage.rows; i++)
		for (int j = 0; j < colorwheelImage.cols; j++)
			if ((int)colorwheelImage.at<Vec4b>(i + y, j + x)[3] == 255)
			{
				Image.at<Vec3b>(i + y, j + x)[0] = colorwheelImage.at<Vec4b>(i, j)[0];
				Image.at<Vec3b>(i + y, j + x)[1] = colorwheelImage.at<Vec4b>(i, j)[1];
				Image.at<Vec3b>(i + y, j + x)[2] = colorwheelImage.at<Vec4b>(i, j)[2];
			}
}

void makecolorwheel(vector<Scalar> &colorwheel)
{
	int RY = 15;	//紅色(Red)     至黃色(Yellow)
	int YG = 15;	//黃色(Yellow)  至綠色(Green)
	int GC = 15;	//綠色(Green)   至青色(Cyan)
	int CB = 15;	//青澀(Cyan)    至藍色(Blue)
	int BM = 15;	//藍色(Blue)    至洋紅(Magenta)
	int MR = 15;	//洋紅(Magenta) 至紅色(Red)

	for (int i = 0; i < RY; i++) colorwheel.push_back(Scalar(255, 255 * i / RY, 0));
	for (int i = 0; i < YG; i++) colorwheel.push_back(Scalar(255 - 255 * i / YG, 255, 0));
	for (int i = 0; i < GC; i++) colorwheel.push_back(Scalar(0, 255, 255 * i / GC));
	for (int i = 0; i < CB; i++) colorwheel.push_back(Scalar(0, 255 - 255 * i / CB, 255));
	for (int i = 0; i < BM; i++) colorwheel.push_back(Scalar(255 * i / BM, 0, 255));
	for (int i = 0; i < MR; i++) colorwheel.push_back(Scalar(255, 0, 255 - 255 * i / MR));
}

void colorwheel(Mat &colorwheelImage, int dim)
{
	if (dim % 2 == 0)
	{
		dim = dim - 1;    //限制直徑為奇數
	}

	if (colorwheelImage.empty())
		colorwheelImage.create(dim, dim, CV_8UC4);

	static vector<Scalar> colorwheel; //Scalar r,g,b  
	if (colorwheel.empty())
		makecolorwheel(colorwheel);

	for (int i = 0; i < dim - 1; ++i)
	{
		for (int j = 0; j < dim - 1; ++j)
		{
			int x = j - (dim - 1) / 2;
			int y = i - (dim - 1) / 2;
			double rad = sqrt(x * x + y * y);
			double angle = atan2(-y, -x) / CV_PI;
			double fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);  //計算角度對應之漸層色的實際索引位置
			int k0 = (int)fk;											//計算角度對應之漸層色的索引位置下界
			int k1 = (k0 + 1) % colorwheel.size();						//計算角度對應之漸層色的索引位置上界(if k0=89; k1 =0)
			float f = fk - k0;											//計算實際索引位置至索引位置下界的距離
			//f = 0; // uncomment to see original color wheel  

			/*設定半徑內不透明、半徑外透明*/
			if (rad <= (dim - 1) / 2)
				colorwheelImage.at<Vec4b>(i, j)[3] = 255;
			else
				colorwheelImage.at<Vec4b>(i, j)[3] = 0;

			for (int b = 0; b < 3; b++)
			{
				float col0 = colorwheel[k0][b] / 255.0;
				float col1 = colorwheel[k1][b] / 255.0;
				float col = (1 - f) * col0 + f * col1;					//漸層色內插
				if (rad <= (dim - 1) / 2)
				{
					col = 1 - (rad / ((dim - 1) / 2)) * (1 - col);		// increase saturation with radius
				}
				colorwheelImage.at<Vec4b>(i, j)[2 - b] = (int)(255.0 * col);
			}
		}
	}
}

void drawMunsellColorSystem(Mat flow, Mat &color, double maxrad)
{
	if (color.empty())
		color.create(flow.rows, flow.cols, CV_8UC3);

	static vector<Scalar> colorwheel; //Scalar r,g,b  
	if (colorwheel.empty())
		makecolorwheel(colorwheel);

	// determine motion range:  
	//float maxrad = -1;

	// Find max flow to normalize fx and fy  
	//for (int i = 0; i < flow.rows; ++i)
	//{
	//	for (int j = 0; j < flow.cols; ++j)
	//	{
	//		Vec2f flow_at_point = flow.at<Vec2f>(i, j);
	//		float fx = flow_at_point[0];
	//		float fy = flow_at_point[1];
	//		if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))
	//			continue;
	//		float rad = sqrt(fx * fx + fy * fy);
	//		maxrad = maxrad > rad ? maxrad : rad;
	//	}
	//}

	for (int i = 0; i < flow.rows; ++i)
	{
		for (int j = 0; j < flow.cols; ++j)
		{
			uchar *data = color.data + color.step[0] * i + color.step[1] * j;
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);

			float fx = flow_at_point[0] / maxrad;
			float fy = flow_at_point[1] / maxrad;
			if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))
			{
				data[0] = data[1] = data[2] = 0;
				continue;
			}
			float rad = sqrt(fx * fx + fy * fy);

			float angle = atan2(-fy, -fx) / CV_PI;    //單位為-1至+1
			float fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);  //計算角度對應之索引位置
			int k0 = (int)fk;
			int k1 = (k0 + 1) % colorwheel.size();
			float f = fk - k0;
			//f = 0; // uncomment to see original color wheel  

			for (int b = 0; b < 3; b++)
			{
				float col0 = colorwheel[k0][b] / 255.0;
				float col1 = colorwheel[k1][b] / 255.0;
				float col = (1 - f) * col0 + f * col1;
				if (rad <= 1)
					col = 1 - rad * (1 - col); // increase saturation with radius  
				else
					col = col;  //out of range
				data[2 - b] = (int)(255.0 * col);
			}
		}
	}
}

void drawSamplingVectorField(const Mat& flow, Mat& cflowmap, int step, double mfactor, const Scalar& color)
{
	for (int y = step; y < cflowmap.rows; y += step)
		for (int x = step; x < cflowmap.cols; x += step)
		{
			const Point2f& fxy = flow.at<Point2f>(y, x);
			line(cflowmap, Point(x, y), Point(cvRound(x + fxy.x * mfactor), cvRound(y + fxy.y * mfactor)), color);
			circle(cflowmap, Point(x, y), 1, color, -1);
		}
}

//void drawSamplingVectorField(const Mat& oldFlow, Mat& flow, Mat& cflowmap, int step, const Scalar& color)
//{
//	for (int y = step; y < cflowmap.rows; y += step)
//		for (int x = step; x < cflowmap.cols; x += step)
//		{
//			Point2f& fxy = flow.at<Point2f>(y, x);
//			const Point2f& oldFxy = oldFlow.at<Point2f>(y, x);
//			float avgFx = 0;
//			float avgFy = 0;
//			float avgOldFx = 0;
//			float avgOldFy = 0;
//			int avgWin = 0;
//
//			if ((step / 3) % 2 == 1)
//				avgWin = step / 3;
//			else
//				avgWin = (step / 3) + 1;
//
//			for (int j = -avgWin / 2; j < avgWin / 2; j++)
//				for (int i = -avgWin / 2; i < avgWin / 2; i++)
//				{
//					avgFx += flow.at<Point2f>(y + j, x + i).x / (avgWin *avgWin);
//					avgFy += flow.at<Point2f>(y + j, x + i).y / (avgWin *avgWin);
//					avgOldFx += oldFlow.at<Point2f>(y + j, x + i).x / (avgWin *avgWin);
//					avgOldFy += oldFlow.at<Point2f>(y + j, x + i).y / (avgWin *avgWin);
//				}
//
//			//鄰近區域平均方向及平均長度拘束
//			if ((fxy.x*avgFx < 0 || fxy.y*avgFy  < 0) || (abs(fxy.x / avgFx)>2) || (abs(fxy.y / avgFy)>2))
//			{
//				fxy.x = avgFx;
//				fxy.y = avgFy;
//			}
//
//			//上一幀平均方向拘束
//			if ((fxy.x*avgOldFx < 0 || fxy.y*avgOldFy < 0) || (abs(fxy.x / avgOldFx)>2) || (abs(fxy.y / avgOldFy)>2))
//			{
//				fxy.x = avgOldFx;
//				fxy.y = avgOldFy;
//			}
//
//			line(cflowmap, Point(x, y), Point(cvRound(x + fxy.x * 5), cvRound(y + fxy.y * 5)), color);
//			circle(cflowmap, Point(x, y), 2, color, -1);
//		}
//}



