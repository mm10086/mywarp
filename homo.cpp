#include "homo.h"
#include <ctime>
using namespace cv;
using namespace std;

/*
这里是去透视（将倾斜视角变换成为俯视视角）的算法
*/
vector<Point2d> imagePoints;
// 鼠标点击事件
void onMouse(int event, int x, int y, int flags, void *ustc) {
	if (event == CV_EVENT_LBUTTONDOWN && imagePoints.size() < 4)
	{
		Mat* mat = (Mat*)ustc;
		imagePoints.push_back(Point2d(x, y));
		cv::putText(*mat, to_string(imagePoints.size()), Point2f(x, y), 1, 3.0, Scalar(255, 0, 0));
		cv::imshow("origin", *mat);
		cout << "触发左键按下事件" << imagePoints.size()<<endl;
		return;
	}
	if (event == CV_EVENT_LBUTTONDOWN && imagePoints.size() >= 4)
	{
		imagePoints.clear();
		Mat* mat = (Mat*)ustc;
		imagePoints.push_back(Point2d(x, y));
		cv::putText(*mat, to_string(imagePoints.size()), Point2f(x, y), 1, 3.0, Scalar(255, 0, 0));
		cv::imshow("origin", *mat);
		cout << "触发左键按下事件" << endl;
		return;
	}
}

int main(int argc, char *argv[]) {
	clock_t  start, end;
	
	Mat image1 = imread("13.jpg");
	Mat image1_h;
	namedWindow("origin", WINDOW_AUTOSIZE);
	imshow("origin", image1);
	cout << "请从左下角开始顺时针选取矩形四个顶点" << endl;
	setMouseCallback("origin", onMouse, &(image1));
	waitKey();
	try	{
		///*============方法1：去透视矩阵h未知，手动标定四个点，获取去透视矩阵，以及去透视之后的图像===========*/
		//HomoMat homo(image1); //初始化需要去透视的图像，默认，无缩放，无标定板

		//start = clock();

		//image1_h = homo.getHomoAndWarp(imagePoints, true); //根据标定的四个点计算homo矩阵，并得到拼接图像

		//end = clock();
		//cout << "time cost: " << double(end - start) / CLK_TCK << " s" << endl;
		//Mat H = homo.getH();
		//cout << H << endl;
		//imwrite("dst.jpg", image1_h);
		//namedWindow("dst", WINDOW_AUTOSIZE);
		//imshow("dst", image1_h);
		//waitKey();


		///*============方法2：已知去透视矩阵h，根据输入的uchar*格式图像，长和宽，获取去透视之后的图像===========*/
		//HomoMat homo(image1); //初始化需要去透视的图像，默认，无缩放，无标定板

		//Mat h_ = (Mat_<double>(3, 3) << 1.1164401, 0.54456758, -94.615791,
		//	-0.0077634314, 1.3175242, -28.867954,
		//	-3.546946e-05, 0.0008676349, 1);

		//start = clock();

		//uchar* charImg = homo.matToChar(image1);             
		//homo.getWarpedImg(charImg, 1280, 720, h_);            //已知homo矩阵，将uchar*格式图片去透视
		//image1_h = homo.charToMat(charImg, 1399, 563);

		//end = clock();
		//cout << "time cost: " << double(end - start) / CLK_TCK << " s" << endl;
		//Mat H = homo.getH();
		//cout << H << endl;
		//
		//imwrite("dst.jpg", image1_h);
		//namedWindow("dst", WINDOW_AUTOSIZE);
		//imshow("dst", image1_h);
		//waitKey();


		/*============方法3：已知去透视矩阵h，根据h，将原图mSrc转变为去透视之后的图像===========*/
		HomoMat homo(image1); //初始化需要去透视的图像，默认，无缩放，无标定板

		Mat h_ = (Mat_<double>(3, 3) << 4.909046030839832, 4.765534626767542, -2502.390201508742,
		-0.1018298622633236, 9.783280513405789, -2526.762121527132,
		-0.0001146982014730432, 0.007554135652120523, 1);

		start = clock();
		image1_h = homo.getWarpedImgInit(h_, true);

		end = clock();
		cout << "time cost: " << double(end - start) / CLK_TCK << " s" << endl;
		Mat H = homo.getH();
		cout << H << endl;

		imwrite("dst.jpg", image1_h);
		namedWindow("dst", WINDOW_AUTOSIZE);
		imshow("dst", image1_h);
		waitKey();
	}
	catch (const std::exception&) {
		cerr << "Tips：至少选取4个点" << endl;
		waitKey();
	}
	return 0;
}
