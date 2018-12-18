#include "homo.h"
#include <ctime>
using namespace cv;
using namespace std;

/*
������ȥ͸�ӣ�����б�ӽǱ任��Ϊ�����ӽǣ����㷨
*/
vector<Point2d> imagePoints;
// ������¼�
void onMouse(int event, int x, int y, int flags, void *ustc) {
	if (event == CV_EVENT_LBUTTONDOWN && imagePoints.size() < 4)
	{
		Mat* mat = (Mat*)ustc;
		imagePoints.push_back(Point2d(x, y));
		cv::putText(*mat, to_string(imagePoints.size()), Point2f(x, y), 1, 3.0, Scalar(255, 0, 0));
		cv::imshow("origin", *mat);
		cout << "������������¼�" << imagePoints.size()<<endl;
		return;
	}
	if (event == CV_EVENT_LBUTTONDOWN && imagePoints.size() >= 4)
	{
		imagePoints.clear();
		Mat* mat = (Mat*)ustc;
		imagePoints.push_back(Point2d(x, y));
		cv::putText(*mat, to_string(imagePoints.size()), Point2f(x, y), 1, 3.0, Scalar(255, 0, 0));
		cv::imshow("origin", *mat);
		cout << "������������¼�" << endl;
		return;
	}
}

int main(int argc, char *argv[]) {
	clock_t  start, end;
	
	Mat image1 = imread("13.jpg");
	Mat image1_h;
	namedWindow("origin", WINDOW_AUTOSIZE);
	imshow("origin", image1);
	cout << "������½ǿ�ʼ˳ʱ��ѡȡ�����ĸ�����" << endl;
	setMouseCallback("origin", onMouse, &(image1));
	waitKey();
	try	{
		///*============����1��ȥ͸�Ӿ���hδ֪���ֶ��궨�ĸ��㣬��ȡȥ͸�Ӿ����Լ�ȥ͸��֮���ͼ��===========*/
		//HomoMat homo(image1); //��ʼ����Ҫȥ͸�ӵ�ͼ��Ĭ�ϣ������ţ��ޱ궨��

		//start = clock();

		//image1_h = homo.getHomoAndWarp(imagePoints, true); //���ݱ궨���ĸ������homo���󣬲��õ�ƴ��ͼ��

		//end = clock();
		//cout << "time cost: " << double(end - start) / CLK_TCK << " s" << endl;
		//Mat H = homo.getH();
		//cout << H << endl;
		//imwrite("dst.jpg", image1_h);
		//namedWindow("dst", WINDOW_AUTOSIZE);
		//imshow("dst", image1_h);
		//waitKey();


		///*============����2����֪ȥ͸�Ӿ���h�����������uchar*��ʽͼ�񣬳��Ϳ���ȡȥ͸��֮���ͼ��===========*/
		//HomoMat homo(image1); //��ʼ����Ҫȥ͸�ӵ�ͼ��Ĭ�ϣ������ţ��ޱ궨��

		//Mat h_ = (Mat_<double>(3, 3) << 1.1164401, 0.54456758, -94.615791,
		//	-0.0077634314, 1.3175242, -28.867954,
		//	-3.546946e-05, 0.0008676349, 1);

		//start = clock();

		//uchar* charImg = homo.matToChar(image1);             
		//homo.getWarpedImg(charImg, 1280, 720, h_);            //��֪homo���󣬽�uchar*��ʽͼƬȥ͸��
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


		/*============����3����֪ȥ͸�Ӿ���h������h����ԭͼmSrcת��Ϊȥ͸��֮���ͼ��===========*/
		HomoMat homo(image1); //��ʼ����Ҫȥ͸�ӵ�ͼ��Ĭ�ϣ������ţ��ޱ궨��

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
		cerr << "Tips������ѡȡ4����" << endl;
		waitKey();
	}
	return 0;
}
