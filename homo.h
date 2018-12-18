#ifndef HOMO_H
#define HOMO_H


#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/opencv.hpp"
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <iostream>
#include "mywarp.h"
using namespace cv;
using namespace std;

/*
这里是去透视（将倾斜视角变换成为俯视视角）的算法 x’=Hx ，x为倾斜视角，x'是俯视视角，H是变换矩阵
具体方法为 在倾斜视角的图片中 从左下角开始顺时针选地面上矩形物体的4个顶点，并推测出
在俯视视角下这四个顶点位置，从而求解变换矩阵H，然后用warpperspective函数计算整张图片去透视后的效果

*/
// 四个顶点
typedef struct
{
	Point2d left_top;
	Point2d left_bottom;
	Point2d right_top;
	Point2d right_bottom;
}four_corners_homo;

//矩形的四条边
typedef struct {
	double mEdge[4];
} mRec;

class HomoMat {

public:
	//s：        对高度的缩放，相机倾斜角越大，缩放应该越大，默认为1
	//bHaveCali：判断是否有矩形标定板,有标定板的情况下方法不一样
	//wh：       标定板的高度与宽度之比
	HomoMat(Mat &src, double s, bool bHaveCali, double wh) :
		mSrc(src), ms_h(s),bHaveCaliBoard(bHaveCali), mCaliH_W(wh){}
	HomoMat(Mat &src): mSrc(src), ms_h(1), bHaveCaliBoard(false), mCaliH_W(1) {}
	HomoMat(): ms_h(1), bHaveCaliBoard(false), mCaliH_W(1) {}
	~HomoMat() {}
	
	
	// 方法1：去透视矩阵h未知，手动标定四个点，获取去透视矩阵，以及去透视之后的图像
	Mat getHomoAndWarp(vector<Point2d> vpoint, bool bCut = true) {
		// 获取选取的四个标定点
		for (int i = 0; i < vpoint.size(); i++) {
			mImagePoints.push_back(vpoint[i]);
		}
		if (mImagePoints.size() < 4) {
			cout << "Tips：至少选取4个点" << endl;
		}
		//求解投影图像标定矩形的基准点
		mRec rec;
		int nearestP = 0;
		calEdgeAndAngle(rec, nearestP);

		// 计算反透视后该矩形的四个点坐标
		vector<Point2d> points_new;
		points_new.resize(4);
		correctedRec(rec, nearestP, points_new);
		// 计算单应矩阵
		Mat H = findHomography(mImagePoints, points_new);
		CalcCorners(H, mSrc);
		//float scale_H = H.at<double>(2, 2);
		//H = H / scale_H;
		
		Mat warp1;
		double width = MAX(mCorners.right_top.x - mCorners.left_top.x, mCorners.right_bottom.x - mCorners.left_bottom.x);
		double height = MAX(-mCorners.right_top.y + mCorners.right_bottom.y, -mCorners.left_top.y + mCorners.left_bottom.y);
		
		if (bCut == true) {
			warpPerspective(mSrc, warp1, H, Size(width, height));
		}
		else {
			mywarpPerspective(mSrc, warp1, H);
		}
		cout << "dst-W:" << mDst.cols << " dst-H:" << mDst.rows << endl;
		mH = H;
		mDst = warp1;
		return mDst;
	}
	//方法2：已知去透视矩阵h，根据输入的uchar*格式图像，长和宽，获取去透视之后的图像
	void getWarpedImg(uchar *pSrc, int nW, int nH, Mat& h, bool bCut = true) {
		Mat mImage(nH, nW, CV_8UC3, pSrc);
		mSrc = mImage;

		CalcCorners(h, mSrc);
		//float scale_H = H.at<double>(2, 2);
		//H = H / scale_H;
		Mat warp1;
		double width = MAX(mCorners.right_top.x - mCorners.left_top.x, mCorners.right_bottom.x - mCorners.left_bottom.x);
		double height = MAX(-mCorners.right_top.y + mCorners.right_bottom.y, -mCorners.left_top.y + mCorners.left_bottom.y);

		if (bCut == true) {
			warpPerspective(mSrc, warp1, h, Size(width, height));
		}
		else {
			mywarpPerspective(mSrc, warp1, h);
		}
		mH = h;
		mDst = warp1;
		int dstW = mDst.cols;
		int dstH = mDst.rows;
		for (int i = 0; i < dstW * dstH * 3; i++) {
			pSrc[i] = mDst.data[i];
		}
	}
	// 方法3：已知去透视矩阵h，根据h，将原图mSrc转变为去透视之后的图像
	Mat getWarpedImgInit(Mat& h, bool bCut = true) {
		CalcCorners(h, mSrc);
		//float scale_H = H.at<double>(2, 2);
		//H = H / scale_H;
		Mat warp1;
		double width = MAX(mCorners.right_top.x - mCorners.left_top.x, mCorners.right_bottom.x - mCorners.left_bottom.x);
		double height = MAX(-mCorners.right_top.y + mCorners.right_bottom.y, -mCorners.left_top.y + mCorners.left_bottom.y);
		//double width = MAX(mCorners.right_top.x, mCorners.right_bottom.x);
		//double height = MAX(mCorners.right_bottom.y, mCorners.left_bottom.y);

		if (bCut == true) {
			warpPerspective(mSrc, warp1, h, Size(width, height));
		}
		else {
			mywarpPerspective(mSrc, warp1, h);
			//warp1 = newwarpPerspective(mSrc, warp1, h);
		}

		mH = h;
		mDst = warp1;
		return mDst;
	}
	uchar* matToChar(Mat& img) {
		return img.data;
	}
	Mat charToMat(uchar* img, int nW, int nH) {
		Mat mImage(nH, nW, CV_8UC3, img);
		return mImage;
	}
    

	Mat getH() {
		return mH;
	}
	Mat getImage() {
		return mDst;
	}

	/*----------------------------
	* 功能 : 将 cv::Mat 数据写入到 .txt 文件
	*----------------------------
	* 函数 : WriteData
	* 访问 : public
	* 返回 : -1：打开文件失败；0：写入数据成功；1：矩阵为空
	*
	* 参数 : fileName [in] 文件名
	* 参数 : matData [in] 矩阵数据
	*/
	int WriteData(string fileName, cv::Mat& matData)
	{
		int retVal = 0;
		// 打开文件 
		ofstream outFile(fileName.c_str(), ios_base::out); //按新建或覆盖方式写入 
		if (!outFile.is_open())
		{
			cout << "打开文件失败" << endl;
			retVal = -1;
			return (retVal);
		}
		// 检查矩阵是否为空 
		if (matData.empty())
		{
			cout << "矩阵为空" << endl;
			retVal = 1;
			return (retVal);
		}
		// 写入数据 
		for (int r = 0; r < matData.rows; r++)
		{
			for (int c = 0; c < matData.cols; c++)
			{
				double data = matData.at<double>(r, c); //读取数据，at<type> - type 是矩阵元素的具体数据格式 
				outFile << data << "\t"; //每列数据用 tab 隔开 
			}
			outFile << endl; //换行 
		}
		return (retVal);
	}

	/*----------------------------
	* 功能 : 从 .txt 文件中读入数据，保存到 cv::Mat 矩阵
	* - 默认按 float 格式读入数据，
	* - 如果没有指定矩阵的行、列和通道数，则输出的矩阵是单通道、N 行 1 列的
	*----------------------------
	* 函数 : LoadData
	* 访问 : public
	* 返回 : -1：打开文件失败；0：按设定的矩阵参数读取数据成功；1：按默认的矩阵参数读取数据
	*
	* 参数 : fileName [in] 文件名
	* 参数 : matData [out] 矩阵数据
	* 参数 : matRows [in] 矩阵行数，默认为 0
	* 参数 : matCols [in] 矩阵列数，默认为 0
	* 参数 : matChns [in] 矩阵通道数，默认为 0
	*/
	int LoadData(string fileName, cv::Mat& matData, int matRows = 3, int matCols = 3, int matChns = 1)
	{
		int retVal = 0;
		// 打开文件 
		ifstream inFile(fileName.c_str(), ios_base::in);
		if (!inFile.is_open())
		{
			cout << "读取文件失败" << endl;
			retVal = -1;
			return (retVal);
		}
		//载入数据 
		istream_iterator<double> begin(inFile); //按 float 格式取文件数据流的起始指针 
		istream_iterator<double> end; //取文件流的终止位置 
		vector<double> inData(begin, end); //将文件数据保存至 std::vector 中 
		vector<double> mData(matRows * matCols * matChns);
		for (int i = 0; i < matRows * matCols * matChns; i++) {
			mData.at(i) = inData.at(i);
		}
		int dstW = inData.at(matRows * matCols * matChns);
		int dstH = inData.at(matRows * matCols * matChns+1);
		cout << "dst-W:" << dstW << " dst-H:" << dstH << endl;
		cv::Mat tmpMat = cv::Mat(mData); //将数据由 std::vector 转换为 cv::Mat 
										  // 输出到命令行窗口 
										  //copy(vec.begin(),vec.end(),ostream_iterator<double>(cout,"\t")); 
										  // 检查设定的矩阵尺寸和通道数 
		size_t dataLength = mData.size();
		//1.通道数 
		if (matChns == 0)
		{
			matChns = 1;
		}
		//2.行列数 
		if (matRows != 0 && matCols == 0)
		{
			matCols = dataLength / matChns / matRows;
		}
		else if (matCols != 0 && matRows == 0)
		{
			matRows = dataLength / matChns / matCols;
		}
		else if (matCols == 0 && matRows == 0)
		{
			matRows = dataLength / matChns;
			matCols = 1;
		}
		//3.数据总长度 
		if (dataLength != (matRows * matCols * matChns))
		{
			cout << "读入的数据长度 不满足 设定的矩阵尺寸与通道数要求，将按默认方式输出矩阵！" << endl;
			retVal = 1;
			matChns = 1;
			matRows = dataLength;
		}

		// 将文件数据保存至输出矩阵 
		matData = tmpMat.reshape(matChns, matRows).clone();
		return (retVal);
	}

private:
	//求解投影图像标定矩形的基准点
	int calEdgeAndAngle(mRec &rec, int &nearestP) {
		double edge[4];
		double maxEdge = 0;
		int indexEdge;
		int indexPoint;
		Point2d p[4];
		// 设置点
		for (int i = 0; i < 4; ++i) {
			p[i] = mImagePoints[i];
		}
		// 计算相邻点之间边长
		for (int i = 0; i < 4; ++i) {
			edge[i] = sqrtf(powf((p[i].x - p[(i + 1) % 4].x), 2) + powf((p[i].y - p[(i + 1) % 4].y), 2));
			rec.mEdge[i] = edge[i];
		}
		//寻找离摄像机最近的点
		//最近的点应该是左下，右下两者其中之一
		if (edge[0] >= edge[2]) { 
			nearestP = 0;
		}
		else {
			nearestP = 3;
		}
		return 0;
	}

	// 计算反透视后该矩形的四个点坐标
	// 如果没有标定板
	// 如果有标定板，标定板必须为矩形
	int correctedRec(mRec &rec, int &nearestP, vector<Point2d> &points_new) {
		points_new[nearestP] = mImagePoints[nearestP];
		double edge1 = rec.mEdge[nearestP];
		double edge2 = rec.mEdge[(nearestP + 3) % 4];
		//没有标定板，根据nearestP相邻边长推测，同时会对高度进行缩放s
		if (bHaveCaliBoard == false) {
			if (nearestP == 0) {
				edge1 = ms_h * edge1;
				points_new[1].x = points_new[nearestP].x;
				points_new[1].y = points_new[nearestP].y - edge1;
				points_new[2].x = points_new[nearestP].x + edge2; 
				points_new[2].y = points_new[nearestP].y - edge1;
				points_new[3].x = points_new[nearestP].x + edge2;
				points_new[3].y = points_new[nearestP].y;
			}
			else {
				edge2 = ms_h * edge2;
				points_new[0].x = points_new[nearestP].x - edge1;
				points_new[0].y = points_new[nearestP].y;
				points_new[1].x = points_new[nearestP].x - edge1;
				points_new[1].y = points_new[nearestP].y - edge2;
				points_new[2].x = points_new[nearestP].x;
				points_new[2].y = points_new[nearestP].y - edge2;
			}
		}
		//有标定板，根据标定板的长宽比推测
		if (bHaveCaliBoard == true) {
			if (nearestP == 0) {
				edge1 = edge2 * mCaliH_W;
				points_new[1].x = points_new[nearestP].x;
				points_new[1].y = points_new[nearestP].y - edge1;
				points_new[2].x = points_new[nearestP].x + edge2;
				points_new[2].y = points_new[nearestP].y - edge1;
				points_new[3].x = points_new[nearestP].x + edge2;
				points_new[3].y = points_new[nearestP].y;
			}
			else {
				edge2 = edge1 * mCaliH_W;
				points_new[0].x = points_new[nearestP].x - edge1;
				points_new[0].y = points_new[nearestP].y;
				points_new[1].x = points_new[nearestP].x - edge1;
				points_new[1].y = points_new[nearestP].y - edge2;
				points_new[2].x = points_new[nearestP].x;
				points_new[2].y = points_new[nearestP].y - edge2;
			}
		}
		return 0;
	}
	// 计算图片src顶点经过投影 H 变换 后的位置，保存在成员变量mCorners中
	void CalcCorners(const Mat& H, const Mat& src)
	{
		double v2[] = { 0, 0, 1 };//左上角
		double v1[3];//变换后的坐标值
		Mat V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
		Mat V1 = Mat(3, 1, CV_64FC1, v1);  //列向量

		V1 = H * V2;
		//左上角(0,0,1)
		cout << "V2: " << V2 << endl;
		cout << "V1: " << V1 << endl;
		mCorners.left_top.x = v1[0] / v1[2];
		mCorners.left_top.y = v1[1] / v1[2];

		//左下角(0,src.rows,1)
		v2[0] = 0;
		v2[1] = src.rows;
		v2[2] = 1;
		V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
		V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
		V1 = H * V2;
		mCorners.left_bottom.x = v1[0] / v1[2];
		mCorners.left_bottom.y = v1[1] / v1[2];

		//右上角(src.cols,0,1)
		v2[0] = src.cols;
		v2[1] = 0;
		v2[2] = 1;
		V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
		V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
		V1 = H * V2;
		mCorners.right_top.x = v1[0] / v1[2];
		mCorners.right_top.y = v1[1] / v1[2];

		//右下角(src.cols,src.rows,1)
		v2[0] = src.cols;
		v2[1] = src.rows;
		v2[2] = 1;
		V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
		V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
		V1 = H * V2;
		mCorners.right_bottom.x = v1[0] / v1[2];
		mCorners.right_bottom.y = v1[1] / v1[2];

	}

	// 自定义的投影变换
	void mywarpPerspective(Mat src, Mat &dst, Mat &T) {
		//此处注意计算模型的坐标系与Mat的不同
		//图像以左上点为（0,0），向左为x轴，向下为y轴，所以前期搜索到的特征点 存的格式是（图像x，图像y）---（rows，cols）
		//而Mat矩阵的是向下为x轴，向左为y轴，所以存的方向为（图像y，图像x）----（cols，rows）----（width，height）
		//这个是计算的时候容易弄混的
		//创建原图的四个顶点的3*4矩阵（此处我的顺序为左上，右上，左下，右下）
		
		Mat tmp(3, 4, CV_64FC1, 1);
		tmp.at < double >(0, 0) = 0;
		tmp.at < double >(1, 0) = 0;
		tmp.at < double >(0, 1) = src.cols;
		tmp.at < double >(1, 1) = 0;
		tmp.at < double >(0, 2) = 0;
		tmp.at < double >(1, 2) = src.rows;
		tmp.at < double >(0, 3) = src.cols;
		tmp.at < double >(1, 3) = src.rows;

		//获得原图四个顶点变换后的坐标，计算变换后的图像尺寸
		Mat corner = T * tmp;      //corner = (x, y) = (cols, rows)
		int width = 0, height = 0;
		double maxw = corner.at < double >(0, 0) / corner.at < double >(2, 0);
		double minw = corner.at < double >(0, 0) / corner.at < double >(2, 0);
		double maxh = corner.at < double >(1, 0) / corner.at < double >(2, 0);

		double minh = corner.at < double >(1, 0) / corner.at < double >(2, 0);
		for (int i = 1; i < 4; i++) {
			maxw = max(maxw, corner.at < double >(0, i) / corner.at < double >(2, i));
			minw = min(minw, corner.at < double >(0, i) / corner.at < double >(2, i));
			maxh = max(maxh, corner.at < double >(1, i) / corner.at < double >(2, i));
			minh = min(minh, corner.at < double >(1, i) / corner.at < double >(2, i));
		}

		//创建向前映射矩阵 map_x, map_y
		//size(height,width)
		dst.create(int(maxh - minh), int(maxw - minw), src.type());
		Mat map_x(dst.size(), CV_32FC1);
		Mat map_y(dst.size(), CV_32FC1);

		Mat proj(3, 1, CV_32FC1, 1);

		//Mat point(3, 1, CV_32FC1, 1);

		T.convertTo(T, CV_32FC1);
		//本句是为了令T与point同类型（同类型才可以相乘，否则报错，也可以使用T.convertTo(T, point.type() );）
		Mat Tinv = T.inv();

		float M[9];
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; j++) {
				M[i * 3 + j] = Tinv.at<float>(i, j);
			}
		}
		 //clock_t  start, end;
		
		// 对图片进行去透视
		 int interpolation = CV_INTER_LINEAR;
		 Range range(0, dst.rows);
		 const int BLOCK_SZ = 32;
		 short XY[BLOCK_SZ * BLOCK_SZ * 2], A[BLOCK_SZ * BLOCK_SZ];
		 int x, y, x1, y1, width_ = dst.cols, height_ = dst.rows;
		 int bh0 = std::min(BLOCK_SZ / 2, height_);
		 int bw0 = std::min(BLOCK_SZ * BLOCK_SZ / bh0, width_);
		 bh0 = std::min(BLOCK_SZ * BLOCK_SZ / bw0, height_);
		 //start = clock();
		 for(y = range.start; y < range.end; y += bh0) {
			 for (x = 0; x < width_; x += bw0) {
				 int bw = std::min(bw0, width_ - x);
				 int bh = std::min(bh0, range.end - y); // height
				 Mat _XY(bh, bw, CV_16SC2, XY), matA;
				 Mat dpart(dst, Rect(x, y, bw, bh));

				 for (y1 = 0; y1 < bh; y1++) {
					 short* xy = XY + y1 * bw * 2;
					 double X0 = M[0] * (x + minw) + M[1] * (y + y1 + minh) + M[2];
					 double Y0 = M[3] * (x + minw) + M[4] * (y + y1 + minh) + M[5];
					 double W0 = M[6] * (x + minw) + M[7] * (y + y1 + minh) + M[8];

					short* alpha = A + y1 * bw;
					x1 = 0;
					for (; x1 < bw; x1++) {
						double W = W0 + M[6] * x1;
						W = W ? INTER_TAB_SIZE / W : 0;
						double fX = std::max((double)INT_MIN, std::min((double)INT_MAX, (X0 + M[0] * x1)*W));
						double fY = std::max((double)INT_MIN, std::min((double)INT_MAX, (Y0 + M[3] * x1)*W));
						int X = saturate_cast<int>(fX);
						int Y = saturate_cast<int>(fY);
						xy[x1 * 2] = saturate_cast<short>(X >> INTER_BITS);
						xy[x1 * 2 + 1] = saturate_cast<short>(Y >> INTER_BITS);
						alpha[x1] = (short)((Y & (INTER_TAB_SIZE - 1)) * INTER_TAB_SIZE + (X & (INTER_TAB_SIZE - 1)));
					}
				 }
				 Mat _matA(bh, bw, CV_16U, A);
				 remap(src, dpart, _XY, _matA, interpolation);
			 }
		 }
		//end = clock();
		//cout << "time cost: " << double(end - start) / CLK_TCK << " s" << endl;
	}

	
	
	four_corners_homo mCorners;

	vector<Point2d> mImagePoints;

	Mat mSrc, mDst; // 原始图片和去透视后图片地址
	Mat mH;

	double ms_h; //对高度的拉伸
	bool bHaveCaliBoard; //是否有标定板
	double mCaliH_W = 1; //标定板高度和宽度的比值

};
#endif // !HOMO_H