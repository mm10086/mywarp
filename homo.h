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
������ȥ͸�ӣ�����б�ӽǱ任��Ϊ�����ӽǣ����㷨 x��=Hx ��xΪ��б�ӽǣ�x'�Ǹ����ӽǣ�H�Ǳ任����
���巽��Ϊ ����б�ӽǵ�ͼƬ�� �����½ǿ�ʼ˳ʱ��ѡ�����Ͼ��������4�����㣬���Ʋ��
�ڸ����ӽ������ĸ�����λ�ã��Ӷ����任����H��Ȼ����warpperspective������������ͼƬȥ͸�Ӻ��Ч��

*/
// �ĸ�����
typedef struct
{
	Point2d left_top;
	Point2d left_bottom;
	Point2d right_top;
	Point2d right_bottom;
}four_corners_homo;

//���ε�������
typedef struct {
	double mEdge[4];
} mRec;

class HomoMat {

public:
	//s��        �Ը߶ȵ����ţ������б��Խ������Ӧ��Խ��Ĭ��Ϊ1
	//bHaveCali���ж��Ƿ��о��α궨��,�б궨�������·�����һ��
	//wh��       �궨��ĸ߶�����֮��
	HomoMat(Mat &src, double s, bool bHaveCali, double wh) :
		mSrc(src), ms_h(s),bHaveCaliBoard(bHaveCali), mCaliH_W(wh){}
	HomoMat(Mat &src): mSrc(src), ms_h(1), bHaveCaliBoard(false), mCaliH_W(1) {}
	HomoMat(): ms_h(1), bHaveCaliBoard(false), mCaliH_W(1) {}
	~HomoMat() {}
	
	
	// ����1��ȥ͸�Ӿ���hδ֪���ֶ��궨�ĸ��㣬��ȡȥ͸�Ӿ����Լ�ȥ͸��֮���ͼ��
	Mat getHomoAndWarp(vector<Point2d> vpoint, bool bCut = true) {
		// ��ȡѡȡ���ĸ��궨��
		for (int i = 0; i < vpoint.size(); i++) {
			mImagePoints.push_back(vpoint[i]);
		}
		if (mImagePoints.size() < 4) {
			cout << "Tips������ѡȡ4����" << endl;
		}
		//���ͶӰͼ��궨���εĻ�׼��
		mRec rec;
		int nearestP = 0;
		calEdgeAndAngle(rec, nearestP);

		// ���㷴͸�Ӻ�þ��ε��ĸ�������
		vector<Point2d> points_new;
		points_new.resize(4);
		correctedRec(rec, nearestP, points_new);
		// ���㵥Ӧ����
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
	//����2����֪ȥ͸�Ӿ���h�����������uchar*��ʽͼ�񣬳��Ϳ���ȡȥ͸��֮���ͼ��
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
	// ����3����֪ȥ͸�Ӿ���h������h����ԭͼmSrcת��Ϊȥ͸��֮���ͼ��
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
	* ���� : �� cv::Mat ����д�뵽 .txt �ļ�
	*----------------------------
	* ���� : WriteData
	* ���� : public
	* ���� : -1�����ļ�ʧ�ܣ�0��д�����ݳɹ���1������Ϊ��
	*
	* ���� : fileName [in] �ļ���
	* ���� : matData [in] ��������
	*/
	int WriteData(string fileName, cv::Mat& matData)
	{
		int retVal = 0;
		// ���ļ� 
		ofstream outFile(fileName.c_str(), ios_base::out); //���½��򸲸Ƿ�ʽд�� 
		if (!outFile.is_open())
		{
			cout << "���ļ�ʧ��" << endl;
			retVal = -1;
			return (retVal);
		}
		// �������Ƿ�Ϊ�� 
		if (matData.empty())
		{
			cout << "����Ϊ��" << endl;
			retVal = 1;
			return (retVal);
		}
		// д������ 
		for (int r = 0; r < matData.rows; r++)
		{
			for (int c = 0; c < matData.cols; c++)
			{
				double data = matData.at<double>(r, c); //��ȡ���ݣ�at<type> - type �Ǿ���Ԫ�صľ������ݸ�ʽ 
				outFile << data << "\t"; //ÿ�������� tab ���� 
			}
			outFile << endl; //���� 
		}
		return (retVal);
	}

	/*----------------------------
	* ���� : �� .txt �ļ��ж������ݣ����浽 cv::Mat ����
	* - Ĭ�ϰ� float ��ʽ�������ݣ�
	* - ���û��ָ��������С��к�ͨ������������ľ����ǵ�ͨ����N �� 1 �е�
	*----------------------------
	* ���� : LoadData
	* ���� : public
	* ���� : -1�����ļ�ʧ�ܣ�0�����趨�ľ��������ȡ���ݳɹ���1����Ĭ�ϵľ��������ȡ����
	*
	* ���� : fileName [in] �ļ���
	* ���� : matData [out] ��������
	* ���� : matRows [in] ����������Ĭ��Ϊ 0
	* ���� : matCols [in] ����������Ĭ��Ϊ 0
	* ���� : matChns [in] ����ͨ������Ĭ��Ϊ 0
	*/
	int LoadData(string fileName, cv::Mat& matData, int matRows = 3, int matCols = 3, int matChns = 1)
	{
		int retVal = 0;
		// ���ļ� 
		ifstream inFile(fileName.c_str(), ios_base::in);
		if (!inFile.is_open())
		{
			cout << "��ȡ�ļ�ʧ��" << endl;
			retVal = -1;
			return (retVal);
		}
		//�������� 
		istream_iterator<double> begin(inFile); //�� float ��ʽȡ�ļ�����������ʼָ�� 
		istream_iterator<double> end; //ȡ�ļ�������ֹλ�� 
		vector<double> inData(begin, end); //���ļ����ݱ����� std::vector �� 
		vector<double> mData(matRows * matCols * matChns);
		for (int i = 0; i < matRows * matCols * matChns; i++) {
			mData.at(i) = inData.at(i);
		}
		int dstW = inData.at(matRows * matCols * matChns);
		int dstH = inData.at(matRows * matCols * matChns+1);
		cout << "dst-W:" << dstW << " dst-H:" << dstH << endl;
		cv::Mat tmpMat = cv::Mat(mData); //�������� std::vector ת��Ϊ cv::Mat 
										  // ����������д��� 
										  //copy(vec.begin(),vec.end(),ostream_iterator<double>(cout,"\t")); 
										  // ����趨�ľ���ߴ��ͨ���� 
		size_t dataLength = mData.size();
		//1.ͨ���� 
		if (matChns == 0)
		{
			matChns = 1;
		}
		//2.������ 
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
		//3.�����ܳ��� 
		if (dataLength != (matRows * matCols * matChns))
		{
			cout << "��������ݳ��� ������ �趨�ľ���ߴ���ͨ����Ҫ�󣬽���Ĭ�Ϸ�ʽ�������" << endl;
			retVal = 1;
			matChns = 1;
			matRows = dataLength;
		}

		// ���ļ����ݱ������������ 
		matData = tmpMat.reshape(matChns, matRows).clone();
		return (retVal);
	}

private:
	//���ͶӰͼ��궨���εĻ�׼��
	int calEdgeAndAngle(mRec &rec, int &nearestP) {
		double edge[4];
		double maxEdge = 0;
		int indexEdge;
		int indexPoint;
		Point2d p[4];
		// ���õ�
		for (int i = 0; i < 4; ++i) {
			p[i] = mImagePoints[i];
		}
		// �������ڵ�֮��߳�
		for (int i = 0; i < 4; ++i) {
			edge[i] = sqrtf(powf((p[i].x - p[(i + 1) % 4].x), 2) + powf((p[i].y - p[(i + 1) % 4].y), 2));
			rec.mEdge[i] = edge[i];
		}
		//Ѱ�������������ĵ�
		//����ĵ�Ӧ�������£�������������֮һ
		if (edge[0] >= edge[2]) { 
			nearestP = 0;
		}
		else {
			nearestP = 3;
		}
		return 0;
	}

	// ���㷴͸�Ӻ�þ��ε��ĸ�������
	// ���û�б궨��
	// ����б궨�壬�궨�����Ϊ����
	int correctedRec(mRec &rec, int &nearestP, vector<Point2d> &points_new) {
		points_new[nearestP] = mImagePoints[nearestP];
		double edge1 = rec.mEdge[nearestP];
		double edge2 = rec.mEdge[(nearestP + 3) % 4];
		//û�б궨�壬����nearestP���ڱ߳��Ʋ⣬ͬʱ��Ը߶Ƚ�������s
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
		//�б궨�壬���ݱ궨��ĳ�����Ʋ�
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
	// ����ͼƬsrc���㾭��ͶӰ H �任 ���λ�ã������ڳ�Ա����mCorners��
	void CalcCorners(const Mat& H, const Mat& src)
	{
		double v2[] = { 0, 0, 1 };//���Ͻ�
		double v1[3];//�任�������ֵ
		Mat V2 = Mat(3, 1, CV_64FC1, v2);  //������
		Mat V1 = Mat(3, 1, CV_64FC1, v1);  //������

		V1 = H * V2;
		//���Ͻ�(0,0,1)
		cout << "V2: " << V2 << endl;
		cout << "V1: " << V1 << endl;
		mCorners.left_top.x = v1[0] / v1[2];
		mCorners.left_top.y = v1[1] / v1[2];

		//���½�(0,src.rows,1)
		v2[0] = 0;
		v2[1] = src.rows;
		v2[2] = 1;
		V2 = Mat(3, 1, CV_64FC1, v2);  //������
		V1 = Mat(3, 1, CV_64FC1, v1);  //������
		V1 = H * V2;
		mCorners.left_bottom.x = v1[0] / v1[2];
		mCorners.left_bottom.y = v1[1] / v1[2];

		//���Ͻ�(src.cols,0,1)
		v2[0] = src.cols;
		v2[1] = 0;
		v2[2] = 1;
		V2 = Mat(3, 1, CV_64FC1, v2);  //������
		V1 = Mat(3, 1, CV_64FC1, v1);  //������
		V1 = H * V2;
		mCorners.right_top.x = v1[0] / v1[2];
		mCorners.right_top.y = v1[1] / v1[2];

		//���½�(src.cols,src.rows,1)
		v2[0] = src.cols;
		v2[1] = src.rows;
		v2[2] = 1;
		V2 = Mat(3, 1, CV_64FC1, v2);  //������
		V1 = Mat(3, 1, CV_64FC1, v1);  //������
		V1 = H * V2;
		mCorners.right_bottom.x = v1[0] / v1[2];
		mCorners.right_bottom.y = v1[1] / v1[2];

	}

	// �Զ����ͶӰ�任
	void mywarpPerspective(Mat src, Mat &dst, Mat &T) {
		//�˴�ע�����ģ�͵�����ϵ��Mat�Ĳ�ͬ
		//ͼ�������ϵ�Ϊ��0,0��������Ϊx�ᣬ����Ϊy�ᣬ����ǰ���������������� ��ĸ�ʽ�ǣ�ͼ��x��ͼ��y��---��rows��cols��
		//��Mat�����������Ϊx�ᣬ����Ϊy�ᣬ���Դ�ķ���Ϊ��ͼ��y��ͼ��x��----��cols��rows��----��width��height��
		//����Ǽ����ʱ������Ū���
		//����ԭͼ���ĸ������3*4���󣨴˴��ҵ�˳��Ϊ���ϣ����ϣ����£����£�
		
		Mat tmp(3, 4, CV_64FC1, 1);
		tmp.at < double >(0, 0) = 0;
		tmp.at < double >(1, 0) = 0;
		tmp.at < double >(0, 1) = src.cols;
		tmp.at < double >(1, 1) = 0;
		tmp.at < double >(0, 2) = 0;
		tmp.at < double >(1, 2) = src.rows;
		tmp.at < double >(0, 3) = src.cols;
		tmp.at < double >(1, 3) = src.rows;

		//���ԭͼ�ĸ�����任������꣬����任���ͼ��ߴ�
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

		//������ǰӳ����� map_x, map_y
		//size(height,width)
		dst.create(int(maxh - minh), int(maxw - minw), src.type());
		Mat map_x(dst.size(), CV_32FC1);
		Mat map_y(dst.size(), CV_32FC1);

		Mat proj(3, 1, CV_32FC1, 1);

		//Mat point(3, 1, CV_32FC1, 1);

		T.convertTo(T, CV_32FC1);
		//������Ϊ����T��pointͬ���ͣ�ͬ���Ͳſ�����ˣ����򱨴�Ҳ����ʹ��T.convertTo(T, point.type() );��
		Mat Tinv = T.inv();

		float M[9];
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; j++) {
				M[i * 3 + j] = Tinv.at<float>(i, j);
			}
		}
		 //clock_t  start, end;
		
		// ��ͼƬ����ȥ͸��
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

	Mat mSrc, mDst; // ԭʼͼƬ��ȥ͸�Ӻ�ͼƬ��ַ
	Mat mH;

	double ms_h; //�Ը߶ȵ�����
	bool bHaveCaliBoard; //�Ƿ��б궨��
	double mCaliH_W = 1; //�궨��߶ȺͿ�ȵı�ֵ

};
#endif // !HOMO_H