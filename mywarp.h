#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/opencv.hpp"
#include "opencv2/core/utility.hpp"
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <iostream>
using namespace cv;
using namespace std;

namespace cv

{
	class newWarpPerspectiveInvoker :public ParallelLoopBody
	{

	public:
		newWarpPerspectiveInvoker(Mat &_src, Mat &_dst, Mat & _T) :
			ParallelLoopBody(), src(_src), dst(_dst), T(_T)
		{
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
			maxw = corner.at < double >(0, 0) / corner.at < double >(2, 0);
			minw = corner.at < double >(0, 0) / corner.at < double >(2, 0);
			maxh = corner.at < double >(1, 0) / corner.at < double >(2, 0);

			minh = corner.at < double >(1, 0) / corner.at < double >(2, 0);
			for (int i = 1; i < 4; i++) {
				maxw = max(maxw, corner.at < double >(0, i) / corner.at < double >(2, i));
				minw = min(minw, corner.at < double >(0, i) / corner.at < double >(2, i));
				maxh = max(maxh, corner.at < double >(1, i) / corner.at < double >(2, i));
				minh = min(minh, corner.at < double >(1, i) / corner.at < double >(2, i));
			}

			dst.create(int(maxh - minh), int(maxw - minw), src.type());
			Mat map_x(dst.size(), CV_32FC1);
			Mat map_y(dst.size(), CV_32FC1);

			Mat proj(3, 1, CV_32FC1, 1);

			//Mat point(3, 1, CV_32FC1, 1);

			T.convertTo(T, CV_32FC1);
			//本句是为了令T与point同类型（同类型才可以相乘，否则报错，也可以使用T.convertTo(T, point.type() );）
		}

		virtual void operator() (const Range& range) const

		{
			//clock_t  start, end;
			//start = clock();
			Mat Tinv = T.inv();
			float M[9];
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; j++) {
					M[i * 3 + j] = Tinv.at<float>(i, j);
				}
			}

			int interpolation = CV_INTER_LINEAR;
			const int BLOCK_SZ = 32;

			short XY[BLOCK_SZ*BLOCK_SZ * 2], A[BLOCK_SZ*BLOCK_SZ];

			int x, y, x1, y1, width = dst.cols, height = dst.rows;

			int bh0 = std::min(BLOCK_SZ / 2, height);

			int bw0 = std::min(BLOCK_SZ*BLOCK_SZ / bh0, width);

			bh0 = std::min(BLOCK_SZ*BLOCK_SZ / bw0, height);
			//end = clock();
			//cout << "time cost1: " << double(end - start) / CLK_TCK << " s" << endl;

			//cout << "hello" << endl;
			
			for (y = range.start; y < range.end; y += bh0)
			{
				
				for (x = 0; x < width; x += bw0)

				{
					//if (y / bh0  == 1 && x / bw0  == 1) continue;
					
					int bw = std::min(bw0, width - x);
					int bh = std::min(bh0, range.end - y); // height
					Mat _XY(bh, bw, CV_16SC2, XY), matA;
					Mat dpart(dst, Rect(x, y, bw, bh));
					
					for (y1 = 0; y1 < bh; y1++)
					{
						short* xy = XY + y1 * bw * 2;
						double X0 = M[0] * (x + minw) + M[1] * (y + y1 + minh) + M[2];
						double Y0 = M[3] * (x + minw) + M[4] * (y + y1 + minh) + M[5];
						double W0 = M[6] * (x + minw) + M[7] * (y + y1 + minh) + M[8];
						short* alpha = A + y1 * bw;
						x1 = 0;
						for (; x1 < bw; x1++)

						{

							double W = W0 + M[6] * x1;

							W = W ? INTER_TAB_SIZE / W : 0;

							double fX = std::max((double)INT_MIN, std::min((double)INT_MAX, (X0 + M[0] * x1)*W));

							double fY = std::max((double)INT_MIN, std::min((double)INT_MAX, (Y0 + M[3] * x1)*W));

							int X = saturate_cast<int>(fX);

							int Y = saturate_cast<int>(fY);



							xy[x1 * 2] = saturate_cast<short>(X >> INTER_BITS);

							xy[x1 * 2 + 1] = saturate_cast<short>(Y >> INTER_BITS);

							alpha[x1] = (short)((Y & (INTER_TAB_SIZE - 1))*INTER_TAB_SIZE +

								(X & (INTER_TAB_SIZE - 1)));

						}

					}
					Mat _matA(bh, bw, CV_16U, A);
					remap(src, dpart, _XY, _matA, interpolation);
				}

			}
			
		}
		Mat getDst() {
			return dst;
		}
		int getDstRow() {
			return dst.rows;
		}
		int getDstCol() {
			return dst.cols;
		}
		int getDstTotal() {
			return dst.total();
		}
	private:
		Mat src;
		Mat dst;
		Mat T;

		double maxw;
		double minw;
		double maxh;
		double minh;
	};

	Mat newwarpPerspective(Mat &_src, Mat &_dst, Mat & _T)

	{
		newWarpPerspectiveInvoker invoker(_src, _dst, _T);
		int dstRow = invoker.getDstRow();
		int dstTotal = invoker.getDstTotal();
		Range range(0, dstRow);
		//clock_t  start, end;
		//start = clock();
		parallel_for_(range, invoker, dstTotal / (double)(1 << 16));
		//end = clock();
		//cout << "time cost1: " << double(end - start) / CLK_TCK << " s" << endl;
		return invoker.getDst();
	}
}



	


