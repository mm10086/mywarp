//#include "opencv.hpp"
//#include <ctime>
//using namespace std;
//using namespace cv;
//
////********去畸变********************
//
//void ReadIntrinsics(Mat &cameraMatrix, Mat &distCoeffs, Size &imageSize, String IntrinsicsPath)
//{
//	bool FSflag = false;
//	FileStorage readfs;
//	FSflag = readfs.open(IntrinsicsPath, FileStorage::READ);
//	if (FSflag == false) cout << "Cannot open the file" << endl;
//	readfs["Camera_Matrix"] >> cameraMatrix;
//	readfs["Distortion_Coefficients"] >> distCoeffs;
//	readfs["image_Width"] >> imageSize.width;
//	readfs["image_Height"] >> imageSize.height;
//	cout << cameraMatrix << endl << distCoeffs << endl << imageSize << endl;
//	readfs.release();
//}
//
//
//
//void Undistort_img(Mat map1, Mat map2, String srcpath, String dstpath)
//{
//	Mat img1, img2;
//	img1 = imread(srcpath);
//	if (img1.empty()) cout << "Cannot open the image" << endl;
//	remap(img1, img2, map1, map2, INTER_LINEAR);
//	imwrite(dstpath, img2);
//	imshow("dst_img", img2);
//}
//
//
//
//void main()
//
//{
//	String src_path = "D:/03Date/vs2017/Chessboard_undistortion/undist/1o.jpg";
//	String dst_path = "D:/03Date/vs2017/Chessboard_undistortion/undist/1dst.jpg";
//	Mat img1 = imread(src_path);
//	imshow("img1", img1);
//	int w = img1.size().width;
//	int h = img1.size().height;
//	//cout << "width: " << w << ", height: "<< h <<endl;
//	//waitKey();
//
//	Mat	cameraMatrix, distCoeffs;
//	Mat map1, map2;
//	Size imageSize;
//
//	clock_t beg, end;
//
//    ////1---1
//	cameraMatrix = (Mat_<double>(3, 3) << 9.86467773e+002, 0., 6.61130981e+002, 0., 9.88230591e+002, 3.53566284e+002, 0., 0., 1.);
//	distCoeffs = (Mat_<double>(1, 4) << -4.27102685e-001, 1.94398835e-001, - 1.95085740e-004, - 5.02103195e-003);
//	////2---3,4,5,6,10,12,14
//	//cameraMatrix = (Mat_<double>(3, 3) << 1.04497583e+003, 0., 6.56181213e+002, 0., 1.04447449e+003, 3.44535522e+002, 0., 0., 1.);
//	//distCoeffs = (Mat_<double>(1, 4) << -4.60653126e-001, 2.15232462e-001, 3.71783390e-003, -2.32074386e-003);
//	////3---2
//	//cameraMatrix = (Mat_<double>(3, 3) << 9.13843140e+002, 0., 6.68015320e+002, 0., 9.14231506e+002, 3.64796021e+002, 0., 0., 1.);
//	//distCoeffs = (Mat_<double>(1, 4) << -3.60711068e-001, 1.29580647e-001, 1.64326758e-003, -2.14289618e-003);
//	////7---7,8,9,11,13
//	//cameraMatrix = (Mat_<double>(3, 3) << 7.54361145e+002, 0. ,6.56909302e+002, 0., 7.55194458e+002, 3.66052032e+002, 0., 0., 1.);
//	//distCoeffs = (Mat_<double>(1, 4) << -2.37145290e-001, 5.63652553e-002, -8.43407819e-004, -1.32338027e-003);
//	
//	imageSize.width = w;
//	imageSize.height = h;
//
//	//String IntrinsicsPath = "D:/03Date/vs2017/Chessboard_undistortion/undist/Intrinsics.yaml";
//
//	//ReadIntrinsics(cameraMatrix, distCoeffs, imageSize, IntrinsicsPath);
//	
//	// 去畸变并保留完整图像
//	initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
//	getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0), imageSize, CV_16SC2, map1, map2);
//	Undistort_img(map1, map2, src_path, dst_path);
//	
//	////// 去畸变保留裁剪图
//	//initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(), Mat(),imageSize, CV_16SC2, map1, map2);
//	//Undistort_img(map1, map2, src_path, dst_path);
//
//	waitKey();
//}
