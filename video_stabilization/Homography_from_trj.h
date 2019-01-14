#include <opencv2/opencv.hpp>
#include "opencv2/objdetect/objdetect.hpp" 
#include "opencv2/features2d/features2d.hpp" 
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/calib3d/calib3d.hpp" 
#include "opencv2/imgproc/imgproc_c.h" 
#include "opencv2/imgproc/imgproc.hpp"   
#include "opencv2/nonfree/features2d.hpp"
#include<opencv2/legacy/legacy.hpp>
#include <string>
#include <vector>
#include <deque>
#include <set>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <ctime> 

using namespace cv;
using namespace std;


void quick_sort(double *s, vector<Mat> &H_list, int l, int r);
double Residual(Mat H, Mat X1, Mat X2);
void Nelder_Mead(Mat &H0, Mat &pt_bg_inlier, Mat &cor_smooth_inlier, int max_iter, double eps, Mat &H, bool show_ransac);
void RANSAC_Foreground_Judgement(vector<Point2d> &pt_bg_cur, vector<Point2d> &Trj_cor_smooth, int max_iter, bool show_ransac, double scale, unsigned char* Foreground_times, int width, int height, vector<int>&index_frame);
Mat Homography_Nelder_Mead_with_outliers(vector<Point2d> &pt_bg_cur, vector<Point2d> &Trj_cor_smooth, int max_iter, Mat& outliers, int height);
inline double cross_product(Point2d &A, Point2d &B, Point2d &C);
bool check_coefficients(Mat &H);
double Residual(Mat H, Mat X1, Mat X2);

//基于RANSAC和轨迹导数的单应矩阵计算函数
Mat Homography_RANSAC_Derivative(vector<Point2d> &pt_bg_last, vector<Point2d> &pt_bg_cur, unsigned char* Foreground_times, int &inler_num);

//DLT归一化算法计算单应矩阵
Mat Homography_DLT(vector<Point2d> &pt_bg_cur, vector<Point2d> &Trj_cor_smooth);

//Mat Homography_Nelder_Mead_with_outliers(vector<Point2d> &pt_bg_cur, vector<Point2d> &Trj_cor_smooth, int max_iter, Mat& outliers, int height, bool &isCorrect);

//基于RANSAC和背景抖动量的全帧抖动量计算函数
Mat Shakiness_RANSAC(vector<Point2d> &shakiness_bg, vector<Point2d> &pt_bg);