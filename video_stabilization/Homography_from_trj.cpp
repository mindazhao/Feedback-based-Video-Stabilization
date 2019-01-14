/*
2015.6.17，编写完成，使用RANSAC+Nelder-Mead算法搜索最佳单应矩阵，并强制其非刚性参数为0
2015.6.17，添加参数限制，倾斜参数、放缩参数不得超出一定阈值
2015.6.18，修改RANSAC算法部分，对于匹配数最多的模型，按照其总体误差值进行排序，取其优者
2015.6.31，6参数改为8参数
2015.7.1，将收敛条件加强，在计算矩阵时候，将double转为double
2015.7.2，修正：Homography_Nelder_Mead中的矩阵A在每次使用前需要清零，因为是+=而不是=，所以每次清零或者声明为局部变量，每次RASAC循环中都重新定义
2015.7.20，在RANSAC部分，更改选点策略，检测随机选择的4个点是否构成一个四边形，防止其中三点甚至四点共线
2015.9.17，添加了DLT+SVD计算单应矩阵的算法，但是误差太大。。。
2015.9.24，Homography_Nelder_Mead_with_outliers多线程计算RANSAC
2015.10.30，参考matlab版本，用random_shuffle算法代替之前的使用四个随机数产生随机抽样序列，效果分分钟好了，内点个数跟matlab版本一样多了
2015.12.15，参考Goldstein，每次RANSAC过程中限制每个块的点数
2016.4.26，使用DLT计算单应矩阵，耗时过长，将放弃
2016.5.26，前景判定中用帧间轨迹的导数计算计算单应矩阵
*/
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
#include <set>
#include <process.h>  
#include <windows.h>  
#include <random>
#include "Homography_from_trj.h"

using namespace cv;
using namespace std;

int myrandom(int i) { return std::rand() % i; }
//归一化算法
void normalization(Mat &X, Mat &X_norm, Mat &norm_mat)
{
	int N = X.cols;
	//形心坐标
	double x0=0, y0=0;//Point2d centroid;
	for(int i=0; i<N; i++)
	{
		x0 += ((double *)X.data)[i];
		y0 += ((double *)X.data)[i+N];
	}
	x0 /= N;
	y0 /= N;
	//计算到形心的距离
	double mean_dist = 0;
	for(int i=0; i<N; i++)
		mean_dist += sqrt((((double *)X.data)[i]-x0)*(((double *)X.data)[i]-x0) + (((double *)X.data)[i+N]-y0)*(((double *)X.data)[i+N]-y0));
	mean_dist /=N;
	double sqrt_2 = sqrt(2.f);
	double mat_0_0 = sqrt_2/mean_dist;
	double mat_0_2 = -1*sqrt_2/mean_dist*x0;
	double mat_1_2 = -1*sqrt_2/mean_dist*y0;
	norm_mat.row(0).col(0) = mat_0_0;
	norm_mat.row(0).col(1) = 0;
	norm_mat.row(0).col(2) = mat_0_2;
	norm_mat.row(1).col(0) = 0;
	norm_mat.row(1).col(1) = mat_0_0;
	norm_mat.row(1).col(2) = mat_1_2;
	norm_mat.row(2).col(0) = 0;
	norm_mat.row(2).col(1) = 0;
	norm_mat.row(2).col(2) = 1;

	X_norm = norm_mat * X;
}
double Residual(Mat H, Mat X1, Mat X2)
{
	int num = X1.cols;
	//评价误差
	Mat X2_ = H*X1;
	Mat X2_row_3 = Mat::zeros(3, num, CV_64F);
	X2_.row(2).copyTo(X2_row_3.row(0));
	X2_.row(2).copyTo(X2_row_3.row(1));
	X2_.row(2).copyTo(X2_row_3.row(2));
	X2_ /= X2_row_3;
	Mat dx = X2_.row(0) - X2.row(0);
	Mat dy = X2_.row(1) - X2.row(1);
	Mat d_x_y = (dx.mul(dx) + dy.mul(dy));
	//返回值err
	double err = sum(d_x_y).val[0];

	return err;
}
//快排
void quick_sort(double *s, vector<Mat> &H_list, int l, int r)
{  
	if (l < r)
	{
		int i = l, j = r;
		double x = s[l];
		Mat temp = H_list[l].clone();
		while (i < j)
		{
			while(i < j && s[j] >= x) // 从右向左找第一个小于x的数  
				j--;
			if(i < j)
			{
				s[i++] = s[j];
				H_list[j].copyTo(H_list[i-1]);
			}

			while(i < j && s[i] < x) // 从左向右找第一个大于等于x的数  
				i++;
			if(i < j)
			{
				s[j--] = s[i];
				H_list[i].copyTo(H_list[j+1]);
			}
		}
		s[i] = x;
		temp.copyTo(H_list[i]);
		quick_sort(s, H_list, l, i - 1); // 递归调用   
		quick_sort(s, H_list, i + 1, r);
	}
}
//参数限定
void constrain_coefficients(Mat &H)
{
	//倾斜参数限定
	if(((double*)H.data)[1] > 0.04)
		((double*)H.data)[1] = 0.04;
	else if(((double*)H.data)[1] < -0.04)
		((double*)H.data)[1] = -0.04;
	if(((double*)H.data)[3] > 0.04)
		((double*)H.data)[3] = 0.04;
	else if(((double*)H.data)[3] < -0.04)
		((double*)H.data)[3] = -0.04;
	//放缩参数限定
	if(((double*)H.data)[0] > 1.04)
		((double*)H.data)[0] = 1.04;
	else if(((double*)H.data)[0] < 0.96)
		((double*)H.data)[0] = 0.96;
	if(((double*)H.data)[4] > 1.04)
		((double*)H.data)[4] = 1.04;
	else if(((double*)H.data)[4] < 0.96)
		((double*)H.data)[4] = 0.96;
}
//参数检查
bool check_coefficients(Mat &H)
{
	//降采样2倍
	//倾斜、放缩参数检查
	if((((double*)H.data)[1] > 0.1) || (((double*)H.data)[1] < -0.1) || (((double*)H.data)[3] > 0.1) || (((double*)H.data)[3] < -0.1) || \
		(((double*)H.data)[0] > 1.1) || (((double*)H.data)[0] < 0.9) || (((double*)H.data)[4] > 1.1) || (((double*)H.data)[4] < 0.9))
		return false;
	else
		return true;
}
void Nelder_Mead(Mat &H0, Mat &pt_bg_inlier, Mat &cor_smooth_inlier, int max_iter, double eps, Mat &H, bool show_best)
{
	const int Max_time = max_iter;
	//透视变换矩阵，即单应性矩阵时，有8个变量
	int var_num = 8;
	vector<Mat> vx(var_num+1);
	H0.copyTo(vx[0]);
	double vf[9] = {0, 0, 0, 0, 0, 0, 0};
	vf[0] = Residual(H0, pt_bg_inlier, cor_smooth_inlier);
	//只将单应矩阵的前两行代入计算
	//cout<<H0<<endl;
	for(int i=0; i<3; i++)
	{
		for(int j=0; j<3; j++)
		{
			if(!(i==2 && j==2))
			{
				H0.copyTo(vx[i*3+j+1]);
				if((fabs(((double*)H0.data)[i*3+j])) < 0.00005)	//如果太小，则认为加上一个很小的扰动
					((double*)vx[i*3+j+1].data)[i*3+j] += 0.005;
				else
					((double*)vx[i*3+j+1].data)[i*3+j] /= 1.05;		//否则，乘以一个系数
				//参数限定
				//constrain_coefficients(vx[i*3+j+1]);
				//cout<<vx[i*3+j+1]<<endl;
				vf[i*3+j+1] = Residual(vx[i*3+j+1], pt_bg_inlier, cor_smooth_inlier);	//计算其对应误差
			}
		}
	}
	//排序
	quick_sort(vf, vx, 0, var_num);

	double max_of_this = 0;
	double max_err = 0;
	while(max_iter>0)
	{
		for(int i=0; i<var_num+1; i++)
		{
			for(int j=i+1; j<var_num+1; j++)
			{
				Mat abs_err = abs(vx[i] - vx[j]);
				for(int k=0; k<3; k++)
				{
					if(((double*)abs_err.data)[k] > max_of_this)
						max_of_this = ((double*)abs_err.data)[k];
					if(((double*)abs_err.data)[k+3] > max_of_this)
						max_of_this = ((double*)abs_err.data)[k+3];
				}
				if(max_of_this > max_err)
					max_err = max_of_this;
			}
		}
		//max_err = fabs(vf[0] - vf[var_num]);
		if(show_best && max_iter %100 == 0)
		{
			if(max_iter %100 == 0)
				cout<<max_err<<"\t";
		}
		//如果各个参数的最大误差足够小，则跳出循环
		//有时候，轨迹数比较少，2*pt_bg_inlier.cols就比较小，收敛条件就会太苛刻
		if(max_err < eps && (vf[0] <= 50))
		{
			if(show_best)
			{
				cout<<"迭代次数:"<<Max_time-max_iter<<endl;
				cout<<"最大最小相差为:"<<max_err<<endl;
				cout<<"已找到最优结果，最小相差为:"<<vf[0]<<endl;
			}
			break;
		}
		//算法核心模块
		Mat best = vx[0];
		double fbest = vf[0];
		Mat soso = vx[var_num-1];
		double fsoso = vf[var_num-1];
		Mat worst = vx[var_num];
		double fworst = vf[var_num];
		Mat center = Mat::zeros(3, 3, CV_64F);
		for(int i=0; i<var_num; i++)
			center += vx[i];
		center /= var_num;
		Mat r = 2*center - worst;
		//参数限定
		//constrain_coefficients(r);
		double fr = Residual(r, pt_bg_inlier, cor_smooth_inlier);
		if(fr < fbest)
		{
			//比最好的结果还好，说明方向正确，考察扩展点，以期望更多的下降
			Mat e = 2*r - center;
			//参数限定
			//constrain_coefficients(e);
			double fe = Residual(e, pt_bg_inlier, cor_smooth_inlier);
			//在扩展点和反射点中选择较优者去替换最差点
			if(fe < fr)
			{
				vx[var_num] = e;//e.clone();
				vf[var_num] = fe;
			}
			else
			{
				vx[var_num] = r;//r.clone();
				vf[var_num] = fr;
			}
		}
		else
		{
			if(fr < fsoso)
			{
				//比次差结果好，能改进
				vx[var_num] = r;//r.clone();
				vf[var_num] = fr;
			}
			else//比次差结果还差，应考虑压缩点
			{
				//当压缩点无法得到更优值的时候，考虑收缩
				bool shrink = false;
				if(fr < fworst)
				{
					//由于r点更优，所以向r点的方向找压缩点
					Mat c = (r + center)/2;
					//参数限定
					//constrain_coefficients(c);
					double fc = Residual(c, pt_bg_inlier, cor_smooth_inlier);
					if(fc < fr)
					{
						//确定从r压缩向c可以改进
						vx[var_num] = c;//c.clone();
						vf[var_num] = fc;
					}
					else
						//否则的话，准备进行收缩
						shrink = true;
				}
				else
				{
					//由于w点更优，所以向w点的方向找压缩点
					Mat c = (worst + center)/2;
					//参数限定
					//constrain_coefficients(c);
					double fc = Residual(c, pt_bg_inlier, cor_smooth_inlier);
					if(fc < fr)
					{
						//确定从r压缩向c可以改进
						vx[var_num] = c;//c.clone();
						vf[var_num] = fc;
					}
					else
						//否则的话，准备进行收缩
						shrink = true;
				}
				if(shrink)
				{
					for(int i=1; i<var_num+1; i++)
					{
						Mat temp = (vx[i] + best) / 2;
						//参数限定
						//constrain_coefficients(temp);
						vx[i] = temp;//temp.clone();
						vf[i] = Residual(vx[i], pt_bg_inlier, cor_smooth_inlier);
					}
				}
			}
		}
		//排序
		quick_sort(vf, vx, 0, var_num);
		//if(max_iter>900)
		//	cout<<"最小误差是"<<vf[0]<<endl;
		max_iter--;
	}
	H = vx[0].clone();
	//cout<<"最优结果"<<H<<endl;
	//cout<<"最小误差是"<<vf[0]<<endl;
	//cout<<Residual(H0, pt_bg_inlier, cor_smooth_inlier);
}
inline double cross_product(Point2d &A, Point2d &B, Point2d &C)
{
	return ((A.x-C.x)*(B.y-C.y)-(B.x-C.x)*(A.y-C.y));
}
//计算单应矩阵，替换findHomography函数,max_iter为Nelder-Mead算法最大迭代次数
//void RANSAC_Foreground_Judgement(vector<Point2d> &pt_bg_cur, vector<Point2d> &Trj_cor_smooth, int max_iter, bool show_ransac, double scale, vector<unsigned char>&Foreground_times, int width, int height)
void RANSAC_Foreground_Judgement(vector<Point2d> &pt_bg_cur, vector<Point2d> &Trj_cor_smooth, int max_iter, bool show_ransac, double scale, unsigned char* Foreground_times, int width, int height, vector<int>&index_frame)
{
	//int64 st, et;
	//st = cvGetTickCount();
	int RANSAC_times = 200;
	double thresh_inlier = 25;// 15 / ((720.0 / height)*(720.0 / height));//80/(scale*scale);
	int num = pt_bg_cur.size();
	//构造归一化坐标向量的矩阵
	Mat pt_bg = Mat::ones(3, num, CV_64F), cor_smooth = Mat::ones(3, num, CV_64F);
	//用于产生随机序列
	vector<int>index_shuffle(num);
	vector<int>block_of_point(num);		//记录每个点属于哪个块
	vector<int>block_num(6);						//记录每个块内的点数
	int blocks = 0;										//记录实际几个块有点
	int row_n = 0, col_n = 0;
	for(int i=0; i<num; i++)
	{
		index_shuffle[i] = i;
		((double*)pt_bg.data)[i] = pt_bg_cur[i].x;
		((double*)pt_bg.data)[i+num] = pt_bg_cur[i].y;
		((double*)cor_smooth.data)[i] = Trj_cor_smooth[i].x;
		((double*)cor_smooth.data)[i+num] = Trj_cor_smooth[i].y;
		row_n = ((int)pt_bg_cur[i].y)/((int)(height/2));
		col_n = ((int)pt_bg_cur[i].x)/((int)(width/3));
		block_of_point[i] = 3*row_n + col_n;
		block_num[block_of_point[i]]++;
	}
	for (int i=0; i<6; i++)
		if (block_num[i])
			blocks++;
	//ofstream outfile_2("temp_smooth.txt");
	//outfile_2<<cor_smooth<<endl;
	//ofstream outfile_1("temp_shaky.txt");
	//outfile_1<<pt_bg<<endl;
	//RANSAC算法，最多100次循环
	srand((unsigned)time(0));
	Mat OK = Mat::zeros(RANSAC_times, num, CV_8U);			//好的结果，1表示该数据与模型匹配得好，0为不好
	vector<int> Score(RANSAC_times);								//评价误差得分，得分越高表示模型越好
	vector<Mat> H(RANSAC_times);									//每次的单应矩阵
	vector<double> Total_err(RANSAC_times);						//总体误差
	Mat thresh = thresh_inlier*Mat::ones(1, num, CV_64F);
	int best_index = -1;		//最好模型的索引值
	int best = -1;				//Score的最大值
	//搜不到在合适范围内的最优值，就再循环一次
	//vector<double> SVD_1000_time(RANSAC_times);
	while(best == -1)
	{
		for(int t=0; t<RANSAC_times; t++)
		{
			//随机抽取四个点，构造左边A矩阵
			vector<int> rand_set;
			vector<int>block_has(6);						//记录在每次RANSAC过程中每个块已经进去的点数，限制每个块最多2个点
			//先用shuffle算法生成随机序列
			random_shuffle(index_shuffle.begin(), index_shuffle.end(), myrandom);
			//限制每个块最多有两个点进入RANSAC，前两个点的压入不用判定
			rand_set.push_back(index_shuffle[0]);
			block_has[block_of_point[index_shuffle[0]]]++;
			rand_set.push_back(index_shuffle[1]);
			block_has[block_of_point[index_shuffle[1]]]++;
			if (blocks==1)
			{
				rand_set.push_back(index_shuffle[2]);
				rand_set.push_back(index_shuffle[3]);
			}
			else
			{
				//压入第三个点
				int shuffle_k = 0;
				int shuffle_index = 2;			
				shuffle_k = index_shuffle[shuffle_index];
				while (block_has[block_of_point[shuffle_k]] == 2 && shuffle_index<num-2)
				{
					shuffle_index++;
					shuffle_k = index_shuffle[shuffle_index];
				}
				block_has[block_of_point[shuffle_k]]++;
				rand_set.push_back(index_shuffle[shuffle_index]);

				//压入第四个点
				shuffle_index++;
				shuffle_k = index_shuffle[shuffle_index];
				while (block_has[block_of_point[shuffle_k]] == 2 && shuffle_index<num-1)
				{
					shuffle_index++;
					shuffle_k = index_shuffle[shuffle_index];
				}
				block_has[block_of_point[shuffle_k]]++;
				rand_set.push_back(index_shuffle[shuffle_index]);
			}

			//A一定要设定为局部变量！！！因为下面使用+=，而不是赋值=！！！！
			Mat A = Mat::zeros(12, 9, CV_64F);
			int j = 0;
			int k = rand_set[0];	//0 <= k < num
			Mat hat = (Mat_<double>(3,3) << 0, -1, ((double*)cor_smooth.data)[k+num], 1, 0, -1*((double*)cor_smooth.data)[k], -1*((double*)cor_smooth.data)[k+num], ((double*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			double x = ((double*)pt_bg.data)[k];
			double y = ((double*)pt_bg.data)[k+num];
			A.rowRange(j*3, j*3+3).colRange(0,3) += hat*x;
			A.rowRange(j*3, j*3+3).colRange(3,6) += hat*y;
			A.rowRange(j*3, j*3+3).colRange(6,9) += hat;

			++j;
			k = rand_set[1];	//0 <= k < num
			hat = (Mat_<double>(3,3) << 0, -1, ((double*)cor_smooth.data)[k+num], 1, 0, -1*((double*)cor_smooth.data)[k], -1*((double*)cor_smooth.data)[k+num], ((double*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			x = ((double*)pt_bg.data)[k];
			y = ((double*)pt_bg.data)[k+num];
			A.rowRange(j*3, j*3+3).colRange(0,3) += hat*x;
			A.rowRange(j*3, j*3+3).colRange(3,6) += hat*y;
			A.rowRange(j*3, j*3+3).colRange(6,9) += hat;

			++j;
			k = rand_set[2];	//0 <= k < num
			hat = (Mat_<double>(3,3) << 0, -1, ((double*)cor_smooth.data)[k+num], 1, 0, -1*((double*)cor_smooth.data)[k], -1*((double*)cor_smooth.data)[k+num], ((double*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			x = ((double*)pt_bg.data)[k];
			y = ((double*)pt_bg.data)[k+num];
			A.rowRange(j*3, j*3+3).colRange(0,3) += hat*x;
			A.rowRange(j*3, j*3+3).colRange(3,6) += hat*y;
			A.rowRange(j*3, j*3+3).colRange(6,9) += hat;

			++j;
			k = rand_set[3];	//0 <= k < num
			hat = (Mat_<double>(3,3) << 0, -1, ((double*)cor_smooth.data)[k+num], 1, 0, -1*((double*)cor_smooth.data)[k], -1*((double*)cor_smooth.data)[k+num], ((double*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			x = ((double*)pt_bg.data)[k];
			y = ((double*)pt_bg.data)[k+num];
			A.rowRange(j*3, j*3+3).colRange(0,3) += hat*x;
			A.rowRange(j*3, j*3+3).colRange(3,6) += hat*y;
			A.rowRange(j*3, j*3+3).colRange(6,9) += hat;

			//cout<<A<<endl;
			//SVD分解生成VT，第9行为最小特征值对应的特征向量
			//测试SVD分解的耗时情况
			//int64 st, et;
			//st = cvGetTickCount();
			//for (int i = 0; i < 1000; i++)
			//{
			//	SVD thissvd(A, SVD::FULL_UV);
			//	Mat VT = thissvd.vt;
			//}
			//et = cvGetTickCount();
			//printf("*******************1000次SVD分解时间: %f***************************************\n", (et - st) / (double)cvGetTickFrequency() / 1000.);
			//SVD_1000_time[t] = (et - st) / (double)cvGetTickFrequency() / 1000.;

			//SVD分解求解
			SVD thissvd(A, SVD::FULL_UV);
			Mat VT = thissvd.vt;
			
			//cout<<VT<<endl;
			//生成本次RANSAC循环对应的归一化的单应矩阵
			H[t] = (Mat_<double>(3,3) <<((double*)VT.data)[72], ((double*)VT.data)[75], ((double*)VT.data)[78], ((double*)VT.data)[73], ((double*)VT.data)[76], ((double*)VT.data)[79], ((double*)VT.data)[74], ((double*)VT.data)[77], ((double*)VT.data)[80]);// / ((double*)VT.data)[80];
			//cout<<H[t]<<endl;
			H[t] /= ((double*)H[t].data)[8];
			//cout<<H[t]<<endl;

			//评价误差
			Mat X2_ = H[t]*pt_bg;
			//cout<<X2_<<endl;
			Mat X2_row_3 = Mat::zeros(3, num, CV_64F);
			X2_.row(2).copyTo(X2_row_3.row(0));
			X2_.row(2).copyTo(X2_row_3.row(1));
			X2_.row(2).copyTo(X2_row_3.row(2));
			X2_ /= X2_row_3;
			//cout<<X2_<<endl;
			Mat dx = X2_.row(0) - cor_smooth.row(0);
			Mat dy = X2_.row(1) - cor_smooth.row(1);
			Mat d_x_y = (dx.mul(dx) + dy.mul(dy));
			//cout<<d_x_y<<endl;
			//结果记录在Total_err、OK和Score矩阵中
			Total_err[t] = sum(d_x_y).val[0];
			OK.row(t) = (d_x_y < thresh) / 255.f;
			//cout<<OK.row(t)<<endl;
			Scalar sum_o = sum(OK.row(t));
			Score[t] = sum(OK.row(t)).val[0];
			//记录最好结果的索引值
			if(Score[t] > best)
			{
				//cout<<H[t]<<endl;
				if(check_coefficients(H[t]))
				{
					best = Score[t];
					best_index = t;
				}
			}
			else if(Score[t] == best)	//模型匹配数量一致时，取误差最小的
			{
				if(Total_err[t] < Total_err[best_index])
				{
					if(check_coefficients(H[t]))
					{
						best = Score[t];
						best_index = t;
					}
				}
			}
		}
	}
	//et = cvGetTickCount();
	//printf("RANSAC循环，100次时间为: %f\n", (et-st)/(double)cvGetTickFrequency()/1000.);
	if(show_ransac)
	{
		cout<<"最好的模型是第"<<best_index<<"个"<<endl;
		cout<<"匹配上了"<<best<<"个"<<endl;
		cout<<"最好的模型是:\n"<<H[best_index]<<endl;
	}
	Mat ok = OK.row(best_index);
	for (int i=0; i<num; i++)
		if (!ok.data[i])
			Foreground_times[index_frame[i]]++;

	////将耗时时间写入文件
	//	ofstream ofile("SVD_1000_time.txt");
	//	for (int i = 0; i < RANSAC_times; i++)
	//		ofile << SVD_1000_time[i] << endl;
}

//计算单应矩阵，替换findHomography函数,max_iter为Nelder-Mead算法最大迭代次数
Mat Homography_Nelder_Mead_with_outliers(vector<Point2d> &pt_bg_cur, vector<Point2d> &Trj_cor_smooth, int max_iter, Mat& outliers, int height)
{
	//int64 st, et;
	//st = cvGetTickCount();
	int RANSAC_times = 1000;
	double thresh_inlier = 35;//30;// 25 / ((720.0 / height)*(720.0 / height));//80/(scale*scale);
	int num = pt_bg_cur.size();
	//构造归一化坐标向量的矩阵
	Mat pt_bg = Mat::ones(3, num, CV_64F), cor_smooth = Mat::ones(3, num, CV_64F);
	//用于产生随机序列
	vector<int>index_shuffle(num);
	for(int i=0; i<num; i++)
	{
		index_shuffle[i] = i;
		((double*)pt_bg.data)[i] = pt_bg_cur[i].x;
		((double*)pt_bg.data)[i+num] = pt_bg_cur[i].y;
		((double*)cor_smooth.data)[i] = Trj_cor_smooth[i].x;
		((double*)cor_smooth.data)[i+num] = Trj_cor_smooth[i].y;
	}

	//RANSAC算法，最多100次循环
	srand((unsigned)time(0));
	Mat OK = Mat::zeros(RANSAC_times, num, CV_8U);			//好的结果，1表示该数据与模型匹配得好，0为不好
	vector<int> Score(RANSAC_times);								//评价误差得分，得分越高表示模型越好
	vector<Mat> H(RANSAC_times);									//每次的单应矩阵
	vector<double> Total_err(RANSAC_times);						//总体误差
	Mat thresh = thresh_inlier*Mat::ones(1, num, CV_64F);
	Mat every_outliers = Mat::zeros(RANSAC_times, num, CV_8U);
	int best_index = -1;		//最好模型的索引值
	int best = -1;				//Score的最大值
	//搜不到在合适范围内的最优值，就再循环一次
	while(best == -1)
	{
		for(int t=0; t<RANSAC_times; t++)
		{
			//随机抽取四个点，构造左边A矩阵
			vector<int> rand_set;
			//先用shuffle算法生成随机序列
			random_shuffle(index_shuffle.begin(), index_shuffle.end(), myrandom);
			rand_set.push_back(index_shuffle[0]);
			rand_set.push_back(index_shuffle[1]);
			rand_set.push_back(index_shuffle[2]);
			rand_set.push_back(index_shuffle[3]);

			//A一定要设定为局部变量！！！因为下面使用+=，而不是赋值=！！！！
			Mat A = Mat::zeros(12, 9, CV_64F);
			int j = 0;
			int k = rand_set[0];	//0 <= k < num
			Mat hat = (Mat_<double>(3,3) << 0, -1, ((double*)cor_smooth.data)[k+num], 1, 0, -1*((double*)cor_smooth.data)[k], -1*((double*)cor_smooth.data)[k+num], ((double*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			double x = ((double*)pt_bg.data)[k];
			double y = ((double*)pt_bg.data)[k+num];
			A.rowRange(j*3, j*3+3).colRange(0,3) += hat*x;
			A.rowRange(j*3, j*3+3).colRange(3,6) += hat*y;
			A.rowRange(j*3, j*3+3).colRange(6,9) += hat;

			++j;
			k = rand_set[1];	//0 <= k < num
			hat = (Mat_<double>(3,3) << 0, -1, ((double*)cor_smooth.data)[k+num], 1, 0, -1*((double*)cor_smooth.data)[k], -1*((double*)cor_smooth.data)[k+num], ((double*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			x = ((double*)pt_bg.data)[k];
			y = ((double*)pt_bg.data)[k+num];
			A.rowRange(j*3, j*3+3).colRange(0,3) += hat*x;
			A.rowRange(j*3, j*3+3).colRange(3,6) += hat*y;
			A.rowRange(j*3, j*3+3).colRange(6,9) += hat;

			++j;
			k = rand_set[2];	//0 <= k < num
			hat = (Mat_<double>(3,3) << 0, -1, ((double*)cor_smooth.data)[k+num], 1, 0, -1*((double*)cor_smooth.data)[k], -1*((double*)cor_smooth.data)[k+num], ((double*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			x = ((double*)pt_bg.data)[k];
			y = ((double*)pt_bg.data)[k+num];
			A.rowRange(j*3, j*3+3).colRange(0,3) += hat*x;
			A.rowRange(j*3, j*3+3).colRange(3,6) += hat*y;
			A.rowRange(j*3, j*3+3).colRange(6,9) += hat;

			//Mat temp = A.rowRange(j*3, j*3+3).colRange(0,3).clone();

			++j;
			k = rand_set[3];	//0 <= k < num
			hat = (Mat_<double>(3,3) << 0, -1, ((double*)cor_smooth.data)[k+num], 1, 0, -1*((double*)cor_smooth.data)[k], -1*((double*)cor_smooth.data)[k+num], ((double*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			x = ((double*)pt_bg.data)[k];
			y = ((double*)pt_bg.data)[k+num];
			A.rowRange(j*3, j*3+3).colRange(0,3) += hat*x;
			A.rowRange(j*3, j*3+3).colRange(3,6) += hat*y;
			A.rowRange(j*3, j*3+3).colRange(6,9) += hat;

			//cout<<A<<endl;
			//ofstream A_file("A_file.txt");
			//A_file<<A<<endl;
			//SVD分解生成VT，第9行为最小特征值对应的特征向量
			SVD thissvd(A,SVD::FULL_UV);
			Mat VT=thissvd.vt; 
			//cout<<VT<<endl;
			//生成本次RANSAC循环对应的归一化的单应矩阵
			H[t] = (Mat_<double>(3,3) <<((double*)VT.data)[72], ((double*)VT.data)[75], ((double*)VT.data)[78], ((double*)VT.data)[73], ((double*)VT.data)[76], ((double*)VT.data)[79], ((double*)VT.data)[74], ((double*)VT.data)[77], ((double*)VT.data)[80]);// / ((double*)VT.data)[80];
			//cout<<H[t]<<endl;
			H[t] /= ((double*)H[t].data)[8];
			//cout<<H[t]<<endl;

			//评价误差
			Mat X2_ = H[t]*pt_bg;
			//cout<<X2_<<endl;
			Mat X2_row_3 = Mat::zeros(3, num, CV_64F);
			X2_.row(2).copyTo(X2_row_3.row(0));
			X2_.row(2).copyTo(X2_row_3.row(1));
			X2_.row(2).copyTo(X2_row_3.row(2));
			X2_ /= X2_row_3;
			//cout<<X2_<<endl;
			Mat dx = X2_.row(0) - cor_smooth.row(0);
			Mat dy = X2_.row(1) - cor_smooth.row(1);
			Mat d_x_y = (dx.mul(dx) + dy.mul(dy));
			//cout<<d_x_y<<endl;
			//结果记录在Total_err、OK和Score矩阵中
			Total_err[t] = sum(d_x_y).val[0];
			OK.row(t) = (d_x_y < thresh) / 255.f;
			//cout<<OK.row(t)<<endl;
			Scalar sum_o = sum(OK.row(t));
			Score[t] = sum(OK.row(t)).val[0];
			//记录最好结果的索引值
			if(Score[t] > best)
			{
				//cout<<H[t]<<endl;
				if(check_coefficients(H[t]))
				{
					best = Score[t];
					best_index = t;
				}
			}
			else if(Score[t] == best)	//模型匹配数量一致时，取误差最小的
			{
				if(Total_err[t] < Total_err[best_index])
				{
					if(check_coefficients(H[t]))
					{
						best = Score[t];
						best_index = t;
					}
				}
			}
		}
	}
	//et = cvGetTickCount();
	//printf("RANSAC循环，100次时间为: %f\n", (et-st)/(double)cvGetTickFrequency()/1000.);
	//cout<<"匹配上了"<<best<<"个"<<endl;
	//outliers = OK.row(best_index).clone();
	//cout<<outliers<<endl;
	//提取出内点
	Mat pt_bg_inlier = Mat::zeros(3, best, CV_64F), cor_smooth_inlier = Mat::zeros(3, best, CV_64F);
	int inlier_ind = 0;
	//cout<<"OK.row(best_index)"<<endl;
	//cout<<OK.row(best_index)<<endl;
	for(int i=0; i<num; i++)
	{
		if(((unsigned char*)OK.data)[best_index*num+i] > 0)
		{
			pt_bg.col(i).copyTo(pt_bg_inlier.col(inlier_ind));
			cor_smooth.col(i).copyTo(cor_smooth_inlier.col(inlier_ind));
			inlier_ind++;
		}
	}
	//Nelder-Mead算法搜索最优值
	//强制单应矩阵变为仿射矩阵，即第3行前两个元素为0
	Mat H0 = H[best_index];
	//Mat H_best = Mat::zeros(3, 3, CV_64F);
	//H0.row(2).col(0) = 0.f;
	//H0.row(2).col(1) = 0.f;
	Mat H_NM = Mat::zeros(3, 3, CV_64F);
	double eps = 0.001;
	//st = cvGetTickCount();
	bool show_ransac = false;
	Nelder_Mead(H0, pt_bg_inlier, cor_smooth_inlier, max_iter, eps, H_NM, show_ransac);
	//et = cvGetTickCount();
	//printf("NM搜索时间: %f\n", (et-st)/(double)cvGetTickFrequency()/1000.);
	return H_NM;
}

//基于RANSAC和轨迹导数的单应矩阵计算函数
Mat Homography_RANSAC_Derivative(vector<Point2d> &pt_bg_last, vector<Point2d> &pt_bg_cur, unsigned char* Foreground_times, int &inlier_num)
{
	int RANSAC_times = 200;
	double thresh_inlier = 5;//30/((720.0/height)*(720.0/height));//80/(scale*scale);
	int num = pt_bg_last.size();
	//构造归一化坐标向量的矩阵
	Mat pt_Mat_last = Mat::ones(3, num, CV_64F), pt_Mat_cur = Mat::ones(3, num, CV_64F);
	Mat pt_derivative = Mat::ones(2, num, CV_64F);
	//用于产生随机序列
	vector<int>index_shuffle(num);
	for(int i=0; i<num; i++)
	{
		index_shuffle[i] = i;
		((double*)pt_Mat_last.data)[i] = pt_bg_last[i].x;
		((double*)pt_Mat_last.data)[i+num] = pt_bg_last[i].y;
		((double*)pt_Mat_cur.data)[i] = pt_bg_cur[i].x;
		((double*)pt_Mat_cur.data)[i+num] = pt_bg_cur[i].y;
		((double*)pt_derivative.data)[i] = pt_bg_cur[i].x - pt_bg_last[i].x;
		((double*)pt_derivative.data)[i+num] = pt_bg_cur[i].y - pt_bg_last[i].y;
	}
	//cout<<"用于计算的两组数据"<<endl;
	//cout<<pt_Mat_cur<<endl;
	//cout<<pt_Mat_last<<endl;

	int best = -1;
	//RANSAC挑选内点
	//RANSAC算法，最多100次循环
	srand((unsigned)time(0));
	Mat OK = Mat::zeros(RANSAC_times, num, CV_8U);			//好的结果，1表示该数据与模型匹配得好，0为不好
	vector<int> Score(RANSAC_times);								//评价误差得分，得分越高表示模型越好
	vector<Mat> H(RANSAC_times);									//每次的单应矩阵
	vector<double> Total_err(RANSAC_times);						//总体误差
	Mat thresh = thresh_inlier*Mat::ones(1, num, CV_64F);
	Mat every_outliers = Mat::zeros(RANSAC_times, num, CV_64F);
	int best_index = -1;		//最好模型的索引值
	//vector<double> LUINV_1000_time(RANSAC_times);
	for(int t=0; t<RANSAC_times; t++)
	{
		//随机抽取四个点，构造左边A矩阵
		vector<int> rand_set;
		//先用shuffle算法生成随机序列
		random_shuffle(index_shuffle.begin(), index_shuffle.end(), myrandom);
		rand_set.push_back(index_shuffle[0]);
		rand_set.push_back(index_shuffle[1]);
		rand_set.push_back(index_shuffle[2]);

		//A一定要设定为局部变量！！！因为下面使用+=，而不是赋值=！！！！
		Mat A = Mat::zeros(6, 6, CV_64F);
		int j = 0;
		int k = rand_set[0];	//0 <= k < num
		Mat hat = (Mat_<double>(2,6) << ((double*)pt_Mat_cur.data)[k], ((double*)pt_Mat_cur.data)[k+num], 1, 0, 0, 0,
			0, 0, 0, ((double*)pt_Mat_cur.data)[k], ((double*)pt_Mat_cur.data)[k+num], 1);
		A.rowRange(j*2, j*2+2).colRange(0,6) += hat;
		
		++j;
		k = rand_set[1];	//0 <= k < num
		hat = (Mat_<double>(2,6) << ((double*)pt_Mat_cur.data)[k], ((double*)pt_Mat_cur.data)[k+num], 1, 0, 0, 0,
			0, 0, 0, ((double*)pt_Mat_cur.data)[k], ((double*)pt_Mat_cur.data)[k+num], 1);
		A.rowRange(j*2, j*2+2).colRange(0,6) += hat;

		++j;
		k = rand_set[2];	//0 <= k < num
		hat = (Mat_<double>(2,6) << ((double*)pt_Mat_cur.data)[k], ((double*)pt_Mat_cur.data)[k+num], 1, 0, 0, 0,
			0, 0, 0, ((double*)pt_Mat_cur.data)[k], ((double*)pt_Mat_cur.data)[k+num], 1);
		A.rowRange(j*2, j*2+2).colRange(0,6) += hat;

		////SVD分解生成VT，第9行为最小特征值对应的特征向量
		//SVD thissvd(A,SVD::FULL_UV);
		//Mat VT=thissvd.vt;
		////生成本次RANSAC循环对应的归一化的单应矩阵
		//H[t] = (Mat_<double>(6, 1) <<((double*)VT.data)[30], ((double*)VT.data)[31], ((double*)VT.data)[32], ((double*)VT.data)[33], ((double*)VT.data)[34], ((double*)VT.data)[35]);
		////H[t] /= ((double*)H[t].data)[5];

		//输出检验
		//cout<<"A矩阵:"<<endl;
		//cout<<A<<endl;
		//ofstream A_file("A.txt");
		//A_file<<A<<endl;
		//ofstream H_file("H.txt");
		//H_file<<H[t]<<endl;
		Mat DX = (Mat_<double>(6, 1) << ((double*)pt_derivative.data)[rand_set[0]], ((double*)pt_derivative.data)[rand_set[0]+num],
			((double*)pt_derivative.data)[rand_set[1]], ((double*)pt_derivative.data)[rand_set[1]+num],
			((double*)pt_derivative.data)[rand_set[2]], ((double*)pt_derivative.data)[rand_set[2]+num]);
		//ofstream DX_file("DX.txt");
		//DX_file<<DX<<endl;

		//求解
		
		//H[t] = A.inv()*(-DX);
		//Mat temp1 = (Mat_<double>(2, 3) << ((double*)H[t].data)[0] + 1, ((double*)H[t].data)[1], ((double*)H[t].data)[2],
		//((double*)H[t].data)[3], ((double*)H[t].data)[4] + 1, ((double*)H[t].data)[5]);
		//cout << "H1: " << temp1 << endl;
		//int64 st, et;
		//st = cvGetTickCount();
		//for (int i = 0; i < 1000; i++)
		//	Mat a = A.inv();
		//et = cvGetTickCount();
		//printf("*******************1000次LU分解求逆时间: %f***************************************\n", (et - st) / (double)cvGetTickFrequency() / 1000.);
		//LUINV_1000_time[t] = (et - st) / (double)cvGetTickFrequency() / 1000.;

		H[t] = A.inv()*DX;
		Mat temp = (Mat_<double>(2, 3) << ((double*)H[t].data)[0], ((double*)H[t].data)[1], ((double*)H[t].data)[2], 
			((double*)H[t].data)[3], ((double*)H[t].data)[4], ((double*)H[t].data)[5]);
		
		//cout <<"H: "<< temp << endl;
		//waitKey(0);

		//评价误差
		Mat delta_X = temp*pt_Mat_cur;
		//cout<<delta_X<<endl;
		//ofstream delta_X_file("delta_X.txt");
		//delta_X_file<<delta_X<<endl;
		Mat d_d_x = delta_X - pt_derivative;
		//cout<<d_d_x<<endl;
		//ofstream d_d_X_file("d_d_x.txt");
		//d_d_X_file<<d_d_x<<endl;
		Mat dx = d_d_x.row(0);
		Mat dy = d_d_x.row(1);
		Mat d_x_y = (dx.mul(dx) + dy.mul(dy));
		every_outliers.row(t) += d_x_y;
		//cout<<d_x_y<<endl;
		//结果记录在Total_err、OK和Score矩阵中
		Total_err[t] = sum(d_x_y).val[0];
		OK.row(t) = (d_x_y < thresh) / 255.f;
		//cout<<OK.row(t)<<endl;
		Scalar sum_o = sum(OK.row(t));
		Score[t] = sum(OK.row(t)).val[0];
		//记录最好结果的索引值
		if(Score[t] > best)
		{
			best = Score[t];
			best_index = t;
		}
		else if(Score[t] == best)	//模型匹配数量一致时，取误差最小的
		{
			if(Total_err[t] < Total_err[best_index])
			{
				best = Score[t];
				best_index = t;
			}
		}
	}

	//记录内点
	//cout<<"匹配上了"<<best<<"个内点"<<endl;
	Mat ok = OK.row(best_index);
	for (int i=0; i<num; i++)
		if (!ok.data[i])
			Foreground_times[i]++;

	inlier_num = best;
	////将耗时时间写入文件
	//ofstream ofile("LUINV_1000_time.txt");
	//for (int i = 0; i < RANSAC_times; i++)
	//	ofile << LUINV_1000_time[i] << endl;

	//返回外点
	//return every_outliers.row(best_index);
	//暂时修改成返回最优模型
	return H[best_index];
}

//DLT归一化算法计算单应矩阵
Mat Homography_DLT(vector<Point2d> &pt_bg_cur, vector<Point2d> &Trj_cor_smooth)
{
	int RANSAC_times = 50;
	double thresh_inlier = 30;//30/((720.0/height)*(720.0/height));//80/(scale*scale);
	int num = pt_bg_cur.size();
	//构造归一化坐标向量的矩阵
	Mat pt_bg = Mat::ones(3, num, CV_64F), cor_smooth = Mat::ones(3, num, CV_64F);
	//用于产生随机序列
	vector<int>index_shuffle(num);
	for(int i=0; i<num; i++)
	{
		index_shuffle[i] = i;
		((double*)pt_bg.data)[i] = pt_bg_cur[i].x;
		((double*)pt_bg.data)[i+num] = pt_bg_cur[i].y;
		((double*)cor_smooth.data)[i] = Trj_cor_smooth[i].x;
		((double*)cor_smooth.data)[i+num] = Trj_cor_smooth[i].y;
	}

	if (num > 4)
	{
		int best = -1;
		//RANSAC挑选内点
		//RANSAC算法，最多100次循环
		srand((unsigned)time(0));
		Mat OK = Mat::zeros(RANSAC_times, num, CV_8U);			//好的结果，1表示该数据与模型匹配得好，0为不好
		vector<int> Score(RANSAC_times);								//评价误差得分，得分越高表示模型越好
		vector<Mat> H(RANSAC_times);									//每次的单应矩阵
		vector<double> Total_err(RANSAC_times);						//总体误差
		Mat thresh = thresh_inlier*Mat::ones(1, num, CV_64F);
		Mat every_outliers = Mat::zeros(RANSAC_times, num, CV_8U);
		int best_index = -1;		//最好模型的索引值

		//搜不到在合适范围内的最优值，就再循环一次
		for(int t=0; t<RANSAC_times; t++)
		{
			//随机抽取四个点，构造左边A矩阵
			vector<int> rand_set;
			//先用shuffle算法生成随机序列
			random_shuffle(index_shuffle.begin(), index_shuffle.end(), myrandom);
			rand_set.push_back(index_shuffle[0]);
			rand_set.push_back(index_shuffle[1]);
			rand_set.push_back(index_shuffle[2]);
			rand_set.push_back(index_shuffle[3]);

			//A一定要设定为局部变量！！！因为下面使用+=，而不是赋值=！！！！
			Mat A = Mat::zeros(12, 9, CV_64F);
			int j = 0;
			int k = rand_set[0];	//0 <= k < num
			Mat hat = (Mat_<double>(3,3) << 0, -1, ((double*)cor_smooth.data)[k+num], 1, 0, -1*((double*)cor_smooth.data)[k], -1*((double*)cor_smooth.data)[k+num], ((double*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			double x = ((double*)pt_bg.data)[k];
			double y = ((double*)pt_bg.data)[k+num];
			A.rowRange(j*3, j*3+3).colRange(0,3) += hat*x;
			A.rowRange(j*3, j*3+3).colRange(3,6) += hat*y;
			A.rowRange(j*3, j*3+3).colRange(6,9) += hat;

			++j;
			k = rand_set[1];	//0 <= k < num
			hat = (Mat_<double>(3,3) << 0, -1, ((double*)cor_smooth.data)[k+num], 1, 0, -1*((double*)cor_smooth.data)[k], -1*((double*)cor_smooth.data)[k+num], ((double*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			x = ((double*)pt_bg.data)[k];
			y = ((double*)pt_bg.data)[k+num];
			A.rowRange(j*3, j*3+3).colRange(0,3) += hat*x;
			A.rowRange(j*3, j*3+3).colRange(3,6) += hat*y;
			A.rowRange(j*3, j*3+3).colRange(6,9) += hat;

			++j;
			k = rand_set[2];	//0 <= k < num
			hat = (Mat_<double>(3,3) << 0, -1, ((double*)cor_smooth.data)[k+num], 1, 0, -1*((double*)cor_smooth.data)[k], -1*((double*)cor_smooth.data)[k+num], ((double*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			x = ((double*)pt_bg.data)[k];
			y = ((double*)pt_bg.data)[k+num];
			A.rowRange(j*3, j*3+3).colRange(0,3) += hat*x;
			A.rowRange(j*3, j*3+3).colRange(3,6) += hat*y;
			A.rowRange(j*3, j*3+3).colRange(6,9) += hat;

			//Mat temp = A.rowRange(j*3, j*3+3).colRange(0,3).clone();

			++j;
			k = rand_set[3];	//0 <= k < num
			hat = (Mat_<double>(3,3) << 0, -1, ((double*)cor_smooth.data)[k+num], 1, 0, -1*((double*)cor_smooth.data)[k], -1*((double*)cor_smooth.data)[k+num], ((double*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			x = ((double*)pt_bg.data)[k];
			y = ((double*)pt_bg.data)[k+num];
			A.rowRange(j*3, j*3+3).colRange(0,3) += hat*x;
			A.rowRange(j*3, j*3+3).colRange(3,6) += hat*y;
			A.rowRange(j*3, j*3+3).colRange(6,9) += hat;

			//SVD分解生成VT，第9行为最小特征值对应的特征向量
			SVD thissvd(A,SVD::FULL_UV);
			Mat VT=thissvd.vt; 
			//生成本次RANSAC循环对应的归一化的单应矩阵
			H[t] = (Mat_<double>(3,3) <<((double*)VT.data)[72], ((double*)VT.data)[75], ((double*)VT.data)[78], ((double*)VT.data)[73], ((double*)VT.data)[76], ((double*)VT.data)[79], ((double*)VT.data)[74], ((double*)VT.data)[77], ((double*)VT.data)[80]);// / ((double*)VT.data)[80];
			H[t] /= ((double*)H[t].data)[8];

			//评价误差
			Mat X2_ = H[t]*pt_bg;
			//cout<<X2_<<endl;
			Mat X2_row_3 = Mat::zeros(3, num, CV_64F);
			X2_.row(2).copyTo(X2_row_3.row(0));
			X2_.row(2).copyTo(X2_row_3.row(1));
			X2_.row(2).copyTo(X2_row_3.row(2));
			X2_ /= X2_row_3;
			//cout<<X2_<<endl;
			Mat dx = X2_.row(0) - cor_smooth.row(0);
			Mat dy = X2_.row(1) - cor_smooth.row(1);
			Mat d_x_y = (dx.mul(dx) + dy.mul(dy));
			//cout<<d_x_y<<endl;
			//结果记录在Total_err、OK和Score矩阵中
			Total_err[t] = sum(d_x_y).val[0];
			OK.row(t) = (d_x_y < thresh) / 255.f;
			//cout<<OK.row(t)<<endl;
			Scalar sum_o = sum(OK.row(t));
			Score[t] = sum(OK.row(t)).val[0];
			//记录最好结果的索引值
			if(Score[t] > best)
			{
				best = Score[t];
				best_index = t;
			}
			else if(Score[t] == best)	//模型匹配数量一致时，取误差最小的
			{
				if(Total_err[t] < Total_err[best_index])
				{
					best = Score[t];
					best_index = t;
				}
			}
		}

		//挑选出内点，再DLT计算
		pt_bg = Mat::ones(3, best, CV_64F), cor_smooth = Mat::ones(3, best, CV_64F);
		for(int i=0; i<num; i++)
		{
			if(((unsigned char*)OK.data)[best_index*best+i] > 0)
			{
				((double*)pt_bg.data)[i] = pt_bg_cur[i].x;
				((double*)pt_bg.data)[i+best] = pt_bg_cur[i].y;
				((double*)cor_smooth.data)[i] = Trj_cor_smooth[i].x;
				((double*)cor_smooth.data)[i+best] = Trj_cor_smooth[i].y;
			}
		}

		num = best;
	}

	Mat norm_mat1 = Mat::ones(3, 3, CV_64F), norm_mat2 = Mat::ones(3, 3, CV_64F);
	Mat X1_norm, X2_norm;
	normalization(pt_bg, X1_norm, norm_mat1);
	normalization(cor_smooth, X2_norm, norm_mat2);

	//DLT计算
	Mat A = Mat::zeros(2*num, 9, CV_64F);
	for (int i=0; i<num; i++)
	{
		Mat temp = (Mat_<double>(2, 9) << 0, 0, 0, ((double*)X1_norm.data)[i], ((double*)X1_norm.data)[num+i], 1, -((double*)X1_norm.data)[i] * ((double*)X2_norm.data)[i], -((double*)X1_norm.data)[i+num] * ((double*)X2_norm.data)[i], -((double*)X2_norm.data)[i+num], \
			((double*)X1_norm.data)[i], ((double*)X1_norm.data)[num+i], 1, 0, 0, 0, -((double*)X1_norm.data)[i] * ((double*)X2_norm.data)[i], -((double*)X1_norm.data)[i+num] * ((double*)X2_norm.data)[i+num], -((double*)X2_norm.data)[i]);
		temp.copyTo(A.rowRange(2*i, 2*i+2).colRange(0, 9));
	}

	SVD thissvd(A,SVD::FULL_UV);
	Mat VT=thissvd.vt; 
	//cout<<VT<<endl;
	//生成本次RANSAC循环对应的归一化的单应矩阵
	Mat Homo = (Mat_<double>(3,3) <<((double*)VT.data)[72], ((double*)VT.data)[75], ((double*)VT.data)[78], ((double*)VT.data)[73], ((double*)VT.data)[76], ((double*)VT.data)[79], ((double*)VT.data)[74], ((double*)VT.data)[77], ((double*)VT.data)[80]);// / ((double*)VT.data)[80];
	Homo /= ((double*)Homo.data)[8];
	//cout<<H<<endl;
	//cout<<norm_mat1<<endl;
	//cout<<norm_mat2<<endl;
	Homo = norm_mat2.inv() * Homo * norm_mat1;
	Homo /= ((double*)Homo.data)[8];
	cout<<Homo<<endl;

	return Homo;
}