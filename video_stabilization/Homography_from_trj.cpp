/*
2015.6.17����д��ɣ�ʹ��RANSAC+Nelder-Mead�㷨������ѵ�Ӧ���󣬲�ǿ����Ǹ��Բ���Ϊ0
2015.6.17����Ӳ������ƣ���б�����������������ó���һ����ֵ
2015.6.18���޸�RANSAC�㷨���֣�����ƥ��������ģ�ͣ��������������ֵ��������ȡ������
2015.6.31��6������Ϊ8����
2015.7.1��������������ǿ���ڼ������ʱ�򣬽�doubleתΪdouble
2015.7.2��������Homography_Nelder_Mead�еľ���A��ÿ��ʹ��ǰ��Ҫ���㣬��Ϊ��+=������=������ÿ�������������Ϊ�ֲ�������ÿ��RASACѭ���ж����¶���
2015.7.20����RANSAC���֣�����ѡ����ԣ�������ѡ���4�����Ƿ񹹳�һ���ı��Σ���ֹ�������������ĵ㹲��
2015.9.17�������DLT+SVD���㵥Ӧ������㷨���������̫�󡣡���
2015.9.24��Homography_Nelder_Mead_with_outliers���̼߳���RANSAC
2015.10.30���ο�matlab�汾����random_shuffle�㷨����֮ǰ��ʹ���ĸ��������������������У�Ч���ַ��Ӻ��ˣ��ڵ������matlab�汾һ������
2015.12.15���ο�Goldstein��ÿ��RANSAC����������ÿ����ĵ���
2016.4.26��ʹ��DLT���㵥Ӧ���󣬺�ʱ������������
2016.5.26��ǰ���ж�����֡��켣�ĵ���������㵥Ӧ����
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
//��һ���㷨
void normalization(Mat &X, Mat &X_norm, Mat &norm_mat)
{
	int N = X.cols;
	//��������
	double x0=0, y0=0;//Point2d centroid;
	for(int i=0; i<N; i++)
	{
		x0 += ((double *)X.data)[i];
		y0 += ((double *)X.data)[i+N];
	}
	x0 /= N;
	y0 /= N;
	//���㵽���ĵľ���
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
	//�������
	Mat X2_ = H*X1;
	Mat X2_row_3 = Mat::zeros(3, num, CV_64F);
	X2_.row(2).copyTo(X2_row_3.row(0));
	X2_.row(2).copyTo(X2_row_3.row(1));
	X2_.row(2).copyTo(X2_row_3.row(2));
	X2_ /= X2_row_3;
	Mat dx = X2_.row(0) - X2.row(0);
	Mat dy = X2_.row(1) - X2.row(1);
	Mat d_x_y = (dx.mul(dx) + dy.mul(dy));
	//����ֵerr
	double err = sum(d_x_y).val[0];

	return err;
}
//����
void quick_sort(double *s, vector<Mat> &H_list, int l, int r)
{  
	if (l < r)
	{
		int i = l, j = r;
		double x = s[l];
		Mat temp = H_list[l].clone();
		while (i < j)
		{
			while(i < j && s[j] >= x) // ���������ҵ�һ��С��x����  
				j--;
			if(i < j)
			{
				s[i++] = s[j];
				H_list[j].copyTo(H_list[i-1]);
			}

			while(i < j && s[i] < x) // ���������ҵ�һ�����ڵ���x����  
				i++;
			if(i < j)
			{
				s[j--] = s[i];
				H_list[i].copyTo(H_list[j+1]);
			}
		}
		s[i] = x;
		temp.copyTo(H_list[i]);
		quick_sort(s, H_list, l, i - 1); // �ݹ����   
		quick_sort(s, H_list, i + 1, r);
	}
}
//�����޶�
void constrain_coefficients(Mat &H)
{
	//��б�����޶�
	if(((double*)H.data)[1] > 0.04)
		((double*)H.data)[1] = 0.04;
	else if(((double*)H.data)[1] < -0.04)
		((double*)H.data)[1] = -0.04;
	if(((double*)H.data)[3] > 0.04)
		((double*)H.data)[3] = 0.04;
	else if(((double*)H.data)[3] < -0.04)
		((double*)H.data)[3] = -0.04;
	//���������޶�
	if(((double*)H.data)[0] > 1.04)
		((double*)H.data)[0] = 1.04;
	else if(((double*)H.data)[0] < 0.96)
		((double*)H.data)[0] = 0.96;
	if(((double*)H.data)[4] > 1.04)
		((double*)H.data)[4] = 1.04;
	else if(((double*)H.data)[4] < 0.96)
		((double*)H.data)[4] = 0.96;
}
//�������
bool check_coefficients(Mat &H)
{
	//������2��
	//��б�������������
	if((((double*)H.data)[1] > 0.1) || (((double*)H.data)[1] < -0.1) || (((double*)H.data)[3] > 0.1) || (((double*)H.data)[3] < -0.1) || \
		(((double*)H.data)[0] > 1.1) || (((double*)H.data)[0] < 0.9) || (((double*)H.data)[4] > 1.1) || (((double*)H.data)[4] < 0.9))
		return false;
	else
		return true;
}
void Nelder_Mead(Mat &H0, Mat &pt_bg_inlier, Mat &cor_smooth_inlier, int max_iter, double eps, Mat &H, bool show_best)
{
	const int Max_time = max_iter;
	//͸�ӱ任���󣬼���Ӧ�Ծ���ʱ����8������
	int var_num = 8;
	vector<Mat> vx(var_num+1);
	H0.copyTo(vx[0]);
	double vf[9] = {0, 0, 0, 0, 0, 0, 0};
	vf[0] = Residual(H0, pt_bg_inlier, cor_smooth_inlier);
	//ֻ����Ӧ�����ǰ���д������
	//cout<<H0<<endl;
	for(int i=0; i<3; i++)
	{
		for(int j=0; j<3; j++)
		{
			if(!(i==2 && j==2))
			{
				H0.copyTo(vx[i*3+j+1]);
				if((fabs(((double*)H0.data)[i*3+j])) < 0.00005)	//���̫С������Ϊ����һ����С���Ŷ�
					((double*)vx[i*3+j+1].data)[i*3+j] += 0.005;
				else
					((double*)vx[i*3+j+1].data)[i*3+j] /= 1.05;		//���򣬳���һ��ϵ��
				//�����޶�
				//constrain_coefficients(vx[i*3+j+1]);
				//cout<<vx[i*3+j+1]<<endl;
				vf[i*3+j+1] = Residual(vx[i*3+j+1], pt_bg_inlier, cor_smooth_inlier);	//�������Ӧ���
			}
		}
	}
	//����
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
		//��������������������㹻С��������ѭ��
		//��ʱ�򣬹켣���Ƚ��٣�2*pt_bg_inlier.cols�ͱȽ�С�����������ͻ�̫����
		if(max_err < eps && (vf[0] <= 50))
		{
			if(show_best)
			{
				cout<<"��������:"<<Max_time-max_iter<<endl;
				cout<<"�����С���Ϊ:"<<max_err<<endl;
				cout<<"���ҵ����Ž������С���Ϊ:"<<vf[0]<<endl;
			}
			break;
		}
		//�㷨����ģ��
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
		//�����޶�
		//constrain_coefficients(r);
		double fr = Residual(r, pt_bg_inlier, cor_smooth_inlier);
		if(fr < fbest)
		{
			//����õĽ�����ã�˵��������ȷ��������չ�㣬������������½�
			Mat e = 2*r - center;
			//�����޶�
			//constrain_coefficients(e);
			double fe = Residual(e, pt_bg_inlier, cor_smooth_inlier);
			//����չ��ͷ������ѡ�������ȥ�滻����
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
				//�ȴβ����ã��ܸĽ�
				vx[var_num] = r;//r.clone();
				vf[var_num] = fr;
			}
			else//�ȴβ������Ӧ����ѹ����
			{
				//��ѹ�����޷��õ�����ֵ��ʱ�򣬿�������
				bool shrink = false;
				if(fr < fworst)
				{
					//����r����ţ�������r��ķ�����ѹ����
					Mat c = (r + center)/2;
					//�����޶�
					//constrain_coefficients(c);
					double fc = Residual(c, pt_bg_inlier, cor_smooth_inlier);
					if(fc < fr)
					{
						//ȷ����rѹ����c���ԸĽ�
						vx[var_num] = c;//c.clone();
						vf[var_num] = fc;
					}
					else
						//����Ļ���׼����������
						shrink = true;
				}
				else
				{
					//����w����ţ�������w��ķ�����ѹ����
					Mat c = (worst + center)/2;
					//�����޶�
					//constrain_coefficients(c);
					double fc = Residual(c, pt_bg_inlier, cor_smooth_inlier);
					if(fc < fr)
					{
						//ȷ����rѹ����c���ԸĽ�
						vx[var_num] = c;//c.clone();
						vf[var_num] = fc;
					}
					else
						//����Ļ���׼����������
						shrink = true;
				}
				if(shrink)
				{
					for(int i=1; i<var_num+1; i++)
					{
						Mat temp = (vx[i] + best) / 2;
						//�����޶�
						//constrain_coefficients(temp);
						vx[i] = temp;//temp.clone();
						vf[i] = Residual(vx[i], pt_bg_inlier, cor_smooth_inlier);
					}
				}
			}
		}
		//����
		quick_sort(vf, vx, 0, var_num);
		//if(max_iter>900)
		//	cout<<"��С�����"<<vf[0]<<endl;
		max_iter--;
	}
	H = vx[0].clone();
	//cout<<"���Ž��"<<H<<endl;
	//cout<<"��С�����"<<vf[0]<<endl;
	//cout<<Residual(H0, pt_bg_inlier, cor_smooth_inlier);
}
inline double cross_product(Point2d &A, Point2d &B, Point2d &C)
{
	return ((A.x-C.x)*(B.y-C.y)-(B.x-C.x)*(A.y-C.y));
}
//���㵥Ӧ�����滻findHomography����,max_iterΪNelder-Mead�㷨����������
//void RANSAC_Foreground_Judgement(vector<Point2d> &pt_bg_cur, vector<Point2d> &Trj_cor_smooth, int max_iter, bool show_ransac, double scale, vector<unsigned char>&Foreground_times, int width, int height)
void RANSAC_Foreground_Judgement(vector<Point2d> &pt_bg_cur, vector<Point2d> &Trj_cor_smooth, int max_iter, bool show_ransac, double scale, unsigned char* Foreground_times, int width, int height, vector<int>&index_frame)
{
	//int64 st, et;
	//st = cvGetTickCount();
	int RANSAC_times = 200;
	double thresh_inlier = 25;// 15 / ((720.0 / height)*(720.0 / height));//80/(scale*scale);
	int num = pt_bg_cur.size();
	//�����һ�����������ľ���
	Mat pt_bg = Mat::ones(3, num, CV_64F), cor_smooth = Mat::ones(3, num, CV_64F);
	//���ڲ����������
	vector<int>index_shuffle(num);
	vector<int>block_of_point(num);		//��¼ÿ���������ĸ���
	vector<int>block_num(6);						//��¼ÿ�����ڵĵ���
	int blocks = 0;										//��¼ʵ�ʼ������е�
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
	//RANSAC�㷨�����100��ѭ��
	srand((unsigned)time(0));
	Mat OK = Mat::zeros(RANSAC_times, num, CV_8U);			//�õĽ����1��ʾ��������ģ��ƥ��úã�0Ϊ����
	vector<int> Score(RANSAC_times);								//�������÷֣��÷�Խ�߱�ʾģ��Խ��
	vector<Mat> H(RANSAC_times);									//ÿ�εĵ�Ӧ����
	vector<double> Total_err(RANSAC_times);						//�������
	Mat thresh = thresh_inlier*Mat::ones(1, num, CV_64F);
	int best_index = -1;		//���ģ�͵�����ֵ
	int best = -1;				//Score�����ֵ
	//�Ѳ����ں��ʷ�Χ�ڵ�����ֵ������ѭ��һ��
	//vector<double> SVD_1000_time(RANSAC_times);
	while(best == -1)
	{
		for(int t=0; t<RANSAC_times; t++)
		{
			//�����ȡ�ĸ��㣬�������A����
			vector<int> rand_set;
			vector<int>block_has(6);						//��¼��ÿ��RANSAC������ÿ�����Ѿ���ȥ�ĵ���������ÿ�������2����
			//����shuffle�㷨�����������
			random_shuffle(index_shuffle.begin(), index_shuffle.end(), myrandom);
			//����ÿ������������������RANSAC��ǰ�������ѹ�벻���ж�
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
				//ѹ���������
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

				//ѹ����ĸ���
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

			//Aһ��Ҫ�趨Ϊ�ֲ�������������Ϊ����ʹ��+=�������Ǹ�ֵ=��������
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
			//SVD�ֽ�����VT����9��Ϊ��С����ֵ��Ӧ����������
			//����SVD�ֽ�ĺ�ʱ���
			//int64 st, et;
			//st = cvGetTickCount();
			//for (int i = 0; i < 1000; i++)
			//{
			//	SVD thissvd(A, SVD::FULL_UV);
			//	Mat VT = thissvd.vt;
			//}
			//et = cvGetTickCount();
			//printf("*******************1000��SVD�ֽ�ʱ��: %f***************************************\n", (et - st) / (double)cvGetTickFrequency() / 1000.);
			//SVD_1000_time[t] = (et - st) / (double)cvGetTickFrequency() / 1000.;

			//SVD�ֽ����
			SVD thissvd(A, SVD::FULL_UV);
			Mat VT = thissvd.vt;
			
			//cout<<VT<<endl;
			//���ɱ���RANSACѭ����Ӧ�Ĺ�һ���ĵ�Ӧ����
			H[t] = (Mat_<double>(3,3) <<((double*)VT.data)[72], ((double*)VT.data)[75], ((double*)VT.data)[78], ((double*)VT.data)[73], ((double*)VT.data)[76], ((double*)VT.data)[79], ((double*)VT.data)[74], ((double*)VT.data)[77], ((double*)VT.data)[80]);// / ((double*)VT.data)[80];
			//cout<<H[t]<<endl;
			H[t] /= ((double*)H[t].data)[8];
			//cout<<H[t]<<endl;

			//�������
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
			//�����¼��Total_err��OK��Score������
			Total_err[t] = sum(d_x_y).val[0];
			OK.row(t) = (d_x_y < thresh) / 255.f;
			//cout<<OK.row(t)<<endl;
			Scalar sum_o = sum(OK.row(t));
			Score[t] = sum(OK.row(t)).val[0];
			//��¼��ý��������ֵ
			if(Score[t] > best)
			{
				//cout<<H[t]<<endl;
				if(check_coefficients(H[t]))
				{
					best = Score[t];
					best_index = t;
				}
			}
			else if(Score[t] == best)	//ģ��ƥ������һ��ʱ��ȡ�����С��
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
	//printf("RANSACѭ����100��ʱ��Ϊ: %f\n", (et-st)/(double)cvGetTickFrequency()/1000.);
	if(show_ransac)
	{
		cout<<"��õ�ģ���ǵ�"<<best_index<<"��"<<endl;
		cout<<"ƥ������"<<best<<"��"<<endl;
		cout<<"��õ�ģ����:\n"<<H[best_index]<<endl;
	}
	Mat ok = OK.row(best_index);
	for (int i=0; i<num; i++)
		if (!ok.data[i])
			Foreground_times[index_frame[i]]++;

	////����ʱʱ��д���ļ�
	//	ofstream ofile("SVD_1000_time.txt");
	//	for (int i = 0; i < RANSAC_times; i++)
	//		ofile << SVD_1000_time[i] << endl;
}

//���㵥Ӧ�����滻findHomography����,max_iterΪNelder-Mead�㷨����������
Mat Homography_Nelder_Mead_with_outliers(vector<Point2d> &pt_bg_cur, vector<Point2d> &Trj_cor_smooth, int max_iter, Mat& outliers, int height)
{
	//int64 st, et;
	//st = cvGetTickCount();
	int RANSAC_times = 1000;
	double thresh_inlier = 35;//30;// 25 / ((720.0 / height)*(720.0 / height));//80/(scale*scale);
	int num = pt_bg_cur.size();
	//�����һ�����������ľ���
	Mat pt_bg = Mat::ones(3, num, CV_64F), cor_smooth = Mat::ones(3, num, CV_64F);
	//���ڲ����������
	vector<int>index_shuffle(num);
	for(int i=0; i<num; i++)
	{
		index_shuffle[i] = i;
		((double*)pt_bg.data)[i] = pt_bg_cur[i].x;
		((double*)pt_bg.data)[i+num] = pt_bg_cur[i].y;
		((double*)cor_smooth.data)[i] = Trj_cor_smooth[i].x;
		((double*)cor_smooth.data)[i+num] = Trj_cor_smooth[i].y;
	}

	//RANSAC�㷨�����100��ѭ��
	srand((unsigned)time(0));
	Mat OK = Mat::zeros(RANSAC_times, num, CV_8U);			//�õĽ����1��ʾ��������ģ��ƥ��úã�0Ϊ����
	vector<int> Score(RANSAC_times);								//�������÷֣��÷�Խ�߱�ʾģ��Խ��
	vector<Mat> H(RANSAC_times);									//ÿ�εĵ�Ӧ����
	vector<double> Total_err(RANSAC_times);						//�������
	Mat thresh = thresh_inlier*Mat::ones(1, num, CV_64F);
	Mat every_outliers = Mat::zeros(RANSAC_times, num, CV_8U);
	int best_index = -1;		//���ģ�͵�����ֵ
	int best = -1;				//Score�����ֵ
	//�Ѳ����ں��ʷ�Χ�ڵ�����ֵ������ѭ��һ��
	while(best == -1)
	{
		for(int t=0; t<RANSAC_times; t++)
		{
			//�����ȡ�ĸ��㣬�������A����
			vector<int> rand_set;
			//����shuffle�㷨�����������
			random_shuffle(index_shuffle.begin(), index_shuffle.end(), myrandom);
			rand_set.push_back(index_shuffle[0]);
			rand_set.push_back(index_shuffle[1]);
			rand_set.push_back(index_shuffle[2]);
			rand_set.push_back(index_shuffle[3]);

			//Aһ��Ҫ�趨Ϊ�ֲ�������������Ϊ����ʹ��+=�������Ǹ�ֵ=��������
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
			//SVD�ֽ�����VT����9��Ϊ��С����ֵ��Ӧ����������
			SVD thissvd(A,SVD::FULL_UV);
			Mat VT=thissvd.vt; 
			//cout<<VT<<endl;
			//���ɱ���RANSACѭ����Ӧ�Ĺ�һ���ĵ�Ӧ����
			H[t] = (Mat_<double>(3,3) <<((double*)VT.data)[72], ((double*)VT.data)[75], ((double*)VT.data)[78], ((double*)VT.data)[73], ((double*)VT.data)[76], ((double*)VT.data)[79], ((double*)VT.data)[74], ((double*)VT.data)[77], ((double*)VT.data)[80]);// / ((double*)VT.data)[80];
			//cout<<H[t]<<endl;
			H[t] /= ((double*)H[t].data)[8];
			//cout<<H[t]<<endl;

			//�������
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
			//�����¼��Total_err��OK��Score������
			Total_err[t] = sum(d_x_y).val[0];
			OK.row(t) = (d_x_y < thresh) / 255.f;
			//cout<<OK.row(t)<<endl;
			Scalar sum_o = sum(OK.row(t));
			Score[t] = sum(OK.row(t)).val[0];
			//��¼��ý��������ֵ
			if(Score[t] > best)
			{
				//cout<<H[t]<<endl;
				if(check_coefficients(H[t]))
				{
					best = Score[t];
					best_index = t;
				}
			}
			else if(Score[t] == best)	//ģ��ƥ������һ��ʱ��ȡ�����С��
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
	//printf("RANSACѭ����100��ʱ��Ϊ: %f\n", (et-st)/(double)cvGetTickFrequency()/1000.);
	//cout<<"ƥ������"<<best<<"��"<<endl;
	//outliers = OK.row(best_index).clone();
	//cout<<outliers<<endl;
	//��ȡ���ڵ�
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
	//Nelder-Mead�㷨��������ֵ
	//ǿ�Ƶ�Ӧ�����Ϊ������󣬼���3��ǰ����Ԫ��Ϊ0
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
	//printf("NM����ʱ��: %f\n", (et-st)/(double)cvGetTickFrequency()/1000.);
	return H_NM;
}

//����RANSAC�͹켣�����ĵ�Ӧ������㺯��
Mat Homography_RANSAC_Derivative(vector<Point2d> &pt_bg_last, vector<Point2d> &pt_bg_cur, unsigned char* Foreground_times, int &inlier_num)
{
	int RANSAC_times = 200;
	double thresh_inlier = 5;//30/((720.0/height)*(720.0/height));//80/(scale*scale);
	int num = pt_bg_last.size();
	//�����һ�����������ľ���
	Mat pt_Mat_last = Mat::ones(3, num, CV_64F), pt_Mat_cur = Mat::ones(3, num, CV_64F);
	Mat pt_derivative = Mat::ones(2, num, CV_64F);
	//���ڲ����������
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
	//cout<<"���ڼ������������"<<endl;
	//cout<<pt_Mat_cur<<endl;
	//cout<<pt_Mat_last<<endl;

	int best = -1;
	//RANSAC��ѡ�ڵ�
	//RANSAC�㷨�����100��ѭ��
	srand((unsigned)time(0));
	Mat OK = Mat::zeros(RANSAC_times, num, CV_8U);			//�õĽ����1��ʾ��������ģ��ƥ��úã�0Ϊ����
	vector<int> Score(RANSAC_times);								//�������÷֣��÷�Խ�߱�ʾģ��Խ��
	vector<Mat> H(RANSAC_times);									//ÿ�εĵ�Ӧ����
	vector<double> Total_err(RANSAC_times);						//�������
	Mat thresh = thresh_inlier*Mat::ones(1, num, CV_64F);
	Mat every_outliers = Mat::zeros(RANSAC_times, num, CV_64F);
	int best_index = -1;		//���ģ�͵�����ֵ
	//vector<double> LUINV_1000_time(RANSAC_times);
	for(int t=0; t<RANSAC_times; t++)
	{
		//�����ȡ�ĸ��㣬�������A����
		vector<int> rand_set;
		//����shuffle�㷨�����������
		random_shuffle(index_shuffle.begin(), index_shuffle.end(), myrandom);
		rand_set.push_back(index_shuffle[0]);
		rand_set.push_back(index_shuffle[1]);
		rand_set.push_back(index_shuffle[2]);

		//Aһ��Ҫ�趨Ϊ�ֲ�������������Ϊ����ʹ��+=�������Ǹ�ֵ=��������
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

		////SVD�ֽ�����VT����9��Ϊ��С����ֵ��Ӧ����������
		//SVD thissvd(A,SVD::FULL_UV);
		//Mat VT=thissvd.vt;
		////���ɱ���RANSACѭ����Ӧ�Ĺ�һ���ĵ�Ӧ����
		//H[t] = (Mat_<double>(6, 1) <<((double*)VT.data)[30], ((double*)VT.data)[31], ((double*)VT.data)[32], ((double*)VT.data)[33], ((double*)VT.data)[34], ((double*)VT.data)[35]);
		////H[t] /= ((double*)H[t].data)[5];

		//�������
		//cout<<"A����:"<<endl;
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

		//���
		
		//H[t] = A.inv()*(-DX);
		//Mat temp1 = (Mat_<double>(2, 3) << ((double*)H[t].data)[0] + 1, ((double*)H[t].data)[1], ((double*)H[t].data)[2],
		//((double*)H[t].data)[3], ((double*)H[t].data)[4] + 1, ((double*)H[t].data)[5]);
		//cout << "H1: " << temp1 << endl;
		//int64 st, et;
		//st = cvGetTickCount();
		//for (int i = 0; i < 1000; i++)
		//	Mat a = A.inv();
		//et = cvGetTickCount();
		//printf("*******************1000��LU�ֽ�����ʱ��: %f***************************************\n", (et - st) / (double)cvGetTickFrequency() / 1000.);
		//LUINV_1000_time[t] = (et - st) / (double)cvGetTickFrequency() / 1000.;

		H[t] = A.inv()*DX;
		Mat temp = (Mat_<double>(2, 3) << ((double*)H[t].data)[0], ((double*)H[t].data)[1], ((double*)H[t].data)[2], 
			((double*)H[t].data)[3], ((double*)H[t].data)[4], ((double*)H[t].data)[5]);
		
		//cout <<"H: "<< temp << endl;
		//waitKey(0);

		//�������
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
		//�����¼��Total_err��OK��Score������
		Total_err[t] = sum(d_x_y).val[0];
		OK.row(t) = (d_x_y < thresh) / 255.f;
		//cout<<OK.row(t)<<endl;
		Scalar sum_o = sum(OK.row(t));
		Score[t] = sum(OK.row(t)).val[0];
		//��¼��ý��������ֵ
		if(Score[t] > best)
		{
			best = Score[t];
			best_index = t;
		}
		else if(Score[t] == best)	//ģ��ƥ������һ��ʱ��ȡ�����С��
		{
			if(Total_err[t] < Total_err[best_index])
			{
				best = Score[t];
				best_index = t;
			}
		}
	}

	//��¼�ڵ�
	//cout<<"ƥ������"<<best<<"���ڵ�"<<endl;
	Mat ok = OK.row(best_index);
	for (int i=0; i<num; i++)
		if (!ok.data[i])
			Foreground_times[i]++;

	inlier_num = best;
	////����ʱʱ��д���ļ�
	//ofstream ofile("LUINV_1000_time.txt");
	//for (int i = 0; i < RANSAC_times; i++)
	//	ofile << LUINV_1000_time[i] << endl;

	//�������
	//return every_outliers.row(best_index);
	//��ʱ�޸ĳɷ�������ģ��
	return H[best_index];
}

//DLT��һ���㷨���㵥Ӧ����
Mat Homography_DLT(vector<Point2d> &pt_bg_cur, vector<Point2d> &Trj_cor_smooth)
{
	int RANSAC_times = 50;
	double thresh_inlier = 30;//30/((720.0/height)*(720.0/height));//80/(scale*scale);
	int num = pt_bg_cur.size();
	//�����һ�����������ľ���
	Mat pt_bg = Mat::ones(3, num, CV_64F), cor_smooth = Mat::ones(3, num, CV_64F);
	//���ڲ����������
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
		//RANSAC��ѡ�ڵ�
		//RANSAC�㷨�����100��ѭ��
		srand((unsigned)time(0));
		Mat OK = Mat::zeros(RANSAC_times, num, CV_8U);			//�õĽ����1��ʾ��������ģ��ƥ��úã�0Ϊ����
		vector<int> Score(RANSAC_times);								//�������÷֣��÷�Խ�߱�ʾģ��Խ��
		vector<Mat> H(RANSAC_times);									//ÿ�εĵ�Ӧ����
		vector<double> Total_err(RANSAC_times);						//�������
		Mat thresh = thresh_inlier*Mat::ones(1, num, CV_64F);
		Mat every_outliers = Mat::zeros(RANSAC_times, num, CV_8U);
		int best_index = -1;		//���ģ�͵�����ֵ

		//�Ѳ����ں��ʷ�Χ�ڵ�����ֵ������ѭ��һ��
		for(int t=0; t<RANSAC_times; t++)
		{
			//�����ȡ�ĸ��㣬�������A����
			vector<int> rand_set;
			//����shuffle�㷨�����������
			random_shuffle(index_shuffle.begin(), index_shuffle.end(), myrandom);
			rand_set.push_back(index_shuffle[0]);
			rand_set.push_back(index_shuffle[1]);
			rand_set.push_back(index_shuffle[2]);
			rand_set.push_back(index_shuffle[3]);

			//Aһ��Ҫ�趨Ϊ�ֲ�������������Ϊ����ʹ��+=�������Ǹ�ֵ=��������
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

			//SVD�ֽ�����VT����9��Ϊ��С����ֵ��Ӧ����������
			SVD thissvd(A,SVD::FULL_UV);
			Mat VT=thissvd.vt; 
			//���ɱ���RANSACѭ����Ӧ�Ĺ�һ���ĵ�Ӧ����
			H[t] = (Mat_<double>(3,3) <<((double*)VT.data)[72], ((double*)VT.data)[75], ((double*)VT.data)[78], ((double*)VT.data)[73], ((double*)VT.data)[76], ((double*)VT.data)[79], ((double*)VT.data)[74], ((double*)VT.data)[77], ((double*)VT.data)[80]);// / ((double*)VT.data)[80];
			H[t] /= ((double*)H[t].data)[8];

			//�������
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
			//�����¼��Total_err��OK��Score������
			Total_err[t] = sum(d_x_y).val[0];
			OK.row(t) = (d_x_y < thresh) / 255.f;
			//cout<<OK.row(t)<<endl;
			Scalar sum_o = sum(OK.row(t));
			Score[t] = sum(OK.row(t)).val[0];
			//��¼��ý��������ֵ
			if(Score[t] > best)
			{
				best = Score[t];
				best_index = t;
			}
			else if(Score[t] == best)	//ģ��ƥ������һ��ʱ��ȡ�����С��
			{
				if(Total_err[t] < Total_err[best_index])
				{
					best = Score[t];
					best_index = t;
				}
			}
		}

		//��ѡ���ڵ㣬��DLT����
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

	//DLT����
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
	//���ɱ���RANSACѭ����Ӧ�Ĺ�һ���ĵ�Ӧ����
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