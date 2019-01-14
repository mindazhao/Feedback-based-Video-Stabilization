/*
2015.6.17����д��ɣ�ʹ��RANSAC+Nelder-Mead�㷨������ѵ�Ӧ���󣬲�ǿ����Ǹ��Բ���Ϊ0
2015.6.17����Ӳ������ƣ���б�����������������ó���һ����ֵ
2015.6.18���޸�RANSAC�㷨���֣�����ƥ��������ģ�ͣ��������������ֵ��������ȡ������
2015.6.31��6������Ϊ8����
2015.7.1��������������ǿ���ڼ������ʱ�򣬽�floatתΪfloat
2015.7.2��������Homography_Nelder_Mead�еľ���A��ÿ��ʹ��ǰ��Ҫ���㣬��Ϊ��+=������=������ÿ�������������Ϊ�ֲ�������ÿ��RASACѭ���ж����¶���
2015.7.20����RANSAC���֣�����ѡ����ԣ�������ѡ���4�����Ƿ񹹳�һ���ı��Σ���ֹ�������������ĵ㹲��
2015.9.17�������DLT+SVD���㵥Ӧ������㷨���������̫�󡣡���
2015.9.24��Homography_Nelder_Mead_with_outliers���̼߳���RANSAC
2015.10.30���ο�matlab�汾����random_shuffle�㷨����֮ǰ��ʹ���ĸ��������������������У�Ч���ַ��Ӻ��ˣ��ڵ������matlab�汾һ������
2015.12.15���ο�Goldstein��ÿ��RANSAC����������ÿ����ĵ���
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
//#include "Homography_from_trj.h"
#include"Homography_from_trj_backup_2016��7��4��д���İ汾�������˲�����.h"
using namespace cv;
using namespace std;

//��һ���㷨
void normalization(Mat &X, Mat &X_norm, Mat &norm_mat)
{
	int N = X.cols;
	//��������
	float x0=0, y0=0;//Point2f centroid;
	for(int i=0; i<N; i++)
	{
		x0 += ((float *)X.data)[i];
		y0 += ((float *)X.data)[i+N];
	}
	x0 /= N;
	y0 /= N;
	//���㵽���ĵľ���
	float mean_dist = 0;
	for(int i=0; i<N; i++)
		mean_dist += sqrt((((float *)X.data)[i]-x0)*(((float *)X.data)[i]-x0) + (((float *)X.data)[i+N]-y0)*(((float *)X.data)[i+N]-y0));
	mean_dist /=N;
	float sqrt_2 = sqrt(2.f);
	float mat_0_0 = sqrt_2/mean_dist;
	float mat_0_2 = -1*sqrt_2/mean_dist*x0;
	float mat_1_2 = -1*sqrt_2/mean_dist*y0;
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
float Residual(Mat H, Mat X1, Mat X2)
{
	int num = X1.cols;
	//�������
	Mat X2_ = H*X1;
	Mat X2_row_3 = Mat::zeros(3, num, CV_32F);
	X2_.row(2).copyTo(X2_row_3.row(0));
	X2_.row(2).copyTo(X2_row_3.row(1));
	X2_.row(2).copyTo(X2_row_3.row(2));
	X2_ /= X2_row_3;
	Mat dx = X2_.row(0) - X2.row(0);
	Mat dy = X2_.row(1) - X2.row(1);
	Mat d_x_y = (dx.mul(dx) + dy.mul(dy));
	//����ֵerr
	float err = sum(d_x_y).val[0];

	return err;
}
//����
void quick_sort(float *s, vector<Mat> &H_list, int l, int r)
{  
	if (l < r)
	{
		int i = l, j = r;
		float x = s[l];
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
	if(((float*)H.data)[1] > 0.04)
		((float*)H.data)[1] = 0.04;
	else if(((float*)H.data)[1] < -0.04)
		((float*)H.data)[1] = -0.04;
	if(((float*)H.data)[3] > 0.04)
		((float*)H.data)[3] = 0.04;
	else if(((float*)H.data)[3] < -0.04)
		((float*)H.data)[3] = -0.04;
	//���������޶�
	if(((float*)H.data)[0] > 1.04)
		((float*)H.data)[0] = 1.04;
	else if(((float*)H.data)[0] < 0.96)
		((float*)H.data)[0] = 0.96;
	if(((float*)H.data)[4] > 1.04)
		((float*)H.data)[4] = 1.04;
	else if(((float*)H.data)[4] < 0.96)
		((float*)H.data)[4] = 0.96;
}
//�������
bool check_coefficients(Mat &H)
{
	//������2��
	//��б�������������
	if((((float*)H.data)[1] > 0.1) || (((float*)H.data)[1] < -0.1) || (((float*)H.data)[3] > 0.1) || (((float*)H.data)[3] < -0.1) || \
		(((float*)H.data)[0] > 1.1) || (((float*)H.data)[0] < 0.9) || (((float*)H.data)[4] > 1.1) || (((float*)H.data)[4] < 0.9))
		return false;
	else
		return true;
}
void Nelder_Mead(Mat &H0, Mat &pt_bg_inlier, Mat &cor_smooth_inlier, int max_iter, float eps, Mat &H, bool show_best)
{
	const int Max_time = max_iter;
	//͸�ӱ任���󣬼���Ӧ�Ծ���ʱ����8������
	int var_num = 8;
	vector<Mat> vx(var_num+1);
	H0.copyTo(vx[0]);
	float vf[9] = {0, 0, 0, 0, 0, 0, 0};
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
				if((fabs(((float*)H0.data)[i*3+j])) < 0.00005)	//���̫С������Ϊ����һ����С���Ŷ�
					((float*)vx[i*3+j+1].data)[i*3+j] += 0.005;
				else
					((float*)vx[i*3+j+1].data)[i*3+j] /= 1.05;		//���򣬳���һ��ϵ��
				//�����޶�
				//constrain_coefficients(vx[i*3+j+1]);
				//cout<<vx[i*3+j+1]<<endl;
				vf[i*3+j+1] = Residual(vx[i*3+j+1], pt_bg_inlier, cor_smooth_inlier);	//�������Ӧ���
			}
		}
	}
	//����
	quick_sort(vf, vx, 0, var_num);

	float max_of_this = 0;
	float max_err = 0;
	while(max_iter>0)
	{
		for(int i=0; i<var_num+1; i++)
		{
			for(int j=i+1; j<var_num+1; j++)
			{
				Mat abs_err = abs(vx[i] - vx[j]);
				for(int k=0; k<3; k++)
				{
					if(((float*)abs_err.data)[k] > max_of_this)
						max_of_this = ((float*)abs_err.data)[k];
					if(((float*)abs_err.data)[k+3] > max_of_this)
						max_of_this = ((float*)abs_err.data)[k+3];
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
		if(max_err < eps && (vf[0] <= 2*pt_bg_inlier.cols || vf[0] <= 120))
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
		float fbest = vf[0];
		Mat soso = vx[var_num-1];
		float fsoso = vf[var_num-1];
		Mat worst = vx[var_num];
		float fworst = vf[var_num];
		Mat center = Mat::zeros(3, 3, CV_32F);
		for(int i=0; i<var_num; i++)
			center += vx[i];
		center /= var_num;
		Mat r = 2*center - worst;
		//�����޶�
		//constrain_coefficients(r);
		float fr = Residual(r, pt_bg_inlier, cor_smooth_inlier);
		if(fr < fbest)
		{
			//����õĽ�����ã�˵��������ȷ��������չ�㣬������������½�
			Mat e = 2*r - center;
			//�����޶�
			//constrain_coefficients(e);
			float fe = Residual(e, pt_bg_inlier, cor_smooth_inlier);
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
					float fc = Residual(c, pt_bg_inlier, cor_smooth_inlier);
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
					float fc = Residual(c, pt_bg_inlier, cor_smooth_inlier);
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
inline float cross_product(Point2f &A, Point2f &B, Point2f &C)
{
	return ((A.x-C.x)*(B.y-C.y)-(B.x-C.x)*(A.y-C.y));
}
//���㵥Ӧ�����滻findHomography����,max_iterΪNelder-Mead�㷨����������
void RANSAC_Foreground_Judgement(vector<Point2f> &pt_bg_cur, vector<Point2f> &Trj_cor_smooth, int max_iter, bool show_ransac, float scale, unsigned char *Foreground_times, int width, int height)
{
	//int64 st, et;
	//st = cvGetTickCount();
	int RANSAC_times = 100;
	float thresh_inlier = 80;//80/(scale*scale);
	int num = pt_bg_cur.size();
	//�����һ�����������ľ���
	Mat pt_bg = Mat::ones(3, num, CV_32F), cor_smooth = Mat::ones(3, num, CV_32F);
	//���ڲ����������
	vector<int>index_shuffle(num);
	vector<int>block_of_point(num);		//��¼ÿ���������ĸ���
	vector<int>block_num(6);						//��¼ÿ�����ڵĵ���
	int blocks = 0;										//��¼ʵ�ʼ������е�
	int row_n = 0, col_n = 0;
	for(int i=0; i<num; i++)
	{
		index_shuffle[i] = i;
		((float*)pt_bg.data)[i] = pt_bg_cur[i].x;
		((float*)pt_bg.data)[i+num] = pt_bg_cur[i].y;
		((float*)cor_smooth.data)[i] = Trj_cor_smooth[i].x;
		((float*)cor_smooth.data)[i+num] = Trj_cor_smooth[i].y;
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
	vector<float> Total_err(RANSAC_times);						//�������
	Mat thresh = thresh_inlier*Mat::ones(1, num, CV_32F);
	int best_index = -1;		//���ģ�͵�����ֵ
	int best = -1;				//Score�����ֵ
	//�Ѳ����ں��ʷ�Χ�ڵ�����ֵ������ѭ��һ��
	while(best == -1)
	{
		for(int t=0; t<RANSAC_times; t++)
		{
			//�����ȡ�ĸ��㣬�������A����
			vector<int> rand_set;
			vector<int>block_has(6);						//��¼��ÿ��RANSAC������ÿ�����Ѿ���ȥ�ĵ���������ÿ�������2����
			//����shuffle�㷨�����������
			random_shuffle(index_shuffle.begin(), index_shuffle.end());
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
			Mat A = Mat::zeros(12, 9, CV_32F);
			int j = 0;
			int k = rand_set[0];	//0 <= k < num
			Mat hat = (Mat_<float>(3,3) << 0, -1, ((float*)cor_smooth.data)[k+num], 1, 0, -1*((float*)cor_smooth.data)[k], -1*((float*)cor_smooth.data)[k+num], ((float*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			float x = ((float*)pt_bg.data)[k];
			float y = ((float*)pt_bg.data)[k+num];
			A.rowRange(j*3, j*3+3).colRange(0,3) += hat*x;
			A.rowRange(j*3, j*3+3).colRange(3,6) += hat*y;
			A.rowRange(j*3, j*3+3).colRange(6,9) += hat;

			++j;
			k = rand_set[1];	//0 <= k < num
			hat = (Mat_<float>(3,3) << 0, -1, ((float*)cor_smooth.data)[k+num], 1, 0, -1*((float*)cor_smooth.data)[k], -1*((float*)cor_smooth.data)[k+num], ((float*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			x = ((float*)pt_bg.data)[k];
			y = ((float*)pt_bg.data)[k+num];
			A.rowRange(j*3, j*3+3).colRange(0,3) += hat*x;
			A.rowRange(j*3, j*3+3).colRange(3,6) += hat*y;
			A.rowRange(j*3, j*3+3).colRange(6,9) += hat;

			++j;
			k = rand_set[2];	//0 <= k < num
			hat = (Mat_<float>(3,3) << 0, -1, ((float*)cor_smooth.data)[k+num], 1, 0, -1*((float*)cor_smooth.data)[k], -1*((float*)cor_smooth.data)[k+num], ((float*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			x = ((float*)pt_bg.data)[k];
			y = ((float*)pt_bg.data)[k+num];
			A.rowRange(j*3, j*3+3).colRange(0,3) += hat*x;
			A.rowRange(j*3, j*3+3).colRange(3,6) += hat*y;
			A.rowRange(j*3, j*3+3).colRange(6,9) += hat;

			++j;
			k = rand_set[3];	//0 <= k < num
			hat = (Mat_<float>(3,3) << 0, -1, ((float*)cor_smooth.data)[k+num], 1, 0, -1*((float*)cor_smooth.data)[k], -1*((float*)cor_smooth.data)[k+num], ((float*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			x = ((float*)pt_bg.data)[k];
			y = ((float*)pt_bg.data)[k+num];
			A.rowRange(j*3, j*3+3).colRange(0,3) += hat*x;
			A.rowRange(j*3, j*3+3).colRange(3,6) += hat*y;
			A.rowRange(j*3, j*3+3).colRange(6,9) += hat;

			//cout<<A<<endl;
			//SVD�ֽ�����VT����9��Ϊ��С����ֵ��Ӧ����������
			SVD thissvd(A,SVD::FULL_UV);
			Mat VT=thissvd.vt; 
			//cout<<VT<<endl;
			//���ɱ���RANSACѭ����Ӧ�Ĺ�һ���ĵ�Ӧ����
			H[t] = (Mat_<float>(3,3) <<((float*)VT.data)[72], ((float*)VT.data)[75], ((float*)VT.data)[78], ((float*)VT.data)[73], ((float*)VT.data)[76], ((float*)VT.data)[79], ((float*)VT.data)[74], ((float*)VT.data)[77], ((float*)VT.data)[80]);// / ((float*)VT.data)[80];
			//cout<<H[t]<<endl;
			H[t] /= ((float*)H[t].data)[8];
			//cout<<H[t]<<endl;

			//�������
			Mat X2_ = H[t]*pt_bg;
			//cout<<X2_<<endl;
			Mat X2_row_3 = Mat::zeros(3, num, CV_32F);
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
	//printf("RANSACѭ����100��ʱ��Ϊ: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.);
	if(show_ransac)
	{
		cout<<"��õ�ģ���ǵ�"<<best_index<<"��"<<endl;
		cout<<"ƥ������"<<best<<"��"<<endl;
		cout<<"��õ�ģ����:\n"<<H[best_index]<<endl;
	}
	Mat ok = OK.row(best_index);
	for (int i=0; i<num; i++)
		if (!ok.data[i])
			Foreground_times[i]++;

}

//���㵥Ӧ�����滻findHomography����,max_iterΪNelder-Mead�㷨����������
Mat Homography_Nelder_Mead_with_outliers(vector<Point2f> &pt_bg_cur, vector<Point2f> &Trj_cor_smooth, int max_iter, Mat& outliers, int height)
{
	//int64 st, et;
	//st = cvGetTickCount();
	int RANSAC_times = 200;
	float thresh_inlier = 30;//80/(scale*scale);
	int num = pt_bg_cur.size();
	//�����һ�����������ľ���
	Mat pt_bg = Mat::ones(3, num, CV_32F), cor_smooth = Mat::ones(3, num, CV_32F);
	//���ڲ����������
	vector<int>index_shuffle(num);
	for(int i=0; i<num; i++)
	{
		index_shuffle[i] = i;
		((float*)pt_bg.data)[i] = pt_bg_cur[i].x;
		((float*)pt_bg.data)[i+num] = pt_bg_cur[i].y;
		((float*)cor_smooth.data)[i] = Trj_cor_smooth[i].x;
		((float*)cor_smooth.data)[i+num] = Trj_cor_smooth[i].y;
	}

	//RANSAC�㷨�����100��ѭ��
	srand((unsigned)time(0));
	Mat OK = Mat::zeros(RANSAC_times, num, CV_8U);			//�õĽ����1��ʾ��������ģ��ƥ��úã�0Ϊ����
	vector<int> Score(RANSAC_times);								//�������÷֣��÷�Խ�߱�ʾģ��Խ��
	vector<Mat> H(RANSAC_times);									//ÿ�εĵ�Ӧ����
	vector<float> Total_err(RANSAC_times);						//�������
	Mat thresh = thresh_inlier*Mat::ones(1, num, CV_32F);
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
			random_shuffle(index_shuffle.begin(), index_shuffle.end());
			rand_set.push_back(index_shuffle[0]);
			rand_set.push_back(index_shuffle[1]);
			rand_set.push_back(index_shuffle[2]);
			rand_set.push_back(index_shuffle[3]);

			//Aһ��Ҫ�趨Ϊ�ֲ�������������Ϊ����ʹ��+=�������Ǹ�ֵ=��������
			Mat A = Mat::zeros(12, 9, CV_32F);
			int j = 0;
			int k = rand_set[0];	//0 <= k < num
			Mat hat = (Mat_<float>(3,3) << 0, -1, ((float*)cor_smooth.data)[k+num], 1, 0, -1*((float*)cor_smooth.data)[k], -1*((float*)cor_smooth.data)[k+num], ((float*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			float x = ((float*)pt_bg.data)[k];
			float y = ((float*)pt_bg.data)[k+num];
			A.rowRange(j*3, j*3+3).colRange(0,3) += hat*x;
			A.rowRange(j*3, j*3+3).colRange(3,6) += hat*y;
			A.rowRange(j*3, j*3+3).colRange(6,9) += hat;

			++j;
			k = rand_set[1];	//0 <= k < num
			hat = (Mat_<float>(3,3) << 0, -1, ((float*)cor_smooth.data)[k+num], 1, 0, -1*((float*)cor_smooth.data)[k], -1*((float*)cor_smooth.data)[k+num], ((float*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			x = ((float*)pt_bg.data)[k];
			y = ((float*)pt_bg.data)[k+num];
			A.rowRange(j*3, j*3+3).colRange(0,3) += hat*x;
			A.rowRange(j*3, j*3+3).colRange(3,6) += hat*y;
			A.rowRange(j*3, j*3+3).colRange(6,9) += hat;

			++j;
			k = rand_set[2];	//0 <= k < num
			hat = (Mat_<float>(3,3) << 0, -1, ((float*)cor_smooth.data)[k+num], 1, 0, -1*((float*)cor_smooth.data)[k], -1*((float*)cor_smooth.data)[k+num], ((float*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			x = ((float*)pt_bg.data)[k];
			y = ((float*)pt_bg.data)[k+num];
			A.rowRange(j*3, j*3+3).colRange(0,3) += hat*x;
			A.rowRange(j*3, j*3+3).colRange(3,6) += hat*y;
			A.rowRange(j*3, j*3+3).colRange(6,9) += hat;

			++j;
			k = rand_set[3];	//0 <= k < num
			hat = (Mat_<float>(3,3) << 0, -1, ((float*)cor_smooth.data)[k+num], 1, 0, -1*((float*)cor_smooth.data)[k], -1*((float*)cor_smooth.data)[k+num], ((float*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			x = ((float*)pt_bg.data)[k];
			y = ((float*)pt_bg.data)[k+num];
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
			H[t] = (Mat_<float>(3,3) <<((float*)VT.data)[72], ((float*)VT.data)[75], ((float*)VT.data)[78], ((float*)VT.data)[73], ((float*)VT.data)[76], ((float*)VT.data)[79], ((float*)VT.data)[74], ((float*)VT.data)[77], ((float*)VT.data)[80]);// / ((float*)VT.data)[80];
			//cout<<H[t]<<endl;
			H[t] /= ((float*)H[t].data)[8];
			//cout<<H[t]<<endl;

			//�������
			Mat X2_ = H[t]*pt_bg;
			//cout<<X2_<<endl;
			Mat X2_row_3 = Mat::zeros(3, num, CV_32F);
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
	//printf("RANSACѭ����100��ʱ��Ϊ: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.);
	//cout<<"ƥ������"<<best<<"��"<<endl;
	//outliers = OK.row(best_index).clone();
	//cout<<outliers<<endl;
	//��ȡ���ڵ�
	Mat pt_bg_inlier = Mat::zeros(3, best, CV_32F), cor_smooth_inlier = Mat::zeros(3, best, CV_32F);
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
	//Mat H_best = Mat::zeros(3, 3, CV_32F);
	//H0.row(2).col(0) = 0.f;
	//H0.row(2).col(1) = 0.f;
	Mat H_NM = Mat::zeros(3, 3, CV_32F);
	float eps = 0.05;
	//st = cvGetTickCount();
	bool show_ransac = false;
	Nelder_Mead(H0, pt_bg_inlier, cor_smooth_inlier, max_iter, eps, H_NM, show_ransac);
	//et = cvGetTickCount();
	//printf("NM����ʱ��: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.);
	return H_NM;
}
