/*
2015.6.1����һ����΢���õİ汾
2015.6.3���޳�ƥ������trainIdx�ظ��ĵ��
2015.6.6�������ʾƥ��������ʱ�Ĵ���
2015.6.8������detector������������������켣��
2015.6.10������˲������������⣬ĿǰЧ���Ѿ��ﵽ2015.4.8�����Ч��
2015.6.12�������˱������һ��㷨���Ѿ���ͨ
2015.6.13��Bundled Paths Optimization�㷨���������IplImage�������ڴ�й¶����
2015.6.17����findHomography���������Լ���д��Nelder-Mead�㷨
2015.6.30��ΪHomography_from_trj�����ഫ��һ��������������ď����������Σ�����Ч�����ã�
2015.7.2��6������Ϊ8��������NM�㷨����������Matlab�汾һ�£������������ӣ����㵥Ӧ���������float��Ϊfloat;
������Homography_from_trj�е�Homography_Nelder_Mead�о���A�Ĵ���ʹ��;	������findHomographyȫ������Homography_Nelder_Mead����,Ч���Ѿ��ӽ�matlab;
2015.7.3����������Ӧģ��
2015.7.4��Harris�ظ�����̫�٣���ƥ�亯�������⣿��ǰ֡����������һ֡������������Trj_desc���б�ǡ�����ƥ�䣬δƥ���ϵ���Trj_descƥ��
������һ�����⣺ÿ�ξ����켣��������֮�󣬻��޳������켣����ô֮ǰcur_desc��Trj_desc�еı��λ�þͻ������ı䣬Ҫ��ά������
�ͱ��뽫��Ӧ��Trajectoriesĩλ���Ǹ�cur_key�ı��λ���޸�һ�£�����Ĳ��øı�
2015.7.7�������һ���µĽ���������ƣ�if(continuity_num < 60) minHessian /= 1.05
2015.7.11���о�bundled paths optimization�㷨�� ���а������õ�ͨ�˲�����
2015.7.16���޸���һ��bug�������޳����ٳ��ֵĹ켣ʱ����������У�֮ǰ�����һ��<���޸�Ϊ>
2015.7.17�������������ⳤ�ȵ���Ƶ�ˣ�
�Ż������޳��������Σ���ʡʱ��
2015.7.17�����ա������������ϵͳ���ڶ��棩�����Ż����ɣ������Ż�
2015.7.19��������+FREAK���У���������ֻ����Harris��������
�Ѿ���֤�ˣ�4�����������С�����ֻ���޶�Harris�㷶Χ��(���ж�һ�㣬�����޶��ĸ��ǣ�)
���е�Ӧ��Լ����һ�飬Ҫ����֡�䵥Ӧ�������Ƶ��������ǿ�find��������
2015.8.22���ο�ICCV 2005һƬ���ģ���FREAK��ƥ�亯������SM�㷨����ԭʼƥ���㷨����k-����ƥ��������SM�㷨�ó���ȷƥ�䣬k=1�������k=3��ʱ��Ч��û�н���̫�࣬����ȥ�������ߴ�����˫�������
ʵ��֤����ƥ���ʴ�����
2015.8.25����֡ƥ������ȥ�ˣ�������������û������̫�࣬Ϊ�ˣ���ͼ��ֳ��ĸ����֣�������ȡ������������������ϳ�Ϊһ�������û�и���
2015.8.27��scale=1�����ԭ�ֱ��ʽ��д������û�и��ƣ����ԣ�����Թ��������Խ��Խ�٣��ܶ�㱻����Ϊǰ�����ˣ���Ȼ��һ��ʼ���������Ͳ��࣬��Ҳ��һ�����⣬����ǰһ����������أ�
������Ҫ��׼��һ�����⣺ǰ�����ж�����Ͳ������ڡ�
2015.9.7����ORB������ΪHarris+FREAK������������ʹ��Harris��������ȡ��FREAK�㷨��Ч�������ȶ�������Harris�����˵̫���ˣ���Ҫ���٣����Һ�������������̫���ˣ�����Ӧ������Ҫ����
2015.9.9����ʼ��ȫ���Ż�����ǰ�����ж�����
2015.9.12����ɽ��̫���У���Ȼ�����÷ֿ��˼·,�����飬������������������������к������
2015.9.22�����̼߳���ǰ�����ж�ģ�飬ʱ�併һ��
2015.9.24��SM�㷨�ж��̼߳���֡�ھ���
2015.11.1����ΪHarris+FREAK����������˳����³����
2015.12.15���һر����켣ʱ�򣬲��������������������ڹ켣�����������б����켣���������ԱȽ�
					�����켣������ǰ���켣�ͷ�
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
#include <sstream>
#include <vector>
#include <deque>
#include <set>
#include <algorithm>
#include <iostream>
//#include "CvxText.h"
#include <fstream>
#include <process.h>  
#include <windows.h>  
#include <regex>
//#include "Affine2D.hpp"
//#include  "normalization.h"
//#include "delaunay.h"
//#include "Homography_from_trj.h"
#include"Homography_from_trj_backup_2016��7��4��д���İ汾�������˲�����.h"
using namespace cv;
using namespace std;

#define MATCH_PAUSE 1
#define H_cnstrt_error_file 0
#define OUTFILE 0
#define SHOW_BG_POINT 0
#define SHOW_DELAUNAY 0
#define OUTPUT_TRJ_COR_MAT 0
#define SHOW_DELAUNAY_BUNDLED 0
#define USE_SURF 0

//�ؼ��Σ��ٽ�������������  
//CRITICAL_SECTION  g_csThreadParameter, g_csThreadCode;  
const int reserve_times = 120;	//���е�500֡��ʱ�򣬿�ʼ�޳���ǰ���һЩ�켣
vector<Point2f> pt_init_1, pt_init_2;
vector<KeyPoint>key_init;
Mat homo_init = Mat::zeros(3, 3, CV_32F);
// �켣�ṹ�壬����켣�����Ժ͹켣����
typedef class trajectory
{
public:
	unsigned int count;						//�켣���ֵĴ���
	int last_number;				//��һ�γ��ֵ�֡��
	unsigned int continuity;					//�������ֵ�֡��
	unsigned int foreground_times;		//����Ϊǰ���켣�Ĵ���
	unsigned int background_times;		//����Ϊ�����켣����
	unsigned int award;					//�����㽱��
	unsigned int penalization;				//ǰ����ͷ�
	deque<Point2f> trj_cor;			//�켣����
	//	Mat descriptors1;							//������������Ӧ����һ����ڴ���
public:
	trajectory(unsigned int ct = 1 , unsigned int ln = 1, unsigned int cnty = 1, unsigned int ft = 0, unsigned int award_time = 0, unsigned int penalization_time = 0, int n_time = reserve_times):trj_cor(n_time)
	{
		count = ct;
		last_number = ln;
		continuity = cnty;
		foreground_times = ft;
		award = award_time;
		penalization = penalization_time;
	};
}Trj;

//������ڼ��㵥Ӧ����Ĳ��л�
typedef class para_for_homo
{
public:
	vector<Point2f> pt_bg_cur;
	vector<Point2f> pt_smooth;
	bool start;
	unsigned char*foreground_times;
	int height;
	int width;
	para_for_homo(vector<Point2f>& pt_cur=pt_init_1, vector<Point2f>& pt_s=pt_init_2, bool actived=false, unsigned char*fg=NULL, int framewidth = 1280, int frameheight = 720)
	{
		pt_bg_cur = pt_cur;
		pt_smooth = pt_s;
		start = actived;
		foreground_times = fg;
		height = frameheight;
		width = framewidth;
	}
}PARA_FOR_HOMO;
//������ڼ���֡�ھ���Ĳ��л�
typedef class para_for_inframedist
{
public:
	vector<KeyPoint> cur_key;
	Mat Dist;
	bool up_left;
	para_for_inframedist(vector<KeyPoint> &keys_cur, Mat &h, bool actived)
	{
		cur_key = keys_cur;
		Dist = h;
		up_left = actived;
	}
}PARA_FOR_INFRAMEDIST;
//���������ȡHarris�ǵ�Ĳ��л�
typedef class para_for_harris
{
public:
	vector<KeyPoint> cur_key;
	Mat img;
	GoodFeaturesToTrackDetector cur_detector;
	int up;			//�ڼ���
	int colum;	//�ڼ���
	float delta_y;	//����ƫ��
	float delta_x;	//����ƫ��
	para_for_harris(vector<KeyPoint> &keys_cur, Mat I, GoodFeaturesToTrackDetector detector, float d_x, float d_y)
	{
		cur_key = keys_cur;
		img = I;
		cur_detector = detector;
		delta_x = d_x;
		delta_y = d_y;
	}
	para_for_harris()
	{

	}
}PARA_FOR_HARRIS;

//���õ�Ӧ������㺯���Ķ��̺߳�����Ҫ��������߳�
unsigned int __stdcall Calculate_Homography(void *para_pt)  
{  
	//LeaveCriticalSection(&g_csThreadParameter);//�뿪���߳���Źؼ����� 
	para_for_homo *these_pt = (para_for_homo*)para_pt;
	//���ڴ����߳���Ҫһ���Ŀ����ģ��������̲߳����ܵ�һʱ��ִ�е�����  
	//while(1)
	{
		while(!these_pt->start);
		if(these_pt->start)
		{
			Mat homo = Mat::zeros(3, 3, CV_32F);
			RANSAC_Foreground_Judgement(these_pt->pt_bg_cur, these_pt->pt_smooth, 100, false, 1.0, these_pt->foreground_times, these_pt->width, these_pt->height);
			these_pt->start = false;
		}
	}	
	return 0;  
}  
//���õ�Ӧ������㺯���Ķ��̺߳�����Ҫ��������߳�
unsigned int __stdcall Calculate_InFrameDistance(void *para_key)  
{  
	PARA_FOR_INFRAMEDIST *these_pt = (PARA_FOR_INFRAMEDIST*)para_key;
	//���ڴ����߳���Ҫһ���Ŀ����ģ��������̲߳����ܵ�һʱ��ִ�е�����  
	int n_node = these_pt->cur_key.size();
	int n_node_1_2 = n_node / 2;
	vector<KeyPoint>cur_key = these_pt->cur_key;
	Mat Dij_1 = these_pt->Dist;//�����߳��л�ֱ�ֵ����ͬ�ľֲ�������������ָ��ͬһ��ȫ�ֱ���Dij_1���᲻��������⣿������������
	float temp = 0.f;
	if(these_pt->up_left)
	{
		for(int i=0; i<n_node_1_2; i++)
		{
			for(int j=i+1; j<n_node-i; j++)	//�Ľ�����Ϊ����һ���Ը�����ֵ������ѭ������Ҳ���Լ��룬֮ǰ������Ȼ�Ǵ�j=0��ʼ��
			{	
				//temp = sqrt((cur_key[i].pt.x-cur_key[j].pt.x)*(cur_key[i].pt.x-cur_key[j].pt.x) + (cur_key[i].pt.y-cur_key[j].pt.y)*(cur_key[i].pt.y-cur_key[j].pt.y));
				temp = fabs(cur_key[i].pt.x-cur_key[j].pt.x) + fabs(cur_key[i].pt.y-cur_key[j].pt.y);
				((float*)Dij_1.data)[i*n_node + j] = temp;
				((float*)Dij_1.data)[j*n_node + i] = temp;
			}
		}
	}	
	else
	{
		for(int j=n_node_1_2; j<n_node; j++)
		{
			for(int i=n_node-j; i<=j; i++)	//�Ľ�����Ϊ����һ���Ը�����ֵ������ѭ������Ҳ���Լ��룬֮ǰ������Ȼ�Ǵ�j=0��ʼ��
			{	
				//temp = sqrt((cur_key[i].pt.x-cur_key[j].pt.x)*(cur_key[i].pt.x-cur_key[j].pt.x) + (cur_key[i].pt.y-cur_key[j].pt.y)*(cur_key[i].pt.y-cur_key[j].pt.y));
				temp = fabs(cur_key[i].pt.x-cur_key[j].pt.x) + fabs(cur_key[i].pt.y-cur_key[j].pt.y);
				((float*)Dij_1.data)[i*n_node + j] = temp;
				((float*)Dij_1.data)[j*n_node + i] = temp;
			}
		}
	}

	return 0;  
}  
//���õ�Ӧ������㺯���Ķ��̺߳�����Ҫ��������߳�
unsigned int __stdcall Detect_Harris(void *para_pt)  
{  
	//LeaveCriticalSection(&g_csThreadParameter);//�뿪���߳���Źؼ����� 
	para_for_harris *cur_detect = (para_for_harris*)para_pt;
	cur_detect->cur_detector.detect(cur_detect->img, cur_detect->cur_key);
	//�������긴ԭ
	int size = cur_detect->cur_key.size();
	float d_x = cur_detect->delta_x;
	float d_y = cur_detect->delta_y;
	if (d_x>0 && d_y>0)
	{
		for (int i=0; i<size; i++)
		{
			cur_detect->cur_key[i].pt.x += d_x;
			cur_detect->cur_key[i].pt.y += d_y;
		}
	}
	else if (d_x>0)
	{
		for (int i=0; i<size; i++)
			cur_detect->cur_key[i].pt.x += d_x;
	}
	else if (d_y>0)
	{
		for (int i=0; i<size; i++)
			cur_detect->cur_key[i].pt.y += d_y;
	}
	return 0;  
}  
//�Ƚ��㷨
bool compare1(const DMatch &d1,const  DMatch &d2)
{
	return d1.trainIdx < d2.trainIdx;
}
//����ƽ��ֵ
float mean(const deque<int> trj_num)
{
	float sum = 0;
	int trjs = trj_num.size();
	for(int i=0; i<trjs; i++)
		sum += trj_num[i];
	sum /= trjs;
	return sum;
}
//�����׼�����Ϊ(N-1)
float std_val(const deque<int> trj_num, float the_mean)
{
	float std_var = 0;
	int trjs = trj_num.size();
	for(int i=0; i<trjs; i++)
		std_var += (trj_num[i] - the_mean) * (trj_num[i] - the_mean);
	std_var /= (trjs - 1);
	return sqrt(std_var);
}
// ���㺺������
unsigned int hamdist2(unsigned char* a, unsigned char* b, size_t size) 
{ 
	HammingLUT lut;  
	unsigned int result;
	result = lut((a), (b), size); 
	return result; 
}  
// �ú���ֻ�õ�Trj_keys.size()��cur_key.size()������ֻ�õ����ߵĳ���!
// Ϊ�˸���Ч��Ӧ�������һ��vector������е�������
void naive_nn_search(Mat& descp1, Mat& descp2, vector<DMatch>& matches) 
{
	//vector<unsigned int> matched_cur, matched_Trj;	//����ƥ���ϵĵ�����
	int cur_key_size = descp2.rows;
	int Trj_keys_size = descp1.rows;
	for( int i = 0; i < cur_key_size; i++)
	{
		unsigned int min_dist = INT_MAX; 
		unsigned int sec_dist = INT_MAX; 
		int min_idx = -1, sec_idx = -1;
		unsigned char* query_feat = descp2.ptr(i); 
		for( int j = 0; j < Trj_keys_size; j++)
		{
			unsigned char* train_feat = descp1.ptr(j);
			unsigned int dist =  hamdist2(query_feat, train_feat, 64); //������һ�����صĴ��������64λ��FREAK���ӣ�
			//��̾���
			if(dist < min_dist)
			{ 
				sec_dist = min_dist; 
				sec_idx = min_idx;
				min_dist = dist; 
				min_idx = j; 
			}
			//�ζ̾���
			else if(dist < sec_dist)
			{ 
				sec_dist = dist; 
				sec_idx = j; 
			} 
		}   
		if(min_dist <=50 && min_dist <= 0.8*sec_dist)//min_dist <= (unsigned int)(sec_dist * 0.7) && min_dist <=100
		{
			//�����������������ظ���ƥ��ԣ�
			bool repeat = false;
			if(matches.size()>0)
			{
				for(int k=0; k<matches.size(); k++)
				{
					if(min_idx == matches.at(k).trainIdx)
						repeat = true;
				}
				if(!repeat)
					matches.push_back(DMatch(i, min_idx, 0, (float)min_dist));
			}
			else matches.push_back(DMatch(i, min_idx, 0, (float)min_dist)); 
		}
	} 
}  
// �ú���ֻ�õ�Trj_keys.size()��cur_key.size()������ֻ�õ����ߵĳ���!
// Ϊ�˸���Ч��Ӧ�������һ��vector������е�������
void naive_nn_search2(vector<KeyPoint>& Trj_keys, Mat& descp1, vector<KeyPoint>& cur_key, Mat& descp2, vector<vector<DMatch>>& matches, const int max_shaky_dist, int k) 
{
	//vector<unsigned int> matched_cur, matched_Trj;	//����ƥ���ϵĵ�����
	int cur_key_size = cur_key.size();
	int Trj_keys_size = Trj_keys.size();
	for( int i = 0; i < cur_key_size; i++)
	{
		unsigned int min_dist = INT_MAX; 
		unsigned int sec_dist = INT_MAX; 
		unsigned int thr_dist = INT_MAX; 
		int min_idx = -1, sec_idx = -1, thr_idx = -1;
		unsigned char* query_feat = descp2.ptr(i); 
		float cur_key_x = cur_key[i].pt.x;
		float cur_key_y = cur_key[i].pt.y;
		for( int j = 0; j < Trj_keys_size; j++)
		{
			unsigned char* train_feat = descp1.ptr(j);
			unsigned int dist =  hamdist2(query_feat, train_feat, 64); //������һ�����صĴ��������64λ��FREAK���ӣ�
			float Trj_key_x = Trj_keys[j].pt.x;
			float Trj_key_y = Trj_keys[j].pt.y;
			//ƥ��������������
			if((cur_key_x - Trj_key_x)*(cur_key_x - Trj_key_x) + (cur_key_y - Trj_key_y)*(cur_key_y - Trj_key_y) < max_shaky_dist)
			{
				//��̾���
				if(dist < min_dist)
				{ 
					thr_dist = sec_dist;
					thr_idx = sec_idx;
					sec_dist = min_dist; 
					sec_idx = min_idx;
					min_dist = dist; 
					min_idx = j; 
				}
				//�ζ̾���
				else if(dist < sec_dist)
				{ 
					thr_dist = sec_dist;
					thr_idx = sec_idx;
					sec_dist = dist; sec_idx = j; 
				} 
				//�ζ̾���
				else if(dist < thr_dist)
				{ 
					thr_dist = dist; thr_idx = j; 
				} 
			}
		}   
		if(min_dist <=125)
		{
			matches[i].push_back(DMatch(i, min_idx, 0, (float)min_dist));
			if(k>1)
				matches[i].push_back(DMatch(i, sec_idx, 0, (float)sec_dist));
			if(k>2)
				matches[i].push_back(DMatch(i, thr_idx, 0, (float)thr_dist));
		}
		else
			matches[i].push_back(DMatch(i, -1, 0, (float)min_dist));
	}
}
//����
void quick_sort(Mat v, vector<Point> &L, int l, int r)
{  
	if (l < r)
	{
		int i = l, j = r;
		float x = ((float*)v.data)[l];
		Point temp = L[l];
		while (i < j)
		{
			while(i < j && ((float*)v.data)[j] <= x) // ���������ҵ�һ��С��x����  
				j--;
			if(i < j)
			{
				((float*)v.data)[i++] = ((float*)v.data)[j];
				L[i-1] = L[j];
			}

			while(i < j && ((float*)v.data)[i] > x) // ���������ҵ�һ�����ڵ���x����  
				i++;
			if(i < j)
			{
				((float*)v.data)[j--] = ((float*)v.data)[i];
				L[j+1] = L[i];
			}
		}
		((float*)v.data)[i] = x;
		L[i] = temp;
		quick_sort(v, L, l, i - 1); // �ݹ����   
		quick_sort(v, L, i + 1, r);
	}
}


//my_spectral_matching�������Լ���д��matlab������д
void my_spectral_matching(vector<KeyPoint> &cur_key, vector<KeyPoint> &last_key, vector<vector<DMatch>> &matches, int &k, vector<DMatch> &X_best)
{
	int64 st, et;
	//st = cvGetTickCount();
	int n_node = cur_key.size();
	int n_label = last_key.size();
	vector<int> start_ind_for_node(n_node);	//ÿ��ӵ����Чƥ��Եĵ�ǰ��������L�е���ʼ����ֵ
	int n_matches = 0;	//��Чƥ��Ը���
	vector<Point> L;	//���еĺ�ѡƥ���
	for (int i=0; i<n_node; i++)
	{
		if(matches[i][0].trainIdx != -1)
			start_ind_for_node[i] = n_matches;
		else
		{
			start_ind_for_node[i] = -1;
			continue;
		}
		for (int j=0; j<k; j++)
		{
			if (matches[i][j].trainIdx != -1)
			{
				L.push_back(Point(i, matches[i][j].trainIdx));
				n_matches ++;
			}
			//else
			//	cout<<i<<"\t"<<j<<endl;
		}
	}
	//et = cvGetTickCount();
	//printf("����L����ʱ��: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 
	////%% ����M�ĶԽ���Ԫ��
	//st = cvGetTickCount();
	Mat M = Mat::zeros(n_matches, n_matches, CV_32F);
	int n_cur = 0;
	float Ham_max = 256.0;
	int tmp = 0;
	for (int i=0; i<n_node-1; i++)
	{
		if(start_ind_for_node[i] != -1)
		{
			n_cur = start_ind_for_node[i+1] - start_ind_for_node[i];
			for(int j=0; j<n_cur; j++)
			{
				tmp = start_ind_for_node[i] + j;
				((float*)M.data)[tmp*n_matches + tmp] = matches[i][j].distance/Ham_max;
			}
		}
	}
	if(start_ind_for_node[n_node-1] != -1)
	{
		n_cur = n_matches - start_ind_for_node[n_node-1];
		for(int j=0; j<n_cur; j++)
		{
			tmp = start_ind_for_node[n_node-1] + j;
			((float*)M.data)[tmp*n_matches + tmp] = matches[n_node-1][j].distance/Ham_max;
		}
	}
	//et = cvGetTickCount();
	//printf("����M����Խ���Ԫ��ʱ��: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 

	////�ȼ���ÿ֡��ÿ��i,j֮����������
	//st = cvGetTickCount();
	Mat Dij_1 = Mat::zeros(n_node, n_node, CV_32F);
	Mat Dij_2 = Mat::zeros(n_label, n_label, CV_32F);
	float temp = 0;

	//���̰߳汾
	//st = cvGetTickCount();
	const int THREAD_NUM = 4;
	HANDLE handle[THREAD_NUM]; 
	PARA_FOR_INFRAMEDIST pthread_array_1 = PARA_FOR_INFRAMEDIST(cur_key, Dij_1, true);
	PARA_FOR_INFRAMEDIST pthread_array_2 = PARA_FOR_INFRAMEDIST(cur_key, Dij_1, false);
	PARA_FOR_INFRAMEDIST pthread_array_3 = PARA_FOR_INFRAMEDIST(last_key, Dij_2, true);
	PARA_FOR_INFRAMEDIST pthread_array_4 = PARA_FOR_INFRAMEDIST(last_key, Dij_2, false);
	handle[0] = (HANDLE)_beginthreadex(NULL, 0, Calculate_InFrameDistance, &pthread_array_1, 0, NULL); 
	handle[1] = (HANDLE)_beginthreadex(NULL, 0, Calculate_InFrameDistance, &pthread_array_2, 0, NULL); 
	handle[2] = (HANDLE)_beginthreadex(NULL, 0, Calculate_InFrameDistance, &pthread_array_3, 0, NULL); 
	handle[3] = (HANDLE)_beginthreadex(NULL, 0, Calculate_InFrameDistance, &pthread_array_4, 0, NULL); 

	WaitForMultipleObjects(THREAD_NUM, handle, TRUE, INFINITE);//INFINITE);//����ȴ�20ms
	for (int i=0; i<4; i++)
		CloseHandle(handle[i]);
	//et = cvGetTickCount();
	//printf("����֡�ھ���ʱ��: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 

	//����M����ķǶԽ���Ԫ��
	//st = cvGetTickCount();
	float sigma = 4;
	float sigma_d_3 = sigma*3;
	float delta_2 = 2*sigma*sigma;
	float tmp1=0, tmp2=0, tmp3=0;
	for(int i=1; i<n_matches; i++)
	{
		for(int j=i+1; j<n_matches; j++)
		{	
			tmp1 = ((float*)Dij_1.data)[(L[i].x)*n_node+L[j].x];
			tmp2 = ((float*)Dij_2.data)[(L[i].y)*n_label+L[j].y];
			temp = tmp1 - tmp2;
			if(fabs(temp) < sigma_d_3)
			{
				tmp3 = 4.5-(temp*temp)/delta_2;
				((float*)M.data)[i*n_matches+j] = tmp3;
				((float*)M.data)[j*n_matches+i] = tmp3;
			}
		}
	}
	//et = cvGetTickCount();
	//printf("����M����ǶԽ���Ԫ��ʱ��: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 

	//%% spectral matching�㷨
	//st = cvGetTickCount();
	Mat v = Mat::ones(n_matches, 1, CV_32F);
	//float x = norm(v);
	v = v/norm(v);
	int iterClimb = 20;//֮ǰȡ30����ʱ̫����ȡ20Ӧ��Ҳ���ԣ���������Ҳ��������

	// �ݷ������������ֵ�����䣩��Ӧ����������
	for(int i = 0; i<iterClimb; i++)
	{
		v = M*v;
		v = v/norm(v);
	}
	//et = cvGetTickCount();
	//printf("�ݷ���������������ʱ��: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 
	//̰�Ĳ����������ƥ��
	//st = cvGetTickCount();
	//vector<DMatch> X_best;
	quick_sort(v, L, 0, n_matches-1);
	//et = cvGetTickCount();
	//printf("��������ʱ��: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 
	float max_v = 0;
	//DMatch best_match;
	//st = cvGetTickCount();
	bool *conflict = new bool[n_matches];	//���ÿ��ƥ����Ƿ��뵱ǰ��õ�ƥ��Գ�ͻ
	for (int i=0; i<n_matches; i++)
		conflict[i] = false;
	int left_matches = n_matches;
	float dist = 10.0;
	while(left_matches)
	{
		int i=0;
		while(conflict[i]) i++;	//�ҵ���һ��δ��ͻ�����ֵ��
		max_v = ((float*)v.data)[i];
		DMatch best_match = DMatch(L[i].x, L[i].y, 0, float(dist));
		X_best.push_back(best_match);
		//�ҳ�������best_match��ͻ��ƥ��ԣ��޳�֮
		for (int j=0; j<n_matches; j++)
		{
			if((L[j].x == best_match.queryIdx || L[j].y == best_match.trainIdx ) && !conflict[j])
			{
				conflict[j] = true;
				left_matches--;
			}
		}
	}
	delete []conflict;
	//et = cvGetTickCount();
	//printf("̰�Ĳ���ʱ��: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 
}
/*
pts��Ҫ�ʷֵ�ɢ�㼯,in
img,�ʷֵĻ���,in
tri,�洢������ʾ����任������,out
*/
// used for doing delaunay trianglation with opencv function
//�ú���������ֹ����ػ�����ȥ���������εĶ���
bool isGoodTri( Vec3i &v, vector<Vec3i> & tri ) 
{
	int a = v[0], b = v[1], c = v[2];
	v[0] = min(a,min(b,c));//v[0]�ҵ��������Ⱥ�˳��0....N-1��NΪ��ĸ���������Сֵ
	v[2] = max(a,max(b,c));//v[2]�洢���ֵ.
	v[1] = a+b+c-v[0]-v[2];//v[1]Ϊ�м�ֵ
	if (v[0] == -1) return false;

	vector<Vec3i>::iterator iter = tri.begin();//��ʼʱΪ��
	for(;iter!=tri.end();iter++)
	{
		Vec3i &check = *iter;//�����ǰ��ѹ��ĺʹ洢���ظ��ˣ���ֹͣ����false��
		if (check[0]==v[0] &&
			check[1]==v[1] &&
			check[2]==v[2])
		{
			break;
		}
	}
	if (iter == tri.end())
	{
		tri.push_back(v);
		return true;
	}
	return false;
}
//���ֲ��ң���Ϊforeground_index�Ǵ�С�����ź����
int binary_search(vector<int>a, int goal)
{
	if(a.size()==0)
	{
		cout<<"����vector�ǿյģ����ƨ��"<<endl;
		return -1;
	}
	int low = 0;
	int high = a.size() - 1;
	while(low <= high)
	{
		int middle = (low + high)/2;
		if(a[middle] == goal)
			return middle;
		//������
		else if(a[middle] > goal)
			high = middle - 1;
		//���Ұ��
		else
			low = middle + 1;
	}
	//û�ҵ�
	return -1;
}
vector<float> MyGauss( int _sigma )
{
	int width = 2 * _sigma+1;
	if ( width<1 )
	{
		width = 1;
	}


	/// �趨��˹�˲������
	int len = width;

	/// ��˹����G
	vector<float> GassMat;

	int cent = len/2;
	float summ = 0;
	for ( int i=0; i<len; i++ )
	{
		int radius = ( cent - i ) * ( cent - i );
		GassMat.push_back(exp( -( (float)radius ) /( 2 * _sigma * _sigma ) ));

		summ += GassMat[i] ;
	}
	for ( int i=0; i<len; i++ )
		GassMat[i] /= (summ+0.001);

	return GassMat ;
}
//FAST��ֵ�ȽϺ���
bool fast_thresh_comp(const KeyPoint &corner1, const KeyPoint &corner2)
{
	return corner1.response > corner2.response;
}
//������������������FAST��������㷨
void MyFAST(Mat& image, vector<KeyPoint>& corners, int maxCorners, double qualityLevel, double minDistance, Mat & mask, bool have_mask)
{
	vector<KeyPoint>FAST_corners;
	//����FAST���
	FAST(image, corners, qualityLevel);

	//��Mask����������һ�������
	Mat temp_mask(image.size(), CV_8U, Scalar(0));
	int num_corners = corners.size();
	int width = image.cols;
	int height = image.rows;
	if (have_mask)
	{
		for (int i=0; i<num_corners; i++)
		{
			if (((unsigned char*)mask.data)[cvRound(corners[i].pt.x+corners[i].pt.y*width)])
			{
				((unsigned char*)temp_mask.data)[cvRound(corners[i].pt.x+corners[i].pt.y*width)] = 255;
				FAST_corners.push_back(corners[i]);
			}
		}
	}
	else
	{
		FAST_corners = corners;
		for (int i=0; i<num_corners; i++)
		{
			((unsigned char*)temp_mask.data)[cvRound(corners[i].pt.x+corners[i].pt.y*width)] = 255;
		}
	}

	num_corners = FAST_corners.size();
	corners.clear();
	//̰���㷨��ɸѡ
	sort(FAST_corners, fast_thresh_comp);
	for (int i=0; i<num_corners && corners.size()<maxCorners; i++)
	{
		if (((unsigned char*)temp_mask.data)[cvRound(FAST_corners[i].pt.x+FAST_corners[i].pt.y*width)])
		{
			int rw_right = min(cvRound(FAST_corners[i].pt.x)+minDistance, width);
			int rw_left = max(cvRound(FAST_corners[i].pt.x)-minDistance, 0);
			int cl_up = max(cvRound(FAST_corners[i].pt.y)-minDistance, 0);
			int cl_down = min(cvRound(FAST_corners[i].pt.y)+minDistance, height);

			//cout<<temp_mask.rowRange(rw_left, rw_right).colRange(cl_up, cl_down)<<endl;
			temp_mask.colRange(rw_left, rw_right).rowRange(cl_up, cl_down) = Mat::zeros(rw_right-rw_left, cl_down-cl_up, CV_8U);
			corners.push_back(FAST_corners[i]);
		}
	}
	//cout<<corners.size()<<endl;
}
//argv��ʽ�������� ��Ƶ�ļ��� ��ֱ�����������������y���� ��ֱ�������������յ���y���� ֡��
int main(int argc, char* argv[])
{
	//��������������
	if (argc < 3)
	{
		cout<<"���������ʽ��"<<endl;
		cout<<"������ ��Ƶ�ļ��� ����ļ���"<<endl;
		return -1;
	}
	//����Ƶ�ļ�
	string filename = string(argv[1]);
	int the_start_num = filename.find_last_of("/");
	string the_name = filename.substr(the_start_num+1);//, filename.length()-4);
	the_name = the_name.substr(0, the_name.length()-4);

	//char* openfile="E://��Ƶȥ��//������Ƶ//18_failure_train.avi";//bg_motion_2, on_road_3��on_road_4��example4_car_input��8ԭ��Ƶ
	char *openfile = &filename[0];
	CvCapture* pCapture=cvCreateFileCapture(openfile);
	if(pCapture==NULL)
	{
		cout<<"video file open error!"<<endl;
		return -1;
	}
	string outfilename = string(argv[2]) + string("/proof_") + the_name + string("_Version.avi");
	char* outfile=&outfilename[0];
	//��ȡ��Ƶ�����Ϣ��֡�ʺʹ�С
	float fps=cvGetCaptureProperty(pCapture,CV_CAP_PROP_FPS);
	int numframes = cvGetCaptureProperty(pCapture, CV_CAP_PROP_FRAME_COUNT);
	cout<<"numframes: "<<numframes<<endl;
	IplImage* frame_ref=NULL, *frame_cur=NULL, *gray, *dst, *stitch_image;
	int null_index = 0;
	while(null_index++<7)
		frame_ref = cvQueryFrame(pCapture);
	//frame_ref = cvQueryFrame(pCapture);
	//frame_ref = cvQueryFrame(pCapture);
	//frame_ref = cvQueryFrame(pCapture);
	//frame_ref = cvQueryFrame(pCapture);
	//frame_ref = cvQueryFrame(pCapture);
	//frame_ref = cvQueryFrame(pCapture);
	//frame_ref = cvQueryFrame(pCapture);
	//frame_ref = cvQueryFrame(pCapture);
	//frame_ref = cvQueryFrame(pCapture);
	//frame_ref = cvQueryFrame(pCapture);
	//frame_ref = cvQueryFrame(pCapture);
	//frame_ref = cvQueryFrame(pCapture);
	//frame_ref = cvQueryFrame(pCapture);
	//��Ƶ֡����ǰ�ߴ磬���ű���
	int width = frame_ref->width;
	int height = frame_ref->height;

	
	//���Ϊ�бߺ����Ƶ���������Ҹ��б�40������
	int howtocrop_width = 0;//(float)width*60/1280;
	int howtocrop_height = 0;//(float)height*40/540;
	cout<<"howtocrop: "<<howtocrop_width<<endl;
	CvSize size=cvSize((int)cvGetCaptureProperty(pCapture,CV_CAP_PROP_FRAME_WIDTH)-howtocrop_width*2,
		(int)cvGetCaptureProperty(pCapture,CV_CAP_PROP_FRAME_HEIGHT)-howtocrop_height*2);

	//���������Ƶ�ļ�
	CvVideoWriter* Save_result=NULL;
	Save_result=cvCreateVideoWriter(outfile,CV_FOURCC('X','V','I','D'),fps,size,1);

	float scale = 1;

	int cropped_start = 0;//96;
	int cropped_end = height;//640

	const float crop_width = width/scale;
	const float crop_height = (cropped_end-cropped_start)/scale;
	const float height_1_2 = crop_height/2;
	const float width_1_4 = crop_width/4, width_2_4 = crop_width/2, width_3_4 = 3*crop_width/4;
	cout<<"����"<<scale<<"��֮����"<<endl;
	CvSize after_size = cvSize(width/scale, height/scale);
	gray = cvCreateImage(cvGetSize(frame_ref), frame_ref->depth, 1);
	dst = cvCreateImage(after_size, frame_ref->depth, 1); 
	// �ο�֡תΪ�Ҷ�ͼ
	cvCvtColor(frame_ref, gray, CV_BGR2GRAY);
	cvResize(gray, dst);
	Mat object(dst);//�������������ݣ�ֻ��������ͷ
	Mat crop_ref = object(Range(cropped_start/scale, cropped_end/scale), Range(0, width/scale));
	//ʹ�ø���Ȥ�������������㣬�����Ǽ��е�
	Mat mask(crop_ref.size(), CV_8U, Scalar(255));
	//rectangle(mask, Point(1050/scale, 31/scale), Point(1280/scale, 81/scale), Scalar(0), -1, CV_8U);
	rectangle(mask, Point(40/scale, 35/scale), Point(774/scale, 96/scale), Scalar(0), -1, CV_8U);
	rectangle(mask, Point(922/scale, 644/scale), Point(1178/scale, 684/scale), Scalar(0), -1, CV_8U);
	//****************************************************************************************************//
	//******************************************�ؼ������ݽṹ���*********************************************//
	vector<KeyPoint> Trj_keys, ref_key_1, ref_key_2, ref_key_3, ref_key_4, ref_key_5, ref_key_6;//, ref_key_7, ref_key_8, ref_key_9;				//��ŵ�һ֡����ǰ֡�Ĺؼ���
	vector<KeyPoint> cur_key, cur_key_1, cur_key_2, cur_key_3, cur_key_4, cur_key_5, cur_key_6;//, cur_key_7, cur_key_8, cur_key_9;
	Mat Trj_desc;//, Trj_desc_1, Trj_desc_2, Trj_desc_3, Trj_desc_4, Trj_desc_5, Trj_desc_6, Trj_desc_7, Trj_desc_8, Trj_desc_9;						//**************��Ҫ������������й켣������������ǰ������******************//
	Mat cur_descriptor;//, cur_desc_1, cur_desc_2, cur_desc_3, cur_desc_4, cur_desc_5, cur_desc_6, cur_desc_7, cur_desc_8, cur_desc_9;

	int64 st, et;
	//const int reserve_times = 500;	//���е�500֡��ʱ�򣬿�ʼ�޳���ǰ���һЩ�켣
	unsigned int num_of_corners = 3000;						//Surf��ֵ�����ͣ������Surf������
	unsigned int last_size = 0;
	const unsigned char gsw = 5;	//�˲����ڴ�С8
	const unsigned char gsw_2 = gsw*2;
	const int max_dist = 1800;
	float max_shaky_dist = max_dist;							//һ����˵����󶶶������ᳬ��30*30��cur_dist<30*30+30*30=1800
	int penalization_relax = 0;							//���ڻָ��ͷ����Ƶ���ʼ״̬
	int penalization_thresh = 1;							//�ͷ���ֵ
	const unsigned int good_match_tolerance = INT_MAX;				//��������ƥ���������
	const unsigned char max_H_cnstrt_tolerance = 25/scale;		//����Homography Constraint�б�ǰ����ʱ����ֵ
	//the following gauss_kernel has been verified so that sum(gauss_kernel)=1;
	//const float gauss_kernel[31] = {0.0230896376799231, 0.0246266299576938, 0.0261494557073281, 0.0276433151385955, 0.0290929257343107, 0.0304827731525912, 0.0317973811174797, 0.0330215939430714, 0.0341408646181710, 0.0351415408820063, 0.0360111414768349, 0.0367386147905601, 0.0373145724104322, 0.0377314906939604, 0.0379838743092176, 0.0380683767756483, 0.0379838743092176, 0.0377314906939604, 0.0373145724104322, 0.0367386147905601, 0.0360111414768349, 0.0351415408820063, 0.0341408646181710, 0.0330215939430714, 0.0317973811174797, 0.0304827731525912, 0.0290929257343107, 0.0276433151385955, 0.0261494557073281, 0.0246266299576938, 0.0230896376799231};
	//const float gauss_kernel[23] = {0.0312305401815636, 0.0340616911991542, 0.0368437399227251, 0.0395250108698724, 0.0420524281533089, 0.0443732201389741, 0.0464367273060946, 0.0481962282319788, 0.0496106925141895, 0.0506463691183050, 0.0512781244174344, 0.0514904558927990, 0.0512781244174344, 0.0506463691183050, 0.0496106925141895, 0.0481962282319788, 0.0464367273060946, 0.0443732201389741, 0.0420524281533089, 0.0395250108698724, 0.0368437399227251, 0.0340616911991542, 0.0312305401815636};
	//const float gauss_kernel[21] = {0.0342500349869951, 0.0376633542618280, 0.0410047365739600, 0.0441983556642708, 0.0471666741100688, 0.0498335068564508, 0.0521272365287996, 0.0539839935144846, 0.0553506048166937, 0.0561871220838075, 0.0564687612052823, 0.0561871220838075, 0.0553506048166937, 0.0539839935144846, 0.0521272365287996, 0.0498335068564508, 0.0471666741100688, 0.0441983556642708, 0.0410047365739600, 0.0376633542618280, 0.0342500349869951};
	//const float gauss_kernel[19] = {0.0379163821475835, 0.0421115284003285, 0.0461969662123557, 0.0500569342747520, 0.0535739143843883, 0.0566344712911464, 0.0591352805416095, 0.0609889016410867, 0.0621288482293915, 0.0625135457547156, 0.0621288482293915, 0.0609889016410867, 0.0591352805416095, 0.0566344712911464, 0.0535739143843883, 0.0500569342747520, 0.0461969662123557, 0.0421115284003285, 0.0379163821475835};
	//const float gauss_kernel[17] = {0.0424626034730037, 0.0477419883829079, 0.0528455638502515, 0.0575878305549686, 0.0617827234955574, 0.0652555611088371, 0.0678550471429178, 0.0694641832139329, 0.0700089975552463, 0.0694641832139329, 0.0678550471429178, 0.0652555611088371, 0.0617827234955574, 0.0575878305549686, 0.0528455638502515, 0.0477419883829079, 0.0424626034730037};
	//const float gauss_kernel[15] = {0.0482490884886937, 0.0550934038915087, 0.0616377739061904, 0.0675664548636212, 0.0725691718501518, 0.0763677593943165, 0.0787416983639167, 0.0795492984832020, 0.0787416983639167, 0.0763677593943165, 0.0725691718501518, 0.0675664548636212, 0.0616377739061904, 0.0550934038915087, 0.0482490884886937};
	//const float gauss_kernel[13] = {0.0558645795815233, 0.0650859248784428, 0.0737520150924311, 0.0812824836723717, 0.0871277109992678, 0.0908347254585746, 0.0921051206347776, 0.0908347254585746, 0.0871277109992678, 0.0812824836723717, 0.0737520150924311, 0.0650859248784428, 0.0558645795815233};
	const float gauss_kernel[11] = {0.0663, 0.0794, 0.0914, 0.1010, 0.1072, 0.1094, 0.1072, 0.1010, 0.0914, 0.0794, 0.0663};
	//const float gauss_kernel[9] = {0.0817, 0.1016, 0.1188, 0.1305, 0.1348, 0.1305, 0.1188, 0.1016, 0.0817};
	//const float gauss_kernel[7] = {0.1063, 0.1403, 0.1658, 0.1752, 0.1658, 0.1403, 0.1063};
	//const float gauss_kernel[5] = {0.1524, 0.2217, 0.2518, 0.2217, 0.1524};
	Mat Gauss = Mat::zeros(gsw_2+1, 1, CV_32F);
	for(int i=0; i<gsw_2+1; i++)
		((float*)Gauss.data)[i] = gauss_kernel[i];
	((float*)Gauss.data)[gsw] = 0;

	const unsigned char H_cnstrt = 1;					//��Ӧ����Լ��
	const int disappear_tolerance = reserve_times;//70;	//����70֡�����־��޳�֮
	const int out_range_tolerance = 20;	//�뿪�߽硢δ���ִ��������̶�
	const float side_range = 10;				//�ж��Ƿ�ӽ��߽����ֵ���������ֵ���Դﵽ30+����
	const int total_time_tolerance = 20;			//���ִ������ٵĹ켣����30�κ����
	const int total_time_below = 5;					//���ִ�������15�εĹ켣�������
	const unsigned char retrieval_num = 100;	//�켣�һظ�����ֵ�������һع켣
	const unsigned char similarity_tolerance = 120;	//�켣��������ֵ�������һع켣
	vector<Mat> Trans_between(gsw);				//t-gsw+1֡��t-gsw֮֡��ĵ�Ӧ���󣬿���Ҫ�޸�Ϊ��ǰ֡������gsw_2֮֡��ĵ�Ӧ���󣬱�����ǰ���켣�ж�ʱ�ظ�����
	vector<Mat> Trans_between_temp(gsw);				//t-gsw+1֡��t-gsw֮֡��ĵ�Ӧ���󣬿���Ҫ�޸�Ϊ��ǰ֡������gsw_2֮֡��ĵ�Ӧ���󣬱�����ǰ���켣�ж�ʱ�ظ�����

	vector<unsigned int> DT;								//�����������ε��������㼯��
	vector<float> Dist;										//����ÿ��ǰ���������������ھӱ������ľ���
	unsigned char trj_num_lower = 55;					//���Ʊ����켣����Ҫ̫�٣�������������ӳ����Ƶ�������ٶ�
	unsigned char trj_num_higher = 110;					//���Ʊ����켣����Ҫ̫�࣬���FREAK��ֵ�������������٣���ʱ�������ܿ���FREAK�����͹켣����
	deque<int> bg_trj;										//��¼��500֡�ı����켣�����仯�������������Ӧ����ʹ��
	float alpha_uper = 0.05;										//�����켣��������ʱ����Ҫ����һ����������
	float alpha_lower = 0.1;									//�����켣��������ʱ����Ҫ����һ����������
	unsigned char lowhigh_times_to_alter = 20;		//ÿ��20֡�޸�һ��lower��higher��ֵ
	unsigned char sub_size = 6;
	deque<int> trj_num_bg;							//�����20֡�������ֵı����켣��Ŀ
	unsigned char quick_pk_th = 10;					                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           	//��Կ�����ƵҪ������ֵ��ÿ��һ��֡���޸�pk_threshֵ
	vector<Point2f> Trj_bg_shorter;						//�켣����֡������gsw_2+1���Ǵ���gsw+1�ı����켣����
	const unsigned char lambda = 100;					//bundled paths Optimization����
	const float beta = 0;									//���׵�����ļ�Ȩϵ��
	const float w_nebor = 0.5;							//���ڹ켣�ļ�Ȩϵ����ԭʼ��ϵ��Ϊ2��̫���ˣ��˴���Ϊ2*w_nebor
	deque<Mat>  nebor_frames;							//�����ڵ�֡��֯��˫�߶��е���ʽ
#if MATCH_PAUSE
	deque<Mat>  nebor_frames_crop;							//�����ڵ�֡��֯��˫�߶��е���ʽ
#endif
	int show_start_num = 3;								//�ӵڼ�֡��ʼ��ʾ�����㡢ǰ���ж�����
	float pk_thresh = 0.00001;									//����FREAK����
	int mindist = 6;												//�����ǵ����С����
	const unsigned char level = 3;							//����������
	const unsigned char Octaves = 2;						//����������
	const unsigned char blur_size = 3;						//ͼ��ƽ�����ڴ�С
	////�ο�֡��ȡFREAK
	//st = cvGetTickCount();
	//GaussianBlur(crop_ref, crop_ref, Size(blur_size, blur_size), 0);
	//et = cvGetTickCount();
	//printf("��˹ģ��ʱ��: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 

	// DETECTION
	// Any openCV detector such as
#if USE_SURF
	SurfFeatureDetector detector(num_of_corners, Octaves, level);
#else
	GoodFeaturesToTrackDetector detector(num_of_corners, pk_thresh, mindist, 3, 1);//��ΪҪ�ֿ飬���ԣ���Ҫ�����м���һ�������������
#endif

	// DESCRIPTOR
	// FREAK extractor(true, true, 22, 4, std::vector<int>());
	FREAK extractor;

	// detect
	st = cvGetTickCount();
	detector.detect(crop_ref, Trj_keys, mask);
	//MyFAST(crop_ref, Trj_keys, num_of_corners, 5, mindist, mask, true);
	et = cvGetTickCount();
	printf("Harris detect time: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 

	// extract
	st = cvGetTickCount();
	extractor.compute(crop_ref, Trj_keys, Trj_desc);
	et = cvGetTickCount();
	printf("Harris number %d\n", Trj_keys.size());
	printf("FREAK extracting time: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 

	// �켣���Ծ����ٳ�ʼ���͸�ֵΪ0
	vector<Trj> Trajectories(Trj_keys.size());			//****************��Ҫ��������������й켣������ֵ*****************//
	//���켣���Ծ���ֵ
	int Trj_keys_size = Trj_keys.size();
	for(int i=0; i<Trj_keys_size; i++)
		Trajectories[i].trj_cor[0] = Trj_keys[i].pt;

	// ֡��
	int t = 1;
	//����һ֡ѹ�����
	nebor_frames.push_back(Mat(frame_ref));
	Mat crop_ref_copy = crop_ref.clone();
#if MATCH_PAUSE
	nebor_frames_crop.push_back(crop_ref_copy);
#endif
	// ��ǵ�ǰ����������Щ�Ѿ����ɵĹ켣ƥ����
	bool *Surf_index = NULL;

	//��¼��һ֡ƥ���ϵĵ���Trajectories�е�λ��
	vector<int> last_in_Trj;
	vector<KeyPoint> last_key = Trj_keys;
	Mat last_desc = Trj_desc.clone(), last_crop = crop_ref.clone();
	stitch_image = cvCreateImage(cvSize(width*2, height), frame_ref->depth, frame_ref->nChannels);

	//��������
	namedWindow("CurMatched", WINDOW_NORMAL); 
	resizeWindow("CurMatched", 720, 480);
	namedWindow("continuity_keypoint", WINDOW_NORMAL); 
	resizeWindow("continuity_keypoint", 720, 480);
	namedWindow("foreground_cur_H_cnstrt", WINDOW_NORMAL); 
	resizeWindow("foreground_cur_H_cnstrt", 720, 480);
#if SHOW_BG_POINT
	namedWindow("SHOW_BG_POINT", WINDOW_NORMAL); 
	resizeWindow("SHOW_BG_POINT", 720, 480);
#endif
	namedWindow("�һ�֮�����еı�����", WINDOW_NORMAL); 
	resizeWindow("�һ�֮�����еı�����", 720, 480);
	string winName = "Matches";
	namedWindow( winName, WINDOW_NORMAL );
	resizeWindow(winName, 2240, 630);

	// ��������
	while(frame_cur = cvQueryFrame(pCapture))
	{
		t++;
		cout<<t<<endl;
		//�Ȳ�����֡��̫�����Ƶ֡
		// �ο�֡תΪ�Ҷ�ͼ
		//cur_frame = Mat(frame_cur);
		st = cvGetTickCount();
		cvCvtColor(frame_cur, gray, CV_BGR2GRAY);
		cvResize(gray, dst);
		Mat sne(dst);//�������������ݣ�ֻ��������ͷ
		Mat crop_cur = sne(Range(cropped_start/scale, cropped_end/scale), Range(0, width/scale));
		et = cvGetTickCount();
		printf("ͼ�����ʱ��: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 
		////��ȡFREAK����;
		//st = cvGetTickCount();
		//GaussianBlur(crop_cur, crop_cur, Size(blur_size, blur_size), 0);
		//et = cvGetTickCount();
		//printf("��˹ģ��ʱ��: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 

		// detect
		cur_key.clear();
		st = cvGetTickCount();
		detector.detect(crop_cur, cur_key, mask);
		//MyFAST(crop_cur, cur_key, num_of_corners, 5, mindist, mask, true);
		et = cvGetTickCount();
		printf("Harris detect time: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 

		// extract
		st = cvGetTickCount();
		extractor.compute(crop_cur, cur_key, cur_descriptor);
		et = cvGetTickCount();
		printf("Harris number %d\n", cur_key.size());
		printf("FREAK extracting time: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 

		cout<<"PK_thresh="<<pk_thresh<<endl;
		//��ȡ��ǰ֡FREAK������
		//ƥ���ϾɵĹ켣��FREAK�������ֵ
		st = cvGetTickCount();
		delete []Surf_index;
		Surf_index = new bool[cur_key.size()];
		int cur_key_size = cur_key.size();
		for(int i=0; i<cur_key_size; i++)
			Surf_index[i] = false;

		vector<DMatch> matches;
		int this_cor_number = t > reserve_times+1 ? reserve_times+1 : t;	//******************************��ʾ��֡���Ƿ񳬹�50�����ڳ���Ƶ���ر���Ҫ������������***********************************//
		int last_in_Trj_size = last_in_Trj.size();
		if(!last_in_Trj.size())
		{
			vector<vector<DMatch>> matches_for_SM(cur_key.size());
			//vector<DMatch> matches_for_SM(cur_key.size());
			//�ǿգ��򽫵�ǰ����������һ֡��������ƥ��
			int k=2;
			// match
			st = cvGetTickCount();
			naive_nn_search2(last_key, last_desc, cur_key, cur_descriptor, matches_for_SM, max_shaky_dist, k); 
			et = cvGetTickCount();
			printf("matching time: %f\n ", (et-st)/(float)cvGetTickFrequency()/1000.); 

			st = cvGetTickCount();
			my_spectral_matching(cur_key, last_key, matches_for_SM, k, matches);
			et = cvGetTickCount();
			printf("My_Spectral_Matching�㷨ƥ��ʱ��: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 
			cout<<"Spectral matchingƥ������"<<matches.size()<<"��"<<endl;
		}
		else
		{
			vector<vector<DMatch>> matches_for_SM(cur_key.size());
			vector<DMatch>matches_temp;
			//�ǿգ��򽫵�ǰ����������һ֡��������ƥ��
			int k=2;
			naive_nn_search2(last_key, last_desc, cur_key, cur_descriptor, matches_for_SM, max_shaky_dist, k); 
			cout<<matches_for_SM.size()<<endl;
			//ofstream m_file("matches_for_SM.txt");
			//m_file<<matches_for_SM<<endl;
			st = cvGetTickCount();
			my_spectral_matching(cur_key, last_key, matches_for_SM, k, matches_temp);
			et = cvGetTickCount();
			printf("My_Spectral_Matching�㷨ƥ��ʱ��: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 
			cout<<"Spectral matching,����һ֡ƥ������"<<matches_temp.size()<<"��"<<endl;
			//���ƥ����̫�ͣ��ͷſ���ֵ����ƥ�䣬���ҷſ�ͷ�����
			int match_times = 1;
			while (matches_temp.size()<num_of_corners/2-100 && match_times<5)
			{
				match_times ++;
				max_shaky_dist+=800;
				for (int i=0; i<cur_key_size; i++)
					matches_for_SM[i].clear();
				//���ͷ�ǰ���켣
				//penalization_thresh = 10;
				//penalization_relax = 10;
				matches_temp.clear();
				naive_nn_search2(last_key, last_desc, cur_key, cur_descriptor, matches_for_SM, max_shaky_dist, k); 
				cout<<matches_for_SM.size()<<endl;
				st = cvGetTickCount();
				my_spectral_matching(cur_key, last_key, matches_for_SM, k, matches_temp);
				et = cvGetTickCount();
				printf("My_Spectral_Matching�㷨ƥ��ʱ��: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 
				cout<<"Spectral matching,����һ֡ƥ������"<<matches_temp.size()<<"��"<<endl;
			}
			cout<<"��ǰƥ����ֵ"<<max_shaky_dist<<endl;
			max_shaky_dist = max_dist;
			//ƥ����ɺ���ֵ��Сһ��???
			//if(t >= show_start_num)
			//{
				Mat showImg;
				drawMatches(crop_cur, cur_key, last_crop, last_key, matches_temp, showImg, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
				
				imshow( winName, showImg );
				//waitKey(0);
				Mat showCurMatched;
				int matchsize = matches_temp.size();
				vector<KeyPoint> matched_keys;
				for (int i=0; i<matchsize; i++)
					matched_keys.push_back(cur_key[matches_temp[i].queryIdx]);
				
				drawKeypoints(crop_cur, matched_keys, showCurMatched);
				imshow("CurMatched", showCurMatched);
				cvWaitKey(5);
			//}
			//ƥ����ת��
			int matches_temp_size = matches_temp.size();
			//cout<<"����һ֡ƥ������"<<matches_temp_size<<"����"<<endl;
			for(int i=0; i<matches_temp_size; i++)
				matches_temp[i].trainIdx = last_in_Trj[matches_temp[i].trainIdx];

			//��¼δƥ���ϵ�������
			vector<bool> mached_cur(cur_key.size());
			mached_cur.assign(cur_key.size(), false);
			for(int i=0; i<matches_temp_size; i++)
				mached_cur[matches_temp[i].queryIdx] = true;//��ǣ�cur_key�е����ֵ�Ѿ���ƥ����
			//cur_key��ʣ�µĵ�
			if(cur_key_size - matches_temp_size)
			{
				vector<int> left_in_cur(cur_key.size()-matches_temp.size());
				vector<KeyPoint> left_cur_key;
				vector<DMatch> matches_left;//δƥ���ϵģ�����Trj_desc�����µ�ƥ��
				int left_num = 0;
				//��¼δƥ���ϵ�������������cur_key��Trj_desc
				Mat left_desc;
				for(int i=0; i<cur_key_size; i++)
				{
					if(!mached_cur[i])
					{
						left_in_cur[left_num] = i;
						left_num++;
						left_desc.push_back(cur_descriptor.row(i));
						left_cur_key.push_back(cur_key[i]);
					}
				}
				//���µĸ�Trj_descƥ��
				naive_nn_search(Trj_desc, left_desc, matches_left); 
				//ƥ����ת��
				int matches_left_size = matches_left.size();
				for(int i=0; i<matches_left_size; i++)
					matches_left[i].queryIdx = left_in_cur[matches_left[i].queryIdx];

				//�޳��ظ���
				//����
				matches_temp.insert(matches_temp.end(), matches_left.begin(), matches_left.end());
				//���յڶ�������������
				sort(matches_temp.begin(), matches_temp.end(), compare1);
				matches_temp_size = matches_temp.size();
				//����֮��trainIdx��������������ظ���
				for(int i=0; i<matches_temp_size; i++)
				{
					matches.push_back(matches_temp[i]);
					if(i > 0)
					{
						if(matches_temp[i].trainIdx == matches_temp[i-1].trainIdx)
						{
							if(mached_cur[matches_temp[i-1].queryIdx])
								matches.pop_back();
							else
							{
								swap(matches[matches.size()-1], matches[matches.size()-2]);
								matches.pop_back();
							}
						}
					}
				}
			}
			else
				matches = matches_temp;
		}
		et = cvGetTickCount();
		printf("������ƥ��ʱ��: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.);
		//****************************************��¼��һ֡��������������������Trajectories�е�λ��**************************************************//
		st = cvGetTickCount();
		last_desc = cur_descriptor.clone();
		last_key = cur_key;
		last_crop = crop_cur.clone();
		last_in_Trj.resize(cur_key_size);
		last_in_Trj.assign(cur_key_size, -1);
		int matches_size = matches.size();
		//��������������ʾ��ǰ��ƥ��㼰����Trajectories�еı��
		vector<KeyPoint> cur_matched_key(matches_size);
		vector<int> cur_matched_key_index_Trj(matches_size);
		for(int i=0; i<matches_size; i++)
		{
			last_in_Trj[matches[i].queryIdx] = matches[i].trainIdx;
			cur_matched_key[i] = cur_key[matches[i].queryIdx];
			cur_matched_key_index_Trj[i] = matches[i].trainIdx;
		}

		//�켣ƥ�����
		last_size = Trj_desc.rows;
		for(int i=0; i<matches_size; i++)
		{
			Surf_index[matches[i].queryIdx] = true;
			Trajectories[matches[i].trainIdx].count += 1;
			Trajectories[matches[i].trainIdx].continuity += 1;
			Trajectories[matches[i].trainIdx].last_number = t;
			if (Trajectories[matches[i].trainIdx].penalization)
				Trajectories[matches[i].trainIdx].penalization--;
			//ǰ�����ٳ��֣����Գͷ�
			if (Trajectories[matches[i].trainIdx].foreground_times && Trajectories[matches[i].trainIdx].last_number <= (t-10) )
			{	
				Trajectories[matches[i].trainIdx].foreground_times += 1;
				if (Trajectories[matches[i].trainIdx].penalization)
					Trajectories[matches[i].trainIdx].penalization += 3;
				else
					Trajectories[matches[i].trainIdx].penalization = 3;
			}
			else if (Trajectories[matches[i].trainIdx].last_number <= (t-10) )
			{	
				Trajectories[matches[i].trainIdx].penalization = 3;
			}
			if(t > reserve_times)
			{
				Trajectories[matches[i].trainIdx].trj_cor.pop_front();
				Trajectories[matches[i].trainIdx].trj_cor.push_back(cur_key[matches[i].queryIdx].pt); //���tҪ��Ϊt_after = t > 1000? t-1000:t
			}
			else
				Trajectories[matches[i].trainIdx].trj_cor[t-1] = cur_key[matches[i].queryIdx].pt; //���tҪ��Ϊt_after = t > 1000? t-1000:t
			cur_descriptor.row(matches.at(i).queryIdx).copyTo(Trj_desc.row(matches.at(i).trainIdx));	////����Harris��ⲻ���г߶Ȳ����ԣ����ԣ��˴��б�Ҫ��ÿ�θ�����������
		}
		//δ���ֵĹ켣��continuity=0
		for(int i=0; i<last_size; i++)
			if(Trajectories[i].last_number != t)
			{
				Trajectories[i].continuity = 0;
				if(t > reserve_times)
				{
					Trajectories[i].trj_cor.pop_front();
					Trajectories[i].trj_cor.push_back(Point2f(0, 0));
				}
				//�Դ�ǰ���켣���ǲ�����и
				//if (Trajectories[i].award)
				//	Trajectories[i].award--;
			}
			//�³��ֵĹ켣���
			for(int i=0; i<cur_key_size; i++)
			{
				//cout<<i<<endl;
				if(!Surf_index[i])
				{
					Trajectories.push_back(Trj(1, t, 1, 0, 0, 0, reserve_times));
					Trj_desc.push_back(cur_descriptor.row(i));
					//****************************************��¼��һ֡��������������������Trajectories�е�λ��**************************************************//
					int Trajectories_size = Trajectories.size();
					last_in_Trj[i] = Trajectories_size-1;
					if(t > reserve_times)
					{
						Trajectories[Trajectories_size-1].trj_cor.pop_front();
						Trajectories[Trajectories_size-1].trj_cor.push_back(cur_key[i].pt); //���tҪ��Ϊt_after = t > 1000? t-1000:t
					}
					else
						Trajectories[Trajectories_size-1].trj_cor[t-1] = cur_key[i].pt; //���tҪ��Ϊt_after = t > 1000? t-1000:t
				}
			}
			et = cvGetTickCount();
			printf("�켣��Ӻ͹켣����ά��ʱ��: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.);
			//���������˲��Ĺ켣�������
			vector<vector<Point2f>>Trj_cor_for_smooth;
			//����gsw_2+1֡�����֣���û�б��ж�Ϊǰ���㳬��H_cnstrt��
			if (t >= gsw_2+1)
			{
				//�ش��޸İ�������������drawmatchesʱÿ�ζ���ʾͬһ֡������
				//����nebor_frames��Ҳ��ͬ��������
				//*****************************************************��mat������IplImage����ֹ�ڴ�й¶��ֹ�ڴ�й¶****************************************************//
				st = cvGetTickCount();
				Mat crop_cur_copy = crop_cur.clone();
				Mat cur_copy(frame_cur, true);
				//��һ���ش��޸ģ�����˴����⣬��ΪcvWarpPerspective������Ҫ����nebor_frames�����ԣ��������Ҳ�������ظ�֡���֣���ģ����������ˣ�
				//���ζ��У����׳��ӣ���֡ѹ���β
				if(t>gsw_2+1)
				{
					nebor_frames.pop_front();
					nebor_frames.push_back(cur_copy);
#if MATCH_PAUSE
					nebor_frames_crop.pop_front();
					nebor_frames_crop.push_back(crop_cur_copy);
#endif
				}
				else
				{
					nebor_frames.push_back(cur_copy);
#if MATCH_PAUSE
					nebor_frames_crop.push_back(crop_cur_copy);
#endif
				}
				et = cvGetTickCount();
				printf("����֡�������ά��ʱ��: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.);
				//�鿴FREAK�������ƥ��Ч��
				st = cvGetTickCount();
				Mat img_matches;
				//����ɸѡ�Ĺ켣�������
				vector<vector<Point2f>>Trj_cor_continuity;
				vector<KeyPoint> cur_continuity_key;
				unsigned int num_trj_for_H_cnstrt = 0;		//*****�������ڵ�Ӧ�����б�ǰ����������******
				vector<int>continuity_index;				//����gsw_2+1֡���֣����ڹ켣�һ�
				unsigned int continuity_num = 0;			//�������ֵĹ켣��
				vector<int>H_cnstrt_index;				//�������ֵı�������continuity_index�е����
				vector<int>foreground_index;				//�������ֵ�ǰ����
				int foreground_num = 0;		//ǰ�������
				vector<bool>continuity_fg;				//�������Ƿ�ǰ���㣬���ڱ������һس�����
				this_cor_number = t > reserve_times ? reserve_times : t;	//******************************�ڴ�ʱ��this_cor_numberӦ����һ�¡��ò������ڳ���Ƶ���ر���Ҫ������������***********************************//
				vector<int>fg_bg_times;
				for(int i=0; i<matches_size; i++)//����Ҫ���������֣���ôֻҪ��ǰ֡��ƥ���ϵĹ켣������
				{
					//ǰ���켣�ͷ�����
					if(Trajectories[matches[i].trainIdx].continuity >= gsw_2+1)// && Trajectories[matches[i].trainIdx].penalization < penalization_thresh)
					{
						//*****************************��Ҫ*********************************�������������ֵ������continuity_index�й�ϵ
						continuity_index.push_back(matches[i].trainIdx);
						continuity_num++;
						//�Ƚ��������Trj_cor_continuity
						vector<Point2f> temp;
						continuity_fg.push_back(false);
						for(int j=gsw_2; j >= 0; j--)
						{
							Point2f tem_pt(Trajectories[matches[i].trainIdx].trj_cor[this_cor_number-j-1].x, Trajectories[matches[i].trainIdx].trj_cor[this_cor_number-j-1].y);//�Ժ�Ҫ��Ϊi%500?
							temp.push_back(tem_pt);
						}
						Trj_cor_continuity.push_back(temp);
						//�����ؼ���ѹ��vector
						cur_continuity_key.push_back(cur_key[matches[i].queryIdx]);
						//***********����gsw_2+1֡���ֲ��ұ���Ϊǰ���켣����������H_cnstrt��*************//
						if(Trajectories[matches[i].trainIdx].foreground_times < H_cnstrt)
						{
							H_cnstrt_index.push_back(continuity_num-1);
							num_trj_for_H_cnstrt++;
						}
						else //if(Trajectories[matches[i].trainIdx].foreground_times >= H_cnstrt)	//�Ѿ�����Ϊǰ����������������������
						{
							fg_bg_times.push_back(Trajectories[matches[i].trainIdx].background_times);
							continuity_fg[continuity_num-1] = true;
							foreground_num++;
							foreground_index.push_back(continuity_num-1);//***************��Ҫ***************������continuity_index�����е���š���������Trajectories�е���š�����Ѱַ����������������
						}
					}
					//else if(Trajectories[matches[i].trainIdx].continuity >= gsw_2+1 && Trajectories[matches[i].trainIdx].foreground_times >= 6)
					//	cout<<"����ǰ���㣡��������������"<<matches[i].trainIdx<<"\t"<<Trajectories[matches[i].trainIdx].foreground_times<<endl;
				}

				//��ʾ�������ĵ�
				cout<<"ƥ���ϵ���:"<<matches_size<<endl;
				cout<<"�����ĵ�����"<<continuity_num<<endl;
				cout<<"һ����"<<foreground_num<<"��ǰ����"<<endl;
				et = cvGetTickCount();
				printf("�����켣��ȡʱ��: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.);

				//*******************���ǰ����**********************//
				st = cvGetTickCount();
				unsigned char *Foreground_times = new unsigned char[num_trj_for_H_cnstrt];					//��¼�����жϹ����У�ÿ�������㱻��Ϊǰ����Ĵ���
				for(int i=0; i<num_trj_for_H_cnstrt; i++)
					Foreground_times[i] = 0;
				//��ǰ֡�Ĺ켣����
				vector<Point2f> pt_cur;
				vector<KeyPoint> key_cur, key_t, pt_contuity;
				vector<DMatch> good_matches;
				int good_match_num = 0;	//������good_match��
				for(int i = 0; i < continuity_num; i++)
				{
					if(!continuity_fg[i])
					{
						good_matches.push_back(DMatch(good_match_num, good_match_num, 0.1));
						pt_cur.push_back(Point2f(Trj_cor_continuity[i][gsw].x, Trj_cor_continuity[i][gsw].y));
						key_cur.push_back(KeyPoint((Point2f(Trj_cor_continuity[i][gsw].x, Trj_cor_continuity[i][gsw].y)), 12.));
						key_t.push_back(KeyPoint((Point2f(Trj_cor_continuity[i][gsw].x, Trj_cor_continuity[i][gsw].y)), 12.));
						good_match_num++;
					}
					pt_contuity.push_back(KeyPoint(Point2f(Trj_cor_continuity[i][gsw].x, Trj_cor_continuity[i][gsw].y), 12));
				}
				Mat continuity_keypoint;
				drawKeypoints(nebor_frames_crop.at(gsw), pt_contuity, continuity_keypoint);
				imshow("continuity_keypoint", continuity_keypoint);
				cvWaitKey(5);
				float H_cnstrt_error = 0;
				//gsw_2���ھ�֡�Ĺ켣����
				vector<vector<Point2f>> pt_nebor(gsw_2);
				vector<Mat> homo_H_cnstrt(gsw_2);
				vector<Point2f>reproj;
				for(int i=0; i<gsw_2; i++)
					homo_H_cnstrt[i] = Mat::zeros(3, 3, CV_32F);

				for(int j = 0; j < continuity_num; j++)
				{
					if(!continuity_fg[j])
						for(int i=0; i<gsw; i++)
						{
							pt_nebor[gsw_2-i-1].push_back(Point2f(Trj_cor_continuity[j][gsw_2-i].x, Trj_cor_continuity[j][gsw_2-i].y));
							pt_nebor[i].push_back(Point2f(Trj_cor_continuity[j][i].x, Trj_cor_continuity[j][i].y));
						}
				}
				//���̳߳�ʼ������
				//�ؼ��γ�ʼ��  
				//InitializeCriticalSection(&g_csThreadParameter);  
				//InitializeCriticalSection(&g_csThreadCode);  
				const int THREAD_NUM = gsw_2;  
				HANDLE handle[THREAD_NUM]; 
				vector<PARA_FOR_HOMO> pthread_array(gsw_2);

				//��ʼ�߳�
				for (int i=0; i<gsw_2; i++)
				{
					//EnterCriticalSection(&g_csThreadParameter);//�������߳���Źؼ�����
					pthread_array[i] = para_for_homo(pt_nebor[i], pt_cur, true, Foreground_times, crop_width, crop_height);
					handle[i] = (HANDLE)_beginthreadex(NULL, 0, Calculate_Homography, &pthread_array[i], 0, NULL); 
				}

				//�ȴ�gsw_2���̼߳������
				WaitForMultipleObjects(THREAD_NUM, handle, TRUE, 500);//INFINITE);//����ȴ�20ms?�������˱����켣����<20������ʱ������Ҫ����ʱ�������𣿣�
				//�����̲߳��ͷ���Դ
				DWORD aExitCode = 0;
				for (int i=0; i<gsw_2; i++)
				{
					//CloseHandle(handle[i]);
					TerminateThread(handle[i], aExitCode);
				}
				//����н��̲�������������Calculate_Homography_completed��Ϊfalse
				bool Calculate_Homography_completed = true;
				for (int i=0; i<gsw_2; i++)
				{
					if (pthread_array[i].start)
						Calculate_Homography_completed = false;
				}
				if (!Calculate_Homography_completed)
				{
					cout<<"���̼߳��㵥Ӧ����ǰ�����ж����֡��쳣��������������������������������������������������1"<<endl;
					continue;
				}
				et = cvGetTickCount();
				printf("ǰ�����ж�ʱ��: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.);
				//for(int k = 0; k < num_trj_for_H_cnstrt; k++)
				//	printf("%d\t",Foreground_times[k]);
				//cout<<endl;
				st = cvGetTickCount();
				//�����ı����켣��Ŀ
				int bg_num = num_trj_for_H_cnstrt;
				//��ǰ֡�ı����켣����
				//****************��Ҫ****************���汳���켣��gsw_2+1֡�еĹ켣���꣬���ڹ켣ƽ��������Ŀ��֤��bg_numһ��
				vector<Point2f> pt_bg_cur;
				//**************��Ҫ***********��¼�����켣��continuity_index�е�λ��
				vector<int>smooth_index;
				vector<int>fg_H_cnstrt_index;
#if SHOW_BG_POINT
				vector<KeyPoint> key_bg_cur;
#endif
				vector<KeyPoint> key_fg_cur_H_cnstrt;
				for(int i=0; i<num_trj_for_H_cnstrt; i++)
				{
					if((float(Foreground_times[i])/(gsw_2)) >= 0.5)
					{
						Trajectories[continuity_index[H_cnstrt_index[i]]].foreground_times++;
						//Trajectories[continuity_index[H_cnstrt_index[i]]].penalization = Trajectories[continuity_index[H_cnstrt_index[i]]].award>0 ? 10/(Trajectories[continuity_index[H_cnstrt_index[i]]].background_times/10) : 10;
						//�����켣��--
						bg_num--;
						continuity_fg[H_cnstrt_index[i]] = true;
						fg_H_cnstrt_index.push_back(i);
						//ǰ���켣��++
						foreground_num++;
						key_fg_cur_H_cnstrt.push_back(key_t[i]);
						foreground_index.push_back(H_cnstrt_index[i]);//***************��Ҫ***************������Ǹõ���continuity_index�����е���š���������Trajectories�е���š�
					}
					else
					{
						//��¼�����켣��continuity_index�е�λ��
						Trajectories[continuity_index[H_cnstrt_index[i]]].background_times++;
						Trajectories[continuity_index[H_cnstrt_index[i]]].award = Trajectories[continuity_index[H_cnstrt_index[i]]].background_times / 40;	//ÿ50�ν���һ��
						smooth_index.push_back(H_cnstrt_index[i]);
						Trj_cor_for_smooth.push_back(Trj_cor_continuity[H_cnstrt_index[i]]);
						pt_bg_cur.push_back(Trj_cor_continuity[H_cnstrt_index[i]][gsw]);
#if SHOW_BG_POINT
						key_bg_cur.push_back(KeyPoint((Point2f(Trj_cor_continuity[H_cnstrt_index[i]][gsw].x, Trj_cor_continuity[H_cnstrt_index[i]][gsw].y)), 12.));
#endif
					}
				}
				//�����ã���ʾ��Щ�㱻�ж�Ϊǰ����
				if(num_trj_for_H_cnstrt-bg_num)
				{
					cout<<"�����ַ�����"<<num_trj_for_H_cnstrt-bg_num<<"��"<<endl;
					Mat cur_foregroud_keypoint;
					drawKeypoints(nebor_frames_crop.at(gsw), key_fg_cur_H_cnstrt, cur_foregroud_keypoint);

					imshow("foreground_cur_H_cnstrt", cur_foregroud_keypoint);
					cvWaitKey(5);
				}
				//��������Ϊ��ǰ�����б��У������ƻ�foreground_index��������
				sort(foreground_index.begin(), foreground_index.end());

#if SHOW_BG_POINT
				drawKeypoints(nebor_frames_crop.at(gsw), key_bg_cur, img_matches);
				imshow("SHOW_BG_POINT", img_matches);
				cvWaitKey(0);
#endif
				printf("bg_num num: %d\n", bg_num); 
				delete []Foreground_times;
				et = cvGetTickCount();
				printf("������켣������ȡʱ��: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.);
				//*******************************�������һ�********************************//
				st = cvGetTickCount();
				if(bg_num < retrieval_num)
				{
					//����3��ǰ���㣬���ܽ���������������
					if(foreground_num-2 > 0)
					{
						//�������������ֵĹ켣���꣬�ڵ�ǰ֡���������������ηָ�
						CvRect rect = {0, 0, crop_width, crop_height};
						CvMemStorage* storage_bundled;
						CvSubdiv2D* subdiv;
						storage_bundled = cvCreateMemStorage(0);
						subdiv = cvCreateSubdiv2D( CV_SEQ_KIND_SUBDIV2D, sizeof(*subdiv),
							sizeof(CvSubdiv2DPoint),
							sizeof(CvQuadEdge2D),
							storage_bundled );//Ϊ�ʷ����ݷ���ռ�
						cvInitSubdivDelaunay2D( subdiv, rect );
						for(int i=0; i<continuity_num; i++)
						{
							CvSubdiv2DPoint *pt = cvSubdivDelaunay2DInsert(subdiv, CvPoint2D32f(Trj_cor_continuity[i][gsw_2]));//�������ʷ��в���õ㣬���Ըõ���������ʷ�
							//pt->id = continuity_index[i];//**************************��Ҫ��������Ϊÿһ���������һ��id********************************//
							pt->id = i;	//��ֱ�ӱ�����Trajectories�е���ţ����Ǳ�����continuity_index�е���ţ��������жϸõ��Ƿ�ǰ���㣬�����ڼ����ھӵ���ǰ���������ʱ��
						}
						//ɸѡ����ȷ�ď����������Σ�������ÿ�������������ε���������
						CvSeqReader  reader;
						cvStartReadSeq( (CvSeq*)(subdiv->edges), &reader, 0 );//ʹ��CvSeqReader����Delaunay����Voronoi��
						int edges_num = subdiv->edges->total;
						Vec3i verticesIdx;
						vector<Vec3i> Delaunay_tri;		//�洢�����������ε���������ļ���
						Point buf[3];							//�洢�����߶�Ӧ�Ķ���
						const Point *pBuf = buf;
						int elem_size = subdiv->edges->elem_size;//�ߵĴ�С
						for(int i=0; i<edges_num; i++)
						{
							CvQuadEdge2D* edge = (CvQuadEdge2D*)(reader.ptr);
							if( CV_IS_SET_ELEM( edge )) 
							{
								CvSubdiv2DEdge t = (CvSubdiv2DEdge)edge; 
								int iPointNum = 3;
								Scalar color=CV_RGB(255,0,0);
								int j;
								for(j = 0; j < iPointNum; j++ )
								{
									CvSubdiv2DPoint* pt = cvSubdiv2DEdgeOrg( t );//��ȡt�ߵ�Դ��
									if( !pt ) break;
									buf[j] = pt->pt;//����洢����
									verticesIdx[j] = pt->id;//��ȡ�����Id�ţ����������id�洢��verticesIdx��
									t = cvSubdiv2DGetEdge( t, CV_NEXT_AROUND_LEFT );//��ȡ��һ����
								}
								if (j != iPointNum) continue;
#if SHOW_DELAUNAY
								if (isGoodTri(verticesIdx, Delaunay_tri))
								{
									polylines( nebor_frames_crop[gsw], &pBuf, &iPointNum, 
										1, true, color,
										1, CV_AA, 0);//����������
								}
#else
								isGoodTri(verticesIdx, Delaunay_tri);
#endif

								t = (CvSubdiv2DEdge)edge+2;//�෴��Ե reversed e
								for(j = 0; j < iPointNum; j++ )
								{
									CvSubdiv2DPoint* pt = cvSubdiv2DEdgeOrg( t );
									if( !pt ) break;
									buf[j] = pt->pt;
									verticesIdx[j] = pt->id;
									t = cvSubdiv2DGetEdge( t, CV_NEXT_AROUND_LEFT );
								}   
								if (j != iPointNum) continue;
#if SHOW_DELAUNAY
								if (isGoodTri(verticesIdx, Delaunay_tri))
								{
									polylines( nebor_frames_crop[gsw], &pBuf, &iPointNum, 
										1, true, color,
										1, CV_AA, 0);
								}
#else
								isGoodTri(verticesIdx, Delaunay_tri);
#endif
							}
							CV_NEXT_SEQ_ELEM( elem_size, reader );
						}
#if SHOW_DELAUNAY
						imshow("������������", nebor_frames_crop[gsw]);
						cvWaitKey(0);
#endif
						//cout<<"�����������ε����������������Ѿ�ѹ��Delaunay_tri"<<endl;
						//Ѱ��ǰ������ھӵ�
						vector<vector<int>> nb_of_fg(foreground_num);
						//Ѱ��ÿһ��%ǰ��%����ھӵ㣬�����ظ������ԣ�Ҫ��ÿһ���ҵ��ھӵ�ʱ���жϸ�ǰ������ھӵ���������Ƿ����и��ھӣ���û�У���ѹ��nb_of_fg����
						//Ҳ�ɲ���ô�鷳����Ϊ�ں����õ�ǰ�����ھ�ʱ��ʹ��set�࣬set���Զ��ų��ظ��ĵ�
						for(int i=0; i < Delaunay_tri.size(); i++)
						{
							//Delaunay_tri��ά�ֵ���continuity_index�е���ţ����ԣ�ǰ���foreground_index���򣬲���Դ˴���Ӱ�죡����������������
							int ind = binary_search(foreground_index, Delaunay_tri[i].val[0]);
							if(ind != -1)
							{
								if(nb_of_fg[ind].size() > 0)
								{
									if(find(nb_of_fg[ind].begin(), nb_of_fg[ind].end(), Delaunay_tri[i].val[1]) == nb_of_fg[ind].end())
										nb_of_fg[ind].push_back(Delaunay_tri[i].val[1]);
									if(find(nb_of_fg[ind].begin(), nb_of_fg[ind].end(), Delaunay_tri[i].val[2]) == nb_of_fg[ind].end())
										nb_of_fg[ind].push_back(Delaunay_tri[i].val[2]);
								}
								else
								{
									nb_of_fg[ind].push_back(Delaunay_tri[i].val[1]);
									nb_of_fg[ind].push_back(Delaunay_tri[i].val[2]);
								}
							}
							ind = binary_search(foreground_index, Delaunay_tri[i].val[1]);
							if(ind != -1)
							{
								if(nb_of_fg[ind].size() > 0)
								{
									if(find(nb_of_fg[ind].begin(), nb_of_fg[ind].end(), Delaunay_tri[i].val[0]) == nb_of_fg[ind].end())
										nb_of_fg[ind].push_back(Delaunay_tri[i].val[0]);
									if(find(nb_of_fg[ind].begin(), nb_of_fg[ind].end(), Delaunay_tri[i].val[2]) == nb_of_fg[ind].end())
										nb_of_fg[ind].push_back(Delaunay_tri[i].val[2]);
								}
								else
								{
									nb_of_fg[ind].push_back(Delaunay_tri[i].val[0]);
									nb_of_fg[ind].push_back(Delaunay_tri[i].val[2]);
								}
							}
							ind = binary_search(foreground_index, Delaunay_tri[i].val[2]);
							if(ind != -1)
							{
								if(nb_of_fg[ind].size() > 0)
								{
									if(find(nb_of_fg[ind].begin(), nb_of_fg[ind].end(), Delaunay_tri[i].val[0]) == nb_of_fg[ind].end())
										nb_of_fg[ind].push_back(Delaunay_tri[i].val[0]);
									if(find(nb_of_fg[ind].begin(), nb_of_fg[ind].end(), Delaunay_tri[i].val[1]) == nb_of_fg[ind].end())
										nb_of_fg[ind].push_back(Delaunay_tri[i].val[1]);
								}
								else
								{
									nb_of_fg[ind].push_back(Delaunay_tri[i].val[0]);
									nb_of_fg[ind].push_back(Delaunay_tri[i].val[1]);
								}
							}
						}
						//cout<<"ǰ������ھӵ������Ѿ���ȫѹ��nb_of_fg"<<endl;
						Mat Trj_cor_x = Mat::zeros(continuity_num, gsw_2+1, CV_32F);
						Mat Trj_cor_y = Mat::zeros(continuity_num, gsw_2+1, CV_32F);
						for(int i=0; i<continuity_num; i++)
						{
							for(int j=gsw_2; j>=0; j--)
							{
								((float*)Trj_cor_x.data)[i*(gsw_2+1)+j] = Trj_cor_continuity[i][j].x - Trj_cor_continuity[i][0].x;
								((float*)Trj_cor_y.data)[i*(gsw_2+1)+j] = Trj_cor_continuity[i][j].y - Trj_cor_continuity[i][0].y;
							}
						}
#if OUTPUT_TRJ_COR_MAT
						ofstream outfile("Trj_cor_mat.txt");
						outfile<<Trj_cor_x<<endl;
						outfile<<Trj_cor_y<<endl;
#endif
						//�ж�ǰ�������ھӵ�Ĺ켣������
						for(int i=0; i<foreground_num; i++)
						{
							//cout<<"i = "<<i<<"\t"<<endl;
							//�����ǰ������Χǰ�������̫�࣬����Ϊǰ����
							int nb_num = nb_of_fg[i].size();

							if(nb_num > 0)
							{
								int sum_of_fg = 0;
								//ͳ�Ƹ�ǰ�����ھӵ��е�ǰ�������
								for(int j=0; j<nb_num; j++)
									sum_of_fg += continuity_fg[nb_of_fg[i][j]];
								//�ھӵ���ǰ������С��һ������ż���켣�����ԣ�����ά��ԭ��
								if(sum_of_fg < nb_num/2)
								{
									int similarity = 0;
									for(int j=0; j<nb_num; j++)
									{
										//1��7�е�����
										Mat d_x = Trj_cor_x.row(foreground_index[i]) - Trj_cor_x.row(nb_of_fg[i][j]);
										Mat d_y = Trj_cor_y.row(foreground_index[i]) - Trj_cor_y.row(nb_of_fg[i][j]);
										float distance = sum(d_x.mul(d_x) + d_y.mul(d_y)).val[0];
										//if((sqrt(distance)) <= similarity_tolerance)
										if(distance <= similarity_tolerance)
											similarity++;
									}
									if((similarity - nb_num/2) >= 0)
									{
										Trajectories[continuity_index[foreground_index[i]]].foreground_times--;
										//�ش��޸ģ��������һصĵ㲻ֱ�ӷŵ������켣�������棬���������ж�������
										if (Trajectories[continuity_index[foreground_index[i]]].foreground_times < H_cnstrt)//max_H_cnstrt_tolerance)
										{
											Trj_cor_for_smooth.push_back(Trj_cor_continuity[foreground_index[i]]);
											continuity_fg[foreground_index[i]] = false;
											bg_num++;
											smooth_index.push_back(foreground_index[i]);
											pt_bg_cur.push_back(Trj_cor_continuity[foreground_index[i]][gsw]);
										}
									}
									//else//�����ͷ���Щ����ǰ����
									//	if(Trajectories[continuity_index[foreground_index[i]]].foreground_times)
									//		Trajectories[continuity_index[foreground_index[i]]].foreground_times++;
								}
								//else//�����ͷ���Щ����ǰ����
								//	Trajectories[continuity_index[foreground_index[i]]].foreground_times++;
							}
							else
								cout<<"��ǰ����û���ھӵ㡣����"<<endl;	//û���ھӵ㣬 ��ʵ��һ��bug���õ�
						}
						cvReleaseMemStorage(&storage_bundled);
						foreground_num = continuity_num - bg_num;
						cout<<"�һز��ֱ�����֮�󣬱��������Ϊ"<<bg_num<<endl;
					}
					else cout<<"�ܿ�ϧ��û��ǰ������Ҫ�һء�����������������"<<endl;
				}
				et = cvGetTickCount();
				printf("�����켣�һ�ʱ��: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.);

				//�һ�֮�����еı�����
				Mat show_all_bg_keypoint;
				vector<KeyPoint> all_bg_key;
				for (int i=0; i<bg_num; i++)
				{
					all_bg_key.push_back(KeyPoint(Trj_cor_for_smooth[i][gsw_2].x, Trj_cor_for_smooth[i][gsw_2].y, 15));
				}
				drawKeypoints(crop_cur, all_bg_key, show_all_bg_keypoint);
				imshow("�һ�֮�����еı�����", show_all_bg_keypoint);
				cvWaitKey(5);
				//***************************************�켣ƽ��****************************************//
				//*************************************��ͨ�˲��㷨****************************************//
				st = cvGetTickCount();
				vector<Point2f> Trj_cor_smooth;
				for(int i=0; i<bg_num; i++)
				{
					Point2f temp_pt;
					for(int j=0; j<gsw_2+1; j++)
						temp_pt += gauss_kernel[j]*Trj_cor_for_smooth[i][j];
					Trj_cor_smooth.push_back(temp_pt);
				}

				et = cvGetTickCount();
				printf("��ͨ�˲�ʱ��: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.);
				//��Trj_cor_for_smooth --> Trj_cor_smooth�ĵ�Ӧ����
				st = cvGetTickCount();
				//����ƽ��
				Mat homo = Mat::zeros(3, 3, CV_32F);
				//���껹ԭ
				for(int i=0; i<bg_num; i++)
				{
					pt_bg_cur[i] *= scale;
					pt_bg_cur[i].y += cropped_start;
					Trj_cor_smooth[i] *= scale;
					Trj_cor_smooth[i].y += cropped_start;
				}
				Mat ransac_outliers = Mat::zeros(1, bg_num, CV_8U);			//�õĽ����1��ʾ��������ģ��ƥ��úã�0Ϊ����
				homo = Homography_Nelder_Mead_with_outliers(pt_bg_cur, Trj_cor_smooth, 200, ransac_outliers, height);///_Mutli_Threads  

				et = cvGetTickCount();
				printf("��Ӧ�������ʱ��: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.);

				st = cvGetTickCount();
				CvMat H = homo;
				std::vector<Point2f> obj_corners(4);
				obj_corners[0] = Point(0,0); obj_corners[1] = Point( nebor_frames[gsw].cols, 0 );
				obj_corners[2] = Point( nebor_frames[gsw].cols, nebor_frames[gsw].rows ); obj_corners[3] = Point( 0, nebor_frames[gsw].rows );
				std::vector<Point2f> scene_corners(4);
				perspectiveTransform(obj_corners, scene_corners, homo);
				//dx��dy�����ж�warp֮���Ƿ���Ҫƽ�ƣ�dx<0��ˮƽƽ�ƣ�dy<0����ֱƽ��
				float dx = (min(scene_corners[0].x, scene_corners[3].x) < 0 ? min(scene_corners[0].x, scene_corners[3].x) : 0);
				float dy = (min(scene_corners[0].y, scene_corners[1].y) < 0 ? min(scene_corners[0].y, scene_corners[1].y) : 0);
				float w_p = (max(scene_corners[1].x, scene_corners[2].x) > width ? max(scene_corners[1].x, scene_corners[2].x) : width) - dx;
				float h_p = (max(scene_corners[2].y, scene_corners[3].y) > height ? max(scene_corners[2].y, scene_corners[3].y) : height) - dy;
				int w_stab = ceil(w_p);
				int h_stab = ceil(h_p);
				IplImage *temp_frame;
				//Mat temp_frame;
				temp_frame = cvCreateImage(cvSize(w_stab, h_stab), frame_ref->depth, frame_ref->nChannels);
				cvWarpPerspective(&(IplImage)nebor_frames[gsw], temp_frame, &H, CV_INTER_LINEAR|CV_WARP_FILL_OUTLIERS, cvScalarAll( 0 ));
				//cvShowImage("͸�ӱ任Ч��", temp_frame);
				//waitKey(0);
				Mat temp_mat(temp_frame);
				//cout<<"���к�: "<<howtocrop<<"\t"<<height-howtocrop<<endl;
				Mat stab_frame = temp_mat(Range(howtocrop_height, height-howtocrop_height), Range(howtocrop_width, width-howtocrop_width));
				//imshow("ԭʼ֡", nebor_frames_crop[gsw]);
				//imshow("�ȶ�֡", stab_frame(Range(cropped_start, cropped_end), Range(30, width-30)));
				//waitKey(0);
				//������Ƶ�����
				cvWriteFrame(Save_result, &IplImage(stab_frame));//(Range(cropped_start/scale, cropped_end/scale), Range(0, width/scale))));
				et = cvGetTickCount();
				printf("�ȶ�֡����ʱ��: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.);
				/******************************************************************************************/
				/***************************************���ƹ켣����******************************************/
				/******************************************************************************************/
				st = cvGetTickCount();
				//�޳�����disappear_tolerance֡û�г��ֵ�֡���ͳ����߽�һ��֡���Ĺ켣
				last_size = Trajectories.size();
				for(int i=0; i<last_size; i++)
				{
					int ln = Trajectories[i].last_number;
					int ln_for_long = ln;
					if(t > reserve_times)
						ln_for_long = reserve_times - (t - ln);
					if(ln <= (t-disappear_tolerance))
					{
						swap(Trajectories[i], Trajectories[last_size-1]);
						Trajectories.pop_back();
						Trj_desc.row(last_size-1).copyTo(Trj_desc.row(i));
						Trj_desc.pop_back();
						/******************************************����last_in_Trj��Trajectories�е�λ��****************************************/
						for(int j=0; j<cur_key_size; j++)
							if(last_in_Trj[j] == last_size-1)//��Ϊ���м��i��Ԫ����ĩβԪ�ضԻ�λ�ã��������һ���޳��������ԣ�ֻ��Ҫ����Ӧ�����һ��Ԫ�ص�cur_key�ı��λ�ø�һ�£�����Ĳ���������һ������
							{
								last_in_Trj[j] = i;
								break;
							}
							last_size--;
					}
					else if((ln <= t-out_range_tolerance)&&((Trajectories[i].trj_cor[ln_for_long-1].x < side_range) || (Trajectories[i].trj_cor[ln_for_long-1].x > crop_width - side_range) || (Trajectories[i].trj_cor[ln_for_long-1].y < side_range) || (Trajectories[i].trj_cor[ln_for_long-1].y > crop_height - side_range)))
					{
						swap(Trajectories[i], Trajectories[last_size-1]);
						Trajectories.pop_back();
						Trj_desc.row(last_size-1).copyTo(Trj_desc.row(i));
						Trj_desc.pop_back();
						/******************************************����last_in_Trj��Trajectories�е�λ��****************************************/
						for(int j=0; j<cur_key_size; j++)
							if(last_in_Trj[j] == last_size-1)//��Ϊ���м��i��Ԫ����ĩβԪ�ضԻ�λ�ã��������һ���޳��������ԣ�ֻ��Ҫ����Ӧ�����һ��Ԫ�ص�cur_key�ı��λ�ø�һ�£�����Ĳ���������һ������
							{
								last_in_Trj[j] = i;
								break;
							}
							last_size--;
					}
					else if((ln < t-total_time_tolerance) && (Trajectories[i].count < total_time_below))
					{
						swap(Trajectories[i], Trajectories[last_size-1]);
						Trajectories.pop_back();
						Trj_desc.row(last_size-1).copyTo(Trj_desc.row(i));
						Trj_desc.pop_back();
						/******************************************����last_in_Trj��Trajectories�е�λ��****************************************/
						for(int j=0; j<cur_key_size; j++)
							if(last_in_Trj[j] == last_size-1)//��Ϊ���м��i��Ԫ����ĩβԪ�ضԻ�λ�ã��������һ���޳��������ԣ�ֻ��Ҫ����Ӧ�����һ��Ԫ�ص�cur_key�ı��λ�ø�һ�£�����Ĳ���������һ������
							{
								last_in_Trj[j] = i;
								break;
							}
							last_size--;
					}
				}
				et = cvGetTickCount();
				printf("�����켣�����͹켣����ά��ʱ��: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.);
				/************************************�������������*************************************/
				bg_trj.push_back(bg_num);
				//if (bg_num > trj_num_higher)
				//{
				//	num_of_corners -= alpha_uper*(bg_num - trj_num_higher);
				//	if(num_of_corners<250)
				//		num_of_corners = 250;
				//	detector = GoodFeaturesToTrackDetector(num_of_corners, pk_thresh, mindist);
				//}
				//else if (bg_num < trj_num_lower)
				//{
				//	num_of_corners += alpha_lower*(trj_num_lower - bg_num);
				//	if (num_of_corners>650)
				//		num_of_corners = 650;
				//	detector = GoodFeaturesToTrackDetector(num_of_corners, pk_thresh, mindist);
				//}
				cout<<"��ǰnum_of_corners: "<<num_of_corners<<endl;
				cout<<"��ǰ��������"<<cur_key.size()<<endl;
				//*****************************************************��ֹ�ڴ�й¶****************************************************//
				cvReleaseImage(&temp_frame);
				cout<<"total trj number is "<<last_size<<endl;
				et = cvGetTickCount();
				printf("�켣��������ģ������ʱ��: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.);
			}
			else
			{
				//�ش��޸İ�������������drawmatchesʱÿ�ζ���ʾͬһ֡������
				//****************************************************��mat������IplImage����ֹ�ڴ�й¶****************************************************//
				Mat crop_cur_copy = crop_cur.clone();
				//��һ���ش��޸ģ�����˴����⣬��ΪcvWarpPerspective������Ҫ����nebor_frames�����ԣ��������Ҳ�������ظ�֡����
				Mat cur_copy(frame_cur, true);
				nebor_frames.push_back(cur_copy);	//δ���е�gsw_2+1֡ʱ��ֱ��ѹ�����
#if MATCH_PAUSE
				nebor_frames_crop.push_back(crop_cur_copy);
#endif
			}
	}
	ofstream bg_trj_file("bg_trj_file.txt");
	int bg_trj_nums = bg_trj.size();
	for (int i=0; i<bg_trj_nums; i++)
	{
		bg_trj_file<<bg_trj[i]<<" ";
	}
	bg_trj_file<<endl;

	cvReleaseVideoWriter(&Save_result);
	return 0;
}