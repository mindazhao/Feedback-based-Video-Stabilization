/*
2015.6.1，第一个稍微可用的版本
2015.6.3，剔除匹配点对中trainIdx重复的点对
2015.6.6，解决显示匹配点对问题时的错误
2015.6.8，调整detector参数，显著提高连续轨迹数
2015.6.10，解决了不能消抖的问题，目前效果已经达到2015.4.8那天的效果
2015.6.12，加入了背景点找回算法，已经调通
2015.6.13，Bundled Paths Optimization算法，并解决了IplImage带来的内存泄露问题
2015.6.17，将findHomography函数换成自己编写的Nelder-Mead算法
2015.6.30，为Homography_from_trj函数多传递一个参数：背景点的徳劳内三角形，但是效果不好！
2015.7.2，6参数变为8参数，且NM算法收敛条件与Matlab版本一致，迭代次数增加，计算单应矩阵参数将float变为float;
修正了Homography_from_trj中的Homography_Nelder_Mead中矩阵A的错误使用;	将所有findHomography全部换成Homography_Nelder_Mead函数,效果已经接近matlab;
2015.7.3，加入自适应模块
2015.7.4，Harris重复个数太少，是匹配函数的问题？当前帧描述符与上一帧的描述符【在Trj_desc中有标记】进行匹配，未匹配上的与Trj_desc匹配
，遇上一个问题：每次经过轨迹数量控制之后，会剔除大量轨迹，那么之前cur_desc在Trj_desc中的标记位置就会有所改变，要想维护它，
就必须将对应于Trajectories末位的那个cur_key的标记位置修改一下，其余的不用改变
2015.7.7，添加了一个新的紧急处理机制：if(continuity_num < 60) minHessian /= 1.05
2015.7.11，感觉bundled paths optimization算法， 不行啊，改用低通滤波看看
2015.7.16，修改了一个bug，就是剔除不再出现的轨迹时候，运算符号中，之前的最后一个<，修改为>
2015.7.17，程序可针对任意长度的视频了！
优化程序，剔除冗余代码段，节省时间
2015.7.17，参照《深入理解计算机系统（第二版）》的优化技巧，进行优化
2015.7.19，光流法+FREAK不行，看来还是只能在Harris上死磕了
已经验证了，4倍降采样不行。。。只好限定Harris点范围了(剪切多一点，或者限定四个角？)
还有单应性约束那一块，要根据帧间单应矩阵来推导，而不是靠find函数计算
2015.8.22，参考ICCV 2005一片论文，将FREAK的匹配函数换成SM算法，由原始匹配算法给出k-近邻匹配结果，用SM算法得出精确匹配，k=1，相对于k=3的时候，效果没有降低太多，而且去掉了作者代码中双随机部分
实验证明，匹配率大大提高
2015.8.25，单帧匹配率上去了，但是连续点数没有增加太多，为此，将图像分成四个部分，单独提取特征点和描述符，最后合称为一个，情况没有改善
2015.8.27，scale=1，针对原分辨率进行处理，情况没有改善，所以，不能怨连续点数越来越少：很多点被误判为前景点了！当然，一开始连续点数就不多，这也是一个问题，但是前一个问题更严重！
这两天要瞅准第一个问题：前景点判定规则和参数调节。
2015.9.7，将ORB特征换为Harris+FREAK，即特征点检测使用Harris，特征提取用FREAK算法，效果还算稳定，但是Harris相对来说太慢了，需要提速，而且后面特征点数量太多了，自适应机制需要调整
2015.9.9，开始做全局优化，简化前景点判定部分
2015.9.12，车山点太集中，仍然尝试用分块的思路,分三块，对三个块的特征点数量都进行合理控制
2015.9.22，多线程加速前景点判定模块，时间降一半
2015.9.24，SM算法中多线程计算帧内距离
2015.11.1，改为Harris+FREAK，大大增加了程序的鲁棒性
2015.12.15，找回背景轨迹时候，不作徳劳内三角形找相邻轨迹，而是与所有背景轨迹进行相似性比较
					背景轨迹奖励、前景轨迹惩罚
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
#include"Homography_from_trj_backup_2016年7月4日写论文版本【调好了参数】.h"
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

//关键段（临界锁）变量声明  
//CRITICAL_SECTION  g_csThreadParameter, g_csThreadCode;  
const int reserve_times = 120;	//运行到500帧的时候，开始剔除最前面的一些轨迹
vector<Point2f> pt_init_1, pt_init_2;
vector<KeyPoint>key_init;
Mat homo_init = Mat::zeros(3, 3, CV_32F);
// 轨迹结构体，保存轨迹的属性和轨迹坐标
typedef class trajectory
{
public:
	unsigned int count;						//轨迹出现的次数
	int last_number;				//上一次出现的帧号
	unsigned int continuity;					//连续出现的帧数
	unsigned int foreground_times;		//被判为前景轨迹的次数
	unsigned int background_times;		//被判为背景轨迹次数
	unsigned int award;					//背景点奖励
	unsigned int penalization;				//前景点惩罚
	deque<Point2f> trj_cor;			//轨迹坐标
	//	Mat descriptors1;							//特征描述符，应放在一大块内存中
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

//设计用于计算单应矩阵的并行化
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
//设计用于计算帧内距离的并行化
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
//设计用于提取Harris角点的并行化
typedef class para_for_harris
{
public:
	vector<KeyPoint> cur_key;
	Mat img;
	GoodFeaturesToTrackDetector cur_detector;
	int up;			//第几行
	int colum;	//第几列
	float delta_y;	//纵向偏差
	float delta_x;	//横向偏差
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

//调用单应矩阵计算函数的多线程函数，要创建多个线程
unsigned int __stdcall Calculate_Homography(void *para_pt)  
{  
	//LeaveCriticalSection(&g_csThreadParameter);//离开子线程序号关键区域 
	para_for_homo *these_pt = (para_for_homo*)para_pt;
	//由于创建线程是要一定的开销的，所以新线程并不能第一时间执行到这来  
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
//调用单应矩阵计算函数的多线程函数，要创建多个线程
unsigned int __stdcall Calculate_InFrameDistance(void *para_key)  
{  
	PARA_FOR_INFRAMEDIST *these_pt = (PARA_FOR_INFRAMEDIST*)para_key;
	//由于创建线程是要一定的开销的，所以新线程并不能第一时间执行到这来  
	int n_node = these_pt->cur_key.size();
	int n_node_1_2 = n_node / 2;
	vector<KeyPoint>cur_key = these_pt->cur_key;
	Mat Dij_1 = these_pt->Dist;//两个线程中会分别赋值给不同的局部变量，但是是指向同一个全局变量Dij_1，会不会造成问题？？隐患？？？
	float temp = 0.f;
	if(these_pt->up_left)
	{
		for(int i=0; i<n_node_1_2; i++)
		{
			for(int j=i+1; j<n_node-i; j++)	//改进，因为下面一次性赋两个值，所以循环次数也可以减半，之前这里仍然是从j=0开始的
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
			for(int i=n_node-j; i<=j; i++)	//改进，因为下面一次性赋两个值，所以循环次数也可以减半，之前这里仍然是从j=0开始的
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
//调用单应矩阵计算函数的多线程函数，要创建多个线程
unsigned int __stdcall Detect_Harris(void *para_pt)  
{  
	//LeaveCriticalSection(&g_csThreadParameter);//离开子线程序号关键区域 
	para_for_harris *cur_detect = (para_for_harris*)para_pt;
	cur_detect->cur_detector.detect(cur_detect->img, cur_detect->cur_key);
	//进行坐标复原
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
//比较算法
bool compare1(const DMatch &d1,const  DMatch &d2)
{
	return d1.trainIdx < d2.trainIdx;
}
//计算平均值
float mean(const deque<int> trj_num)
{
	float sum = 0;
	int trjs = trj_num.size();
	for(int i=0; i<trjs; i++)
		sum += trj_num[i];
	sum /= trjs;
	return sum;
}
//计算标准差，除数为(N-1)
float std_val(const deque<int> trj_num, float the_mean)
{
	float std_var = 0;
	int trjs = trj_num.size();
	for(int i=0; i<trjs; i++)
		std_var += (trj_num[i] - the_mean) * (trj_num[i] - the_mean);
	std_var /= (trjs - 1);
	return sqrt(std_var);
}
// 计算汉明距离
unsigned int hamdist2(unsigned char* a, unsigned char* b, size_t size) 
{ 
	HammingLUT lut;  
	unsigned int result;
	result = lut((a), (b), size); 
	return result; 
}  
// 该函数只用到Trj_keys.size()和cur_key.size()函数，只用到二者的长度!
// 为了更高效，应单独设计一个vector存放所有的描述符
void naive_nn_search(Mat& descp1, Mat& descp2, vector<DMatch>& matches) 
{
	//vector<unsigned int> matched_cur, matched_Trj;	//保存匹配上的点对序号
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
			unsigned int dist =  hamdist2(query_feat, train_feat, 64); //修正了一个严重的错误！这个是64位的FREAK算子！
			//最短距离
			if(dist < min_dist)
			{ 
				sec_dist = min_dist; 
				sec_idx = min_idx;
				min_dist = dist; 
				min_idx = j; 
			}
			//次短距离
			else if(dist < sec_dist)
			{ 
				sec_dist = dist; 
				sec_idx = j; 
			} 
		}   
		if(min_dist <=50 && min_dist <= 0.8*sec_dist)//min_dist <= (unsigned int)(sec_dist * 0.7) && min_dist <=100
		{
			//若不处理，则会有许多重复的匹配对！
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
// 该函数只用到Trj_keys.size()和cur_key.size()函数，只用到二者的长度!
// 为了更高效，应单独设计一个vector存放所有的描述符
void naive_nn_search2(vector<KeyPoint>& Trj_keys, Mat& descp1, vector<KeyPoint>& cur_key, Mat& descp2, vector<vector<DMatch>>& matches, const int max_shaky_dist, int k) 
{
	//vector<unsigned int> matched_cur, matched_Trj;	//保存匹配上的点对序号
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
			unsigned int dist =  hamdist2(query_feat, train_feat, 64); //修正了一个严重的错误！这个是64位的FREAK算子！
			float Trj_key_x = Trj_keys[j].pt.x;
			float Trj_key_y = Trj_keys[j].pt.y;
			//匹配点坐标距离限制
			if((cur_key_x - Trj_key_x)*(cur_key_x - Trj_key_x) + (cur_key_y - Trj_key_y)*(cur_key_y - Trj_key_y) < max_shaky_dist)
			{
				//最短距离
				if(dist < min_dist)
				{ 
					thr_dist = sec_dist;
					thr_idx = sec_idx;
					sec_dist = min_dist; 
					sec_idx = min_idx;
					min_dist = dist; 
					min_idx = j; 
				}
				//次短距离
				else if(dist < sec_dist)
				{ 
					thr_dist = sec_dist;
					thr_idx = sec_idx;
					sec_dist = dist; sec_idx = j; 
				} 
				//次短距离
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
//快排
void quick_sort(Mat v, vector<Point> &L, int l, int r)
{  
	if (l < r)
	{
		int i = l, j = r;
		float x = ((float*)v.data)[l];
		Point temp = L[l];
		while (i < j)
		{
			while(i < j && ((float*)v.data)[j] <= x) // 从右向左找第一个小于x的数  
				j--;
			if(i < j)
			{
				((float*)v.data)[i++] = ((float*)v.data)[j];
				L[i-1] = L[j];
			}

			while(i < j && ((float*)v.data)[i] > x) // 从左向右找第一个大于等于x的数  
				i++;
			if(i < j)
			{
				((float*)v.data)[j--] = ((float*)v.data)[i];
				L[j+1] = L[i];
			}
		}
		((float*)v.data)[i] = x;
		L[i] = temp;
		quick_sort(v, L, l, i - 1); // 递归调用   
		quick_sort(v, L, i + 1, r);
	}
}


//my_spectral_matching，根据自己编写的matlab函数编写
void my_spectral_matching(vector<KeyPoint> &cur_key, vector<KeyPoint> &last_key, vector<vector<DMatch>> &matches, int &k, vector<DMatch> &X_best)
{
	int64 st, et;
	//st = cvGetTickCount();
	int n_node = cur_key.size();
	int n_label = last_key.size();
	vector<int> start_ind_for_node(n_node);	//每个拥有有效匹配对的当前特征点在L中的起始索引值
	int n_matches = 0;	//有效匹配对个数
	vector<Point> L;	//所有的候选匹配对
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
	//printf("生成L矩阵时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 
	////%% 计算M的对角线元素
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
	//printf("生成M矩阵对角线元素时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 

	////先计算每帧内每个i,j之间的坐标距离
	//st = cvGetTickCount();
	Mat Dij_1 = Mat::zeros(n_node, n_node, CV_32F);
	Mat Dij_2 = Mat::zeros(n_label, n_label, CV_32F);
	float temp = 0;

	//多线程版本
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

	WaitForMultipleObjects(THREAD_NUM, handle, TRUE, INFINITE);//INFINITE);//至多等待20ms
	for (int i=0; i<4; i++)
		CloseHandle(handle[i]);
	//et = cvGetTickCount();
	//printf("计算帧内距离时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 

	//计算M矩阵的非对角线元素
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
	//printf("生成M矩阵非对角线元素时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 

	//%% spectral matching算法
	//st = cvGetTickCount();
	Mat v = Mat::ones(n_matches, 1, CV_32F);
	//float x = norm(v);
	v = v/norm(v);
	int iterClimb = 20;//之前取30，耗时太长，取20应该也可以，反正后面也会收敛了

	// 幂法计算最大特征值（及其）对应的特征向量
	for(int i = 0; i<iterClimb; i++)
	{
		v = M*v;
		v = v/norm(v);
	}
	//et = cvGetTickCount();
	//printf("幂法计算主特征向量时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 
	//贪心策略求出最优匹配
	//st = cvGetTickCount();
	//vector<DMatch> X_best;
	quick_sort(v, L, 0, n_matches-1);
	//et = cvGetTickCount();
	//printf("快速排序时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 
	float max_v = 0;
	//DMatch best_match;
	//st = cvGetTickCount();
	bool *conflict = new bool[n_matches];	//标记每个匹配对是否与当前最好的匹配对冲突
	for (int i=0; i<n_matches; i++)
		conflict[i] = false;
	int left_matches = n_matches;
	float dist = 10.0;
	while(left_matches)
	{
		int i=0;
		while(conflict[i]) i++;	//找到第一个未冲突的最大值项
		max_v = ((float*)v.data)[i];
		DMatch best_match = DMatch(L[i].x, L[i].y, 0, float(dist));
		X_best.push_back(best_match);
		//找出所有与best_match冲突的匹配对，剔除之
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
	//printf("贪心策略时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 
}
/*
pts，要剖分的散点集,in
img,剖分的画布,in
tri,存储三个表示顶点变换的正数,out
*/
// used for doing delaunay trianglation with opencv function
//该函数用来防止多次重画并消去虚拟三角形的顶点
bool isGoodTri( Vec3i &v, vector<Vec3i> & tri ) 
{
	int a = v[0], b = v[1], c = v[2];
	v[0] = min(a,min(b,c));//v[0]找到点插入的先后顺序（0....N-1，N为点的个数）的最小值
	v[2] = max(a,max(b,c));//v[2]存储最大值.
	v[1] = a+b+c-v[0]-v[2];//v[1]为中间值
	if (v[0] == -1) return false;

	vector<Vec3i>::iterator iter = tri.begin();//开始时为空
	for(;iter!=tri.end();iter++)
	{
		Vec3i &check = *iter;//如果当前待压入的和存储的重复了，则停止返回false。
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
//二分查找，因为foreground_index是从小到大排好序的
int binary_search(vector<int>a, int goal)
{
	if(a.size()==0)
	{
		cout<<"传入vector是空的，查个屁啊"<<endl;
		return -1;
	}
	int low = 0;
	int high = a.size() - 1;
	while(low <= high)
	{
		int middle = (low + high)/2;
		if(a[middle] == goal)
			return middle;
		//在左半边
		else if(a[middle] > goal)
			high = middle - 1;
		//在右半边
		else
			low = middle + 1;
	}
	//没找到
	return -1;
}
vector<float> MyGauss( int _sigma )
{
	int width = 2 * _sigma+1;
	if ( width<1 )
	{
		width = 1;
	}


	/// 设定高斯滤波器宽度
	int len = width;

	/// 高斯函数G
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
//FAST阈值比较函数
bool fast_thresh_comp(const KeyPoint &corner1, const KeyPoint &corner2)
{
	return corner1.response > corner2.response;
}
//限制两个特征点距离的FAST特征检测算法
void MyFAST(Mat& image, vector<KeyPoint>& corners, int maxCorners, double qualityLevel, double minDistance, Mat & mask, bool have_mask)
{
	vector<KeyPoint>FAST_corners;
	//基本FAST检测
	FAST(image, corners, qualityLevel);

	//在Mask基础上生成一张掩码表
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
	//贪心算法，筛选
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
//argv格式：程序名 视频文件名 竖直方向检测特征点起点线y坐标 竖直方向检测特征点终点线y坐标 帧率
int main(int argc, char* argv[])
{
	//检查输入参数个数
	if (argc < 3)
	{
		cout<<"输入参数格式："<<endl;
		cout<<"程序名 视频文件名 输出文件夹"<<endl;
		return -1;
	}
	//打开视频文件
	string filename = string(argv[1]);
	int the_start_num = filename.find_last_of("/");
	string the_name = filename.substr(the_start_num+1);//, filename.length()-4);
	the_name = the_name.substr(0, the_name.length()-4);

	//char* openfile="E://视频去抖//测试视频//18_failure_train.avi";//bg_motion_2, on_road_3，on_road_4，example4_car_input，8原视频
	char *openfile = &filename[0];
	CvCapture* pCapture=cvCreateFileCapture(openfile);
	if(pCapture==NULL)
	{
		cout<<"video file open error!"<<endl;
		return -1;
	}
	string outfilename = string(argv[2]) + string("/proof_") + the_name + string("_Version.avi");
	char* outfile=&outfilename[0];
	//获取视频相关信息，帧率和大小
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
	//视频帧缩放前尺寸，缩放比例
	int width = frame_ref->width;
	int height = frame_ref->height;

	
	//输出为切边后的视频，上下左右各切边40个像素
	int howtocrop_width = 0;//(float)width*60/1280;
	int howtocrop_height = 0;//(float)height*40/540;
	cout<<"howtocrop: "<<howtocrop_width<<endl;
	CvSize size=cvSize((int)cvGetCaptureProperty(pCapture,CV_CAP_PROP_FRAME_WIDTH)-howtocrop_width*2,
		(int)cvGetCaptureProperty(pCapture,CV_CAP_PROP_FRAME_HEIGHT)-howtocrop_height*2);

	//创建输出视频文件
	CvVideoWriter* Save_result=NULL;
	Save_result=cvCreateVideoWriter(outfile,CV_FOURCC('X','V','I','D'),fps,size,1);

	float scale = 1;

	int cropped_start = 0;//96;
	int cropped_end = height;//640

	const float crop_width = width/scale;
	const float crop_height = (cropped_end-cropped_start)/scale;
	const float height_1_2 = crop_height/2;
	const float width_1_4 = crop_width/4, width_2_4 = crop_width/2, width_3_4 = 3*crop_width/4;
	cout<<"放缩"<<scale<<"倍之后处理"<<endl;
	CvSize after_size = cvSize(width/scale, height/scale);
	gray = cvCreateImage(cvGetSize(frame_ref), frame_ref->depth, 1);
	dst = cvCreateImage(after_size, frame_ref->depth, 1); 
	// 参考帧转为灰度图
	cvCvtColor(frame_ref, gray, CV_BGR2GRAY);
	cvResize(gray, dst);
	Mat object(dst);//不拷贝矩阵数据，只拷贝矩阵头
	Mat crop_ref = object(Range(cropped_start/scale, cropped_end/scale), Range(0, width/scale));
	//使用感兴趣区域来找特征点，而不是剪切掉
	Mat mask(crop_ref.size(), CV_8U, Scalar(255));
	//rectangle(mask, Point(1050/scale, 31/scale), Point(1280/scale, 81/scale), Scalar(0), -1, CV_8U);
	rectangle(mask, Point(40/scale, 35/scale), Point(774/scale, 96/scale), Scalar(0), -1, CV_8U);
	rectangle(mask, Point(922/scale, 644/scale), Point(1178/scale, 684/scale), Scalar(0), -1, CV_8U);
	//****************************************************************************************************//
	//******************************************关键的数据结构设计*********************************************//
	vector<KeyPoint> Trj_keys, ref_key_1, ref_key_2, ref_key_3, ref_key_4, ref_key_5, ref_key_6;//, ref_key_7, ref_key_8, ref_key_9;				//存放第一帧、当前帧的关键点
	vector<KeyPoint> cur_key, cur_key_1, cur_key_2, cur_key_3, cur_key_4, cur_key_5, cur_key_6;//, cur_key_7, cur_key_8, cur_key_9;
	Mat Trj_desc;//, Trj_desc_1, Trj_desc_2, Trj_desc_3, Trj_desc_4, Trj_desc_5, Trj_desc_6, Trj_desc_7, Trj_desc_8, Trj_desc_9;						//**************重要！！！存放所有轨迹的描述符、当前描述符******************//
	Mat cur_descriptor;//, cur_desc_1, cur_desc_2, cur_desc_3, cur_desc_4, cur_desc_5, cur_desc_6, cur_desc_7, cur_desc_8, cur_desc_9;

	int64 st, et;
	//const int reserve_times = 500;	//运行到500帧的时候，开始剔除最前面的一些轨迹
	unsigned int num_of_corners = 3000;						//Surf阈值，降低，会提高Surf点数量
	unsigned int last_size = 0;
	const unsigned char gsw = 5;	//滤波窗口大小8
	const unsigned char gsw_2 = gsw*2;
	const int max_dist = 1800;
	float max_shaky_dist = max_dist;							//一般来说，最大抖动量不会超过30*30，cur_dist<30*30+30*30=1800
	int penalization_relax = 0;							//用于恢复惩罚机制到初始状态
	int penalization_thresh = 1;							//惩罚阈值
	const unsigned int good_match_tolerance = INT_MAX;				//最大允许的匹配向量误差
	const unsigned char max_H_cnstrt_tolerance = 25/scale;		//利用Homography Constraint判别前景点时的阈值
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

	const unsigned char H_cnstrt = 1;					//单应矩阵约束
	const int disappear_tolerance = reserve_times;//70;	//连续70帧不出现就剔除之
	const int out_range_tolerance = 20;	//离开边界、未出现次数的容忍度
	const float side_range = 10;				//判断是否接近边界的阈值，抖动最大值可以达到30+像素
	const int total_time_tolerance = 20;			//出现次数很少的轨迹将在30次后清除
	const int total_time_below = 5;					//出现次数少于15次的轨迹将被清除
	const unsigned char retrieval_num = 100;	//轨迹找回个数阈值，用于找回轨迹
	const unsigned char similarity_tolerance = 120;	//轨迹相似性阈值，用于找回轨迹
	vector<Mat> Trans_between(gsw);				//t-gsw+1帧与t-gsw帧之间的单应矩阵，可能要修改为当前帧与相邻gsw_2帧之间的单应矩阵，避免在前景轨迹判断时重复计算
	vector<Mat> Trans_between_temp(gsw);				//t-gsw+1帧与t-gsw帧之间的单应矩阵，可能要修改为当前帧与相邻gsw_2帧之间的单应矩阵，避免在前景轨迹判断时重复计算

	vector<unsigned int> DT;								//徳劳内三角形的三个顶点集合
	vector<float> Dist;										//保存每个前景点相对于其各个邻居背景点间的距离
	unsigned char trj_num_lower = 55;					//控制背景轨迹数不要太少，这两个参数反映了视频的运行速度
	unsigned char trj_num_higher = 110;					//控制背景轨迹数不要太多，免得FREAK阈值降低至不能再少（此时将不再能控制FREAK点数和轨迹数）
	deque<int> bg_trj;										//记录近500帧的背景轨迹数量变化情况，调试自适应部分使用
	float alpha_uper = 0.05;										//背景轨迹数量过多时候，需要减少一部分特征点
	float alpha_lower = 0.1;									//背景轨迹数量过少时候，需要增加一部分特征点
	unsigned char lowhigh_times_to_alter = 20;		//每隔20帧修改一次lower和higher阈值
	unsigned char sub_size = 6;
	deque<int> trj_num_bg;							//保存近20帧连续出现的背景轨迹数目
	unsigned char quick_pk_th = 10;					                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           	//针对快速视频要设置阈值，每隔一定帧数修改pk_thresh值
	vector<Point2f> Trj_bg_shorter;						//轨迹连续帧数不足gsw_2+1但是大于gsw+1的背景轨迹坐标
	const unsigned char lambda = 100;					//bundled paths Optimization参数
	const float beta = 0;									//二阶导数项的加权系数
	const float w_nebor = 0.5;							//相邻轨迹的加权系数，原始的系数为2，太大了，此处改为2*w_nebor
	deque<Mat>  nebor_frames;							//将相邻的帧组织成双边队列的形式
#if MATCH_PAUSE
	deque<Mat>  nebor_frames_crop;							//将相邻的帧组织成双边队列的形式
#endif
	int show_start_num = 3;								//从第几帧开始显示特征点、前景判定过程
	float pk_thresh = 0.00001;									//控制FREAK点数
	int mindist = 6;												//两个角点间最小距离
	const unsigned char level = 3;							//金字塔层数
	const unsigned char Octaves = 2;						//金字塔个数
	const unsigned char blur_size = 3;						//图像平滑窗口大小
	////参考帧提取FREAK
	//st = cvGetTickCount();
	//GaussianBlur(crop_ref, crop_ref, Size(blur_size, blur_size), 0);
	//et = cvGetTickCount();
	//printf("高斯模糊时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 

	// DETECTION
	// Any openCV detector such as
#if USE_SURF
	SurfFeatureDetector detector(num_of_corners, Octaves, level);
#else
	GoodFeaturesToTrackDetector detector(num_of_corners, pk_thresh, mindist, 3, 1);//因为要分块，所以，需要抑制中间那一个块的特征点数
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

	// 轨迹属性矩阵再初始化就赋值为0
	vector<Trj> Trajectories(Trj_keys.size());			//****************重要！！！！存放所有轨迹的属性值*****************//
	//给轨迹属性矩阵赋值
	int Trj_keys_size = Trj_keys.size();
	for(int i=0; i<Trj_keys_size; i++)
		Trajectories[i].trj_cor[0] = Trj_keys[i].pt;

	// 帧号
	int t = 1;
	//将第一帧压入队列
	nebor_frames.push_back(Mat(frame_ref));
	Mat crop_ref_copy = crop_ref.clone();
#if MATCH_PAUSE
	nebor_frames_crop.push_back(crop_ref_copy);
#endif
	// 标记当前特征点中哪些已经被旧的轨迹匹配上
	bool *Surf_index = NULL;

	//记录上一帧匹配上的点在Trajectories中的位置
	vector<int> last_in_Trj;
	vector<KeyPoint> last_key = Trj_keys;
	Mat last_desc = Trj_desc.clone(), last_crop = crop_ref.clone();
	stitch_image = cvCreateImage(cvSize(width*2, height), frame_ref->depth, frame_ref->nChannels);

	//窗口命名
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
	namedWindow("找回之后，所有的背景点", WINDOW_NORMAL); 
	resizeWindow("找回之后，所有的背景点", 720, 480);
	string winName = "Matches";
	namedWindow( winName, WINDOW_NORMAL );
	resizeWindow(winName, 2240, 630);

	// 函数主体
	while(frame_cur = cvQueryFrame(pCapture))
	{
		t++;
		cout<<t<<endl;
		//先不处理帧号太大的视频帧
		// 参考帧转为灰度图
		//cur_frame = Mat(frame_cur);
		st = cvGetTickCount();
		cvCvtColor(frame_cur, gray, CV_BGR2GRAY);
		cvResize(gray, dst);
		Mat sne(dst);//不拷贝矩阵数据，只拷贝矩阵头
		Mat crop_cur = sne(Range(cropped_start/scale, cropped_end/scale), Range(0, width/scale));
		et = cvGetTickCount();
		printf("图像剪切时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 
		////提取FREAK特征;
		//st = cvGetTickCount();
		//GaussianBlur(crop_cur, crop_cur, Size(blur_size, blur_size), 0);
		//et = cvGetTickCount();
		//printf("高斯模糊时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 

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
		//提取当前帧FREAK描述符
		//匹配上旧的轨迹的FREAK点的索引值
		st = cvGetTickCount();
		delete []Surf_index;
		Surf_index = new bool[cur_key.size()];
		int cur_key_size = cur_key.size();
		for(int i=0; i<cur_key_size; i++)
			Surf_index[i] = false;

		vector<DMatch> matches;
		int this_cor_number = t > reserve_times+1 ? reserve_times+1 : t;	//******************************表示，帧数是否超过50，对于长视频，特别重要！！！！！！***********************************//
		int last_in_Trj_size = last_in_Trj.size();
		if(!last_in_Trj.size())
		{
			vector<vector<DMatch>> matches_for_SM(cur_key.size());
			//vector<DMatch> matches_for_SM(cur_key.size());
			//非空，则将当前描述符与上一帧的描述符匹配
			int k=2;
			// match
			st = cvGetTickCount();
			naive_nn_search2(last_key, last_desc, cur_key, cur_descriptor, matches_for_SM, max_shaky_dist, k); 
			et = cvGetTickCount();
			printf("matching time: %f\n ", (et-st)/(float)cvGetTickFrequency()/1000.); 

			st = cvGetTickCount();
			my_spectral_matching(cur_key, last_key, matches_for_SM, k, matches);
			et = cvGetTickCount();
			printf("My_Spectral_Matching算法匹配时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 
			cout<<"Spectral matching匹配上了"<<matches.size()<<"个"<<endl;
		}
		else
		{
			vector<vector<DMatch>> matches_for_SM(cur_key.size());
			vector<DMatch>matches_temp;
			//非空，则将当前描述符与上一帧的描述符匹配
			int k=2;
			naive_nn_search2(last_key, last_desc, cur_key, cur_descriptor, matches_for_SM, max_shaky_dist, k); 
			cout<<matches_for_SM.size()<<endl;
			//ofstream m_file("matches_for_SM.txt");
			//m_file<<matches_for_SM<<endl;
			st = cvGetTickCount();
			my_spectral_matching(cur_key, last_key, matches_for_SM, k, matches_temp);
			et = cvGetTickCount();
			printf("My_Spectral_Matching算法匹配时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 
			cout<<"Spectral matching,跟上一帧匹配上了"<<matches_temp.size()<<"个"<<endl;
			//如果匹配率太低，就放宽阈值重新匹配，并且放宽惩罚机制
			int match_times = 1;
			while (matches_temp.size()<num_of_corners/2-100 && match_times<5)
			{
				match_times ++;
				max_shaky_dist+=800;
				for (int i=0; i<cur_key_size; i++)
					matches_for_SM[i].clear();
				//不惩罚前景轨迹
				//penalization_thresh = 10;
				//penalization_relax = 10;
				matches_temp.clear();
				naive_nn_search2(last_key, last_desc, cur_key, cur_descriptor, matches_for_SM, max_shaky_dist, k); 
				cout<<matches_for_SM.size()<<endl;
				st = cvGetTickCount();
				my_spectral_matching(cur_key, last_key, matches_for_SM, k, matches_temp);
				et = cvGetTickCount();
				printf("My_Spectral_Matching算法匹配时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 
				cout<<"Spectral matching,跟上一帧匹配上了"<<matches_temp.size()<<"个"<<endl;
			}
			cout<<"当前匹配阈值"<<max_shaky_dist<<endl;
			max_shaky_dist = max_dist;
			//匹配完成后阈值缩小一点???
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
			//匹配结果转换
			int matches_temp_size = matches_temp.size();
			//cout<<"跟上一帧匹配上了"<<matches_temp_size<<"个点"<<endl;
			for(int i=0; i<matches_temp_size; i++)
				matches_temp[i].trainIdx = last_in_Trj[matches_temp[i].trainIdx];

			//记录未匹配上的描述符
			vector<bool> mached_cur(cur_key.size());
			mached_cur.assign(cur_key.size(), false);
			for(int i=0; i<matches_temp_size; i++)
				mached_cur[matches_temp[i].queryIdx] = true;//标记，cur_key中的这个值已经被匹配上
			//cur_key中剩下的点
			if(cur_key_size - matches_temp_size)
			{
				vector<int> left_in_cur(cur_key.size()-matches_temp.size());
				vector<KeyPoint> left_cur_key;
				vector<DMatch> matches_left;//未匹配上的，则与Trj_desc中余下的匹配
				int left_num = 0;
				//记录未匹配上的描述符，包括cur_key和Trj_desc
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
				//余下的跟Trj_desc匹配
				naive_nn_search(Trj_desc, left_desc, matches_left); 
				//匹配结果转换
				int matches_left_size = matches_left.size();
				for(int i=0; i<matches_left_size; i++)
					matches_left[i].queryIdx = left_in_cur[matches_left[i].queryIdx];

				//剔除重复的
				//相连
				matches_temp.insert(matches_temp.end(), matches_left.begin(), matches_left.end());
				//按照第二个量排序排序
				sort(matches_temp.begin(), matches_temp.end(), compare1);
				matches_temp_size = matches_temp.size();
				//排序之后，trainIdx最多有连续两个重复的
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
		printf("特征点匹配时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.);
		//****************************************记录上一帧的特征点描述符及其在Trajectories中的位置**************************************************//
		st = cvGetTickCount();
		last_desc = cur_descriptor.clone();
		last_key = cur_key;
		last_crop = crop_cur.clone();
		last_in_Trj.resize(cur_key_size);
		last_in_Trj.assign(cur_key_size, -1);
		int matches_size = matches.size();
		//下面两项用于显示当前的匹配点及其在Trajectories中的编号
		vector<KeyPoint> cur_matched_key(matches_size);
		vector<int> cur_matched_key_index_Trj(matches_size);
		for(int i=0; i<matches_size; i++)
		{
			last_in_Trj[matches[i].queryIdx] = matches[i].trainIdx;
			cur_matched_key[i] = cur_key[matches[i].queryIdx];
			cur_matched_key_index_Trj[i] = matches[i].trainIdx;
		}

		//轨迹匹配添加
		last_size = Trj_desc.rows;
		for(int i=0; i<matches_size; i++)
		{
			Surf_index[matches[i].queryIdx] = true;
			Trajectories[matches[i].trainIdx].count += 1;
			Trajectories[matches[i].trainIdx].continuity += 1;
			Trajectories[matches[i].trainIdx].last_number = t;
			if (Trajectories[matches[i].trainIdx].penalization)
				Trajectories[matches[i].trainIdx].penalization--;
			//前景点再出现，予以惩罚
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
				Trajectories[matches[i].trainIdx].trj_cor.push_back(cur_key[matches[i].queryIdx].pt); //最后t要改为t_after = t > 1000? t-1000:t
			}
			else
				Trajectories[matches[i].trainIdx].trj_cor[t-1] = cur_key[matches[i].queryIdx].pt; //最后t要改为t_after = t > 1000? t-1000:t
			cur_descriptor.row(matches.at(i).queryIdx).copyTo(Trj_desc.row(matches.at(i).trainIdx));	////由于Harris检测不具有尺度不变性，所以，此处有必要在每次更新特征向量
		}
		//未出现的轨迹的continuity=0
		for(int i=0; i<last_size; i++)
			if(Trajectories[i].last_number != t)
			{
				Trajectories[i].continuity = 0;
				if(t > reserve_times)
				{
					Trajectories[i].trj_cor.pop_front();
					Trajectories[i].trj_cor.push_back(Point2f(0, 0));
				}
				//对待前景轨迹还是不能松懈
				//if (Trajectories[i].award)
				//	Trajectories[i].award--;
			}
			//新出现的轨迹添加
			for(int i=0; i<cur_key_size; i++)
			{
				//cout<<i<<endl;
				if(!Surf_index[i])
				{
					Trajectories.push_back(Trj(1, t, 1, 0, 0, 0, reserve_times));
					Trj_desc.push_back(cur_descriptor.row(i));
					//****************************************记录上一帧的特征点描述符及其在Trajectories中的位置**************************************************//
					int Trajectories_size = Trajectories.size();
					last_in_Trj[i] = Trajectories_size-1;
					if(t > reserve_times)
					{
						Trajectories[Trajectories_size-1].trj_cor.pop_front();
						Trajectories[Trajectories_size-1].trj_cor.push_back(cur_key[i].pt); //最后t要改为t_after = t > 1000? t-1000:t
					}
					else
						Trajectories[Trajectories_size-1].trj_cor[t-1] = cur_key[i].pt; //最后t要改为t_after = t > 1000? t-1000:t
				}
			}
			et = cvGetTickCount();
			printf("轨迹添加和轨迹矩阵维护时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.);
			//最终用于滤波的轨迹坐标矩阵
			vector<vector<Point2f>>Trj_cor_for_smooth;
			//连续gsw_2+1帧都出现，且没有被判定为前景点超过H_cnstrt次
			if (t >= gsw_2+1)
			{
				//重大修改啊！！！修正了drawmatches时每次都显示同一帧的问题
				//对于nebor_frames，也是同样的问题
				//*****************************************************用mat而不是IplImage，防止内存泄露防止内存泄露****************************************************//
				st = cvGetTickCount();
				Mat crop_cur_copy = crop_cur.clone();
				Mat cur_copy(frame_cur, true);
				//又一个重大修改，解决了大问题，因为cvWarpPerspective函数需要调用nebor_frames，所以，这个队列也不能有重复帧出现！妈的！搞死老子了！
				//环形队列，队首出队，新帧压入队尾
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
				printf("相邻帧储存队列维护时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.);
				//查看FREAK特征点的匹配效果
				st = cvGetTickCount();
				Mat img_matches;
				//用于筛选的轨迹坐标矩阵
				vector<vector<Point2f>>Trj_cor_continuity;
				vector<KeyPoint> cur_continuity_key;
				unsigned int num_trj_for_H_cnstrt = 0;		//*****仅仅用于单应矩阵判别前景点程序段中******
				vector<int>continuity_index;				//连续gsw_2+1帧出现，用于轨迹找回
				unsigned int continuity_num = 0;			//连续出现的轨迹数
				vector<int>H_cnstrt_index;				//连续出现的背景点在continuity_index中的序号
				vector<int>foreground_index;				//连续出现的前景点
				int foreground_num = 0;		//前景点计数
				vector<bool>continuity_fg;				//该连续是否前景点，用于背景点找回程序中
				this_cor_number = t > reserve_times ? reserve_times : t;	//******************************在此时，this_cor_number应更新一下。该参数对于长视频，特别重要！！！！！！***********************************//
				vector<int>fg_bg_times;
				for(int i=0; i<matches_size; i++)//由于要求连续出现，那么只要当前帧中匹配上的轨迹就行了
				{
					//前景轨迹惩罚机制
					if(Trajectories[matches[i].trainIdx].continuity >= gsw_2+1)// && Trajectories[matches[i].trainIdx].penalization < penalization_thresh)
					{
						//*****************************重要*********************************后面的所有索引值，都跟continuity_index有关系
						continuity_index.push_back(matches[i].trainIdx);
						continuity_num++;
						//先将坐标放入Trj_cor_continuity
						vector<Point2f> temp;
						continuity_fg.push_back(false);
						for(int j=gsw_2; j >= 0; j--)
						{
							Point2f tem_pt(Trajectories[matches[i].trainIdx].trj_cor[this_cor_number-j-1].x, Trajectories[matches[i].trainIdx].trj_cor[this_cor_number-j-1].y);//以后要改为i%500?
							temp.push_back(tem_pt);
						}
						Trj_cor_continuity.push_back(temp);
						//连续关键点压入vector
						cur_continuity_key.push_back(cur_key[matches[i].queryIdx]);
						//***********连续gsw_2+1帧出现并且被判为前景轨迹次数不超过H_cnstrt次*************//
						if(Trajectories[matches[i].trainIdx].foreground_times < H_cnstrt)
						{
							H_cnstrt_index.push_back(continuity_num-1);
							num_trj_for_H_cnstrt++;
						}
						else //if(Trajectories[matches[i].trainIdx].foreground_times >= H_cnstrt)	//已经被判为前景，将其坐标收入向量中
						{
							fg_bg_times.push_back(Trajectories[matches[i].trainIdx].background_times);
							continuity_fg[continuity_num-1] = true;
							foreground_num++;
							foreground_index.push_back(continuity_num-1);//***************重要***************保存在continuity_index中排列的序号【而不是在Trajectories中的序号】，好寻址啊！！！！！！！
						}
					}
					//else if(Trajectories[matches[i].trainIdx].continuity >= gsw_2+1 && Trajectories[matches[i].trainIdx].foreground_times >= 6)
					//	cout<<"糟糕的前景点！！！！！！！！"<<matches[i].trainIdx<<"\t"<<Trajectories[matches[i].trainIdx].foreground_times<<endl;
				}

				//显示出连续的点
				cout<<"匹配上点数:"<<matches_size<<endl;
				cout<<"连续的点数："<<continuity_num<<endl;
				cout<<"一共有"<<foreground_num<<"个前景点"<<endl;
				et = cvGetTickCount();
				printf("连续轨迹提取时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.);

				//*******************检测前景点**********************//
				st = cvGetTickCount();
				unsigned char *Foreground_times = new unsigned char[num_trj_for_H_cnstrt];					//记录下面判断过程中，每个背景点被判为前景点的次数
				for(int i=0; i<num_trj_for_H_cnstrt; i++)
					Foreground_times[i] = 0;
				//当前帧的轨迹坐标
				vector<Point2f> pt_cur;
				vector<KeyPoint> key_cur, key_t, pt_contuity;
				vector<DMatch> good_matches;
				int good_match_num = 0;	//仅用于good_match中
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
				//gsw_2个邻居帧的轨迹坐标
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
				//多线程初始化变量
				//关键段初始化  
				//InitializeCriticalSection(&g_csThreadParameter);  
				//InitializeCriticalSection(&g_csThreadCode);  
				const int THREAD_NUM = gsw_2;  
				HANDLE handle[THREAD_NUM]; 
				vector<PARA_FOR_HOMO> pthread_array(gsw_2);

				//开始线程
				for (int i=0; i<gsw_2; i++)
				{
					//EnterCriticalSection(&g_csThreadParameter);//进入子线程序号关键区域
					pthread_array[i] = para_for_homo(pt_nebor[i], pt_cur, true, Foreground_times, crop_width, crop_height);
					handle[i] = (HANDLE)_beginthreadex(NULL, 0, Calculate_Homography, &pthread_array[i], 0, NULL); 
				}

				//等待gsw_2个线程计算完毕
				WaitForMultipleObjects(THREAD_NUM, handle, TRUE, 500);//INFINITE);//至多等待20ms?在设置了背景轨迹个数<20就跳过时，还需要设置时间限制吗？？
				//结束线程并释放资源
				DWORD aExitCode = 0;
				for (int i=0; i<gsw_2; i++)
				{
					//CloseHandle(handle[i]);
					TerminateThread(handle[i], aExitCode);
				}
				//如果有进程不是正常结束，Calculate_Homography_completed就为false
				bool Calculate_Homography_completed = true;
				for (int i=0; i<gsw_2; i++)
				{
					if (pthread_array[i].start)
						Calculate_Homography_completed = false;
				}
				if (!Calculate_Homography_completed)
				{
					cout<<"多线程计算单应矩阵【前景点判定部分】异常！！！！！！！！！！！！！！！！！！！！！！！！！1"<<endl;
					continue;
				}
				et = cvGetTickCount();
				printf("前景点判定时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.);
				//for(int k = 0; k < num_trj_for_H_cnstrt; k++)
				//	printf("%d\t",Foreground_times[k]);
				//cout<<endl;
				st = cvGetTickCount();
				//真正的背景轨迹数目
				int bg_num = num_trj_for_H_cnstrt;
				//当前帧的背景轨迹坐标
				//****************重要****************保存背景轨迹的gsw_2+1帧中的轨迹坐标，用于轨迹平滑，其数目保证与bg_num一致
				vector<Point2f> pt_bg_cur;
				//**************重要***********记录背景轨迹在continuity_index中的位置
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
						//背景轨迹数--
						bg_num--;
						continuity_fg[H_cnstrt_index[i]] = true;
						fg_H_cnstrt_index.push_back(i);
						//前景轨迹数++
						foreground_num++;
						key_fg_cur_H_cnstrt.push_back(key_t[i]);
						foreground_index.push_back(H_cnstrt_index[i]);//***************重要***************保存的是该点在continuity_index中排列的序号【而不是在Trajectories中的序号】
					}
					else
					{
						//记录背景轨迹在continuity_index中的位置
						Trajectories[continuity_index[H_cnstrt_index[i]]].background_times++;
						Trajectories[continuity_index[H_cnstrt_index[i]]].award = Trajectories[continuity_index[H_cnstrt_index[i]]].background_times / 40;	//每50次奖励一次
						smooth_index.push_back(H_cnstrt_index[i]);
						Trj_cor_for_smooth.push_back(Trj_cor_continuity[H_cnstrt_index[i]]);
						pt_bg_cur.push_back(Trj_cor_continuity[H_cnstrt_index[i]][gsw]);
#if SHOW_BG_POINT
						key_bg_cur.push_back(KeyPoint((Point2f(Trj_cor_continuity[H_cnstrt_index[i]][gsw].x, Trj_cor_continuity[H_cnstrt_index[i]][gsw].y)), 12.));
#endif
					}
				}
				//调试用，显示哪些点被判定为前景点
				if(num_trj_for_H_cnstrt-bg_num)
				{
					cout<<"本次又发现了"<<num_trj_for_H_cnstrt-bg_num<<"个"<<endl;
					Mat cur_foregroud_keypoint;
					drawKeypoints(nebor_frames_crop.at(gsw), key_fg_cur_H_cnstrt, cur_foregroud_keypoint);

					imshow("foreground_cur_H_cnstrt", cur_foregroud_keypoint);
					cvWaitKey(5);
				}
				//先排序！因为在前景点判别中，可能破坏foreground_index的有序性
				sort(foreground_index.begin(), foreground_index.end());

#if SHOW_BG_POINT
				drawKeypoints(nebor_frames_crop.at(gsw), key_bg_cur, img_matches);
				imshow("SHOW_BG_POINT", img_matches);
				cvWaitKey(0);
#endif
				printf("bg_num num: %d\n", bg_num); 
				delete []Foreground_times;
				et = cvGetTickCount();
				printf("背景点轨迹矩阵提取时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.);
				//*******************************背景点找回********************************//
				st = cvGetTickCount();
				if(bg_num < retrieval_num)
				{
					//多于3个前景点，才能建立徳劳内三角网
					if(foreground_num-2 > 0)
					{
						//将所有连续出现的轨迹坐标，在当前帧上作徳劳内三角形分割
						CvRect rect = {0, 0, crop_width, crop_height};
						CvMemStorage* storage_bundled;
						CvSubdiv2D* subdiv;
						storage_bundled = cvCreateMemStorage(0);
						subdiv = cvCreateSubdiv2D( CV_SEQ_KIND_SUBDIV2D, sizeof(*subdiv),
							sizeof(CvSubdiv2DPoint),
							sizeof(CvQuadEdge2D),
							storage_bundled );//为剖分数据分配空间
						cvInitSubdivDelaunay2D( subdiv, rect );
						for(int i=0; i<continuity_num; i++)
						{
							CvSubdiv2DPoint *pt = cvSubdivDelaunay2DInsert(subdiv, CvPoint2D32f(Trj_cor_continuity[i][gsw_2]));//向三角剖分中插入该点，即对该点进行三角剖分
							//pt->id = continuity_index[i];//**************************重要！！！！为每一个顶点分配一个id********************************//
							pt->id = i;	//不直接保存在Trajectories中的序号，而是保存在continuity_index中的序号，这样好判断该点是否前景点，尤其在计算邻居点中前景点个数的时候
						}
						//筛选出正确的徳劳内三角形，并保存每个徳劳内三角形的三个顶点
						CvSeqReader  reader;
						cvStartReadSeq( (CvSeq*)(subdiv->edges), &reader, 0 );//使用CvSeqReader遍历Delaunay或者Voronoi边
						int edges_num = subdiv->edges->total;
						Vec3i verticesIdx;
						vector<Vec3i> Delaunay_tri;		//存储徳劳内三角形的三个顶点的集合
						Point buf[3];							//存储三个边对应的顶点
						const Point *pBuf = buf;
						int elem_size = subdiv->edges->elem_size;//边的大小
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
									CvSubdiv2DPoint* pt = cvSubdiv2DEdgeOrg( t );//获取t边的源点
									if( !pt ) break;
									buf[j] = pt->pt;//将点存储起来
									verticesIdx[j] = pt->id;//获取顶点的Id号，将三个点的id存储到verticesIdx中
									t = cvSubdiv2DGetEdge( t, CV_NEXT_AROUND_LEFT );//获取下一条边
								}
								if (j != iPointNum) continue;
#if SHOW_DELAUNAY
								if (isGoodTri(verticesIdx, Delaunay_tri))
								{
									polylines( nebor_frames_crop[gsw], &pBuf, &iPointNum, 
										1, true, color,
										1, CV_AA, 0);//画出三条边
								}
#else
								isGoodTri(verticesIdx, Delaunay_tri);
#endif

								t = (CvSubdiv2DEdge)edge+2;//相反边缘 reversed e
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
						imshow("徳劳内三角形", nebor_frames_crop[gsw]);
						cvWaitKey(0);
#endif
						//cout<<"徳劳内三角形的所有三个点的序号已经压入Delaunay_tri"<<endl;
						//寻找前景点的邻居点
						vector<vector<int>> nb_of_fg(foreground_num);
						//寻找每一个%前景%点的邻居点，不能重复，所以，要在每一次找到邻居点时候，判断该前景点的邻居点矩阵里面是否已有该邻居，如没有，才压入nb_of_fg矩阵
						//也可不这么麻烦，因为在后面用到前景点邻居时候，使用set类，set可自动排除重复的点
						for(int i=0; i < Delaunay_tri.size(); i++)
						{
							//Delaunay_tri中维持的是continuity_index中的序号，所以，前面对foreground_index排序，不会对此处有影响！！！！！！！！！
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
						//cout<<"前景点的邻居点的序号已经完全压入nb_of_fg"<<endl;
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
						//判断前景点与邻居点的轨迹相似性
						for(int i=0; i<foreground_num; i++)
						{
							//cout<<"i = "<<i<<"\t"<<endl;
							//如果该前景点周围前景点个数太多，则判为前景点
							int nb_num = nb_of_fg[i].size();

							if(nb_num > 0)
							{
								int sum_of_fg = 0;
								//统计该前景点邻居点中的前景点个数
								for(int j=0; j<nb_num; j++)
									sum_of_fg += continuity_fg[nb_of_fg[i][j]];
								//邻居点中前景点数小于一半比例才计算轨迹相似性，否则，维持原判
								if(sum_of_fg < nb_num/2)
								{
									int similarity = 0;
									for(int j=0; j<nb_num; j++)
									{
										//1行7列的向量
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
										//重大修改！！！不找回的点不直接放到背景轨迹矩阵里面，而是重新判定！！！
										if (Trajectories[continuity_index[foreground_index[i]]].foreground_times < H_cnstrt)//max_H_cnstrt_tolerance)
										{
											Trj_cor_for_smooth.push_back(Trj_cor_continuity[foreground_index[i]]);
											continuity_fg[foreground_index[i]] = false;
											bg_num++;
											smooth_index.push_back(foreground_index[i]);
											pt_bg_cur.push_back(Trj_cor_continuity[foreground_index[i]][gsw]);
										}
									}
									//else//继续惩罚那些糟糕的前景点
									//	if(Trajectories[continuity_index[foreground_index[i]]].foreground_times)
									//		Trajectories[continuity_index[foreground_index[i]]].foreground_times++;
								}
								//else//继续惩罚那些糟糕的前景点
								//	Trajectories[continuity_index[foreground_index[i]]].foreground_times++;
							}
							else
								cout<<"该前景点没有邻居点。。。"<<endl;	//没有邻居点， 其实是一个bug，得调
						}
						cvReleaseMemStorage(&storage_bundled);
						foreground_num = continuity_num - bg_num;
						cout<<"找回部分背景点之后，背景点个数为"<<bg_num<<endl;
					}
					else cout<<"很可惜，没有前景点需要找回。。。。。。。。。"<<endl;
				}
				et = cvGetTickCount();
				printf("背景轨迹找回时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.);

				//找回之后，所有的背景点
				Mat show_all_bg_keypoint;
				vector<KeyPoint> all_bg_key;
				for (int i=0; i<bg_num; i++)
				{
					all_bg_key.push_back(KeyPoint(Trj_cor_for_smooth[i][gsw_2].x, Trj_cor_for_smooth[i][gsw_2].y, 15));
				}
				drawKeypoints(crop_cur, all_bg_key, show_all_bg_keypoint);
				imshow("找回之后，所有的背景点", show_all_bg_keypoint);
				cvWaitKey(5);
				//***************************************轨迹平滑****************************************//
				//*************************************低通滤波算法****************************************//
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
				printf("低通滤波时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.);
				//求Trj_cor_for_smooth --> Trj_cor_smooth的单应矩阵
				st = cvGetTickCount();
				//背景平滑
				Mat homo = Mat::zeros(3, 3, CV_32F);
				//坐标还原
				for(int i=0; i<bg_num; i++)
				{
					pt_bg_cur[i] *= scale;
					pt_bg_cur[i].y += cropped_start;
					Trj_cor_smooth[i] *= scale;
					Trj_cor_smooth[i].y += cropped_start;
				}
				Mat ransac_outliers = Mat::zeros(1, bg_num, CV_8U);			//好的结果，1表示该数据与模型匹配得好，0为不好
				homo = Homography_Nelder_Mead_with_outliers(pt_bg_cur, Trj_cor_smooth, 200, ransac_outliers, height);///_Mutli_Threads  

				et = cvGetTickCount();
				printf("单应矩阵计算时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.);

				st = cvGetTickCount();
				CvMat H = homo;
				std::vector<Point2f> obj_corners(4);
				obj_corners[0] = Point(0,0); obj_corners[1] = Point( nebor_frames[gsw].cols, 0 );
				obj_corners[2] = Point( nebor_frames[gsw].cols, nebor_frames[gsw].rows ); obj_corners[3] = Point( 0, nebor_frames[gsw].rows );
				std::vector<Point2f> scene_corners(4);
				perspectiveTransform(obj_corners, scene_corners, homo);
				//dx、dy用于判断warp之后是否需要平移，dx<0则水平平移，dy<0则竖直平移
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
				//cvShowImage("透视变换效果", temp_frame);
				//waitKey(0);
				Mat temp_mat(temp_frame);
				//cout<<"剪切后: "<<howtocrop<<"\t"<<height-howtocrop<<endl;
				Mat stab_frame = temp_mat(Range(howtocrop_height, height-howtocrop_height), Range(howtocrop_width, width-howtocrop_width));
				//imshow("原始帧", nebor_frames_crop[gsw]);
				//imshow("稳定帧", stab_frame(Range(cropped_start, cropped_end), Range(30, width-30)));
				//waitKey(0);
				//保存视频和输出
				cvWriteFrame(Save_result, &IplImage(stab_frame));//(Range(cropped_start/scale, cropped_end/scale), Range(0, width/scale))));
				et = cvGetTickCount();
				printf("稳定帧生成时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.);
				/******************************************************************************************/
				/***************************************控制轨迹数量******************************************/
				/******************************************************************************************/
				st = cvGetTickCount();
				//剔除连续disappear_tolerance帧没有出现的帧，和超出边界一定帧数的轨迹
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
						/******************************************修正last_in_Trj在Trajectories中的位置****************************************/
						for(int j=0; j<cur_key_size; j++)
							if(last_in_Trj[j] == last_size-1)//因为是中间第i个元素与末尾元素对换位置，并将最后一个剔除掉，所以，只需要将对应于最后一个元素的cur_key的标记位置改一下，其余的不用做“减一”操作
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
						/******************************************修正last_in_Trj在Trajectories中的位置****************************************/
						for(int j=0; j<cur_key_size; j++)
							if(last_in_Trj[j] == last_size-1)//因为是中间第i个元素与末尾元素对换位置，并将最后一个剔除掉，所以，只需要将对应于最后一个元素的cur_key的标记位置改一下，其余的不用做“减一”操作
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
						/******************************************修正last_in_Trj在Trajectories中的位置****************************************/
						for(int j=0; j<cur_key_size; j++)
							if(last_in_Trj[j] == last_size-1)//因为是中间第i个元素与末尾元素对换位置，并将最后一个剔除掉，所以，只需要将对应于最后一个元素的cur_key的标记位置改一下，其余的不用做“减一”操作
							{
								last_in_Trj[j] = i;
								break;
							}
							last_size--;
					}
				}
				et = cvGetTickCount();
				printf("废弃轨迹丢弃和轨迹矩阵维护时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.);
				/************************************调整特征点个数*************************************/
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
				cout<<"当前num_of_corners: "<<num_of_corners<<endl;
				cout<<"当前特征点数"<<cur_key.size()<<endl;
				//*****************************************************防止内存泄露****************************************************//
				cvReleaseImage(&temp_frame);
				cout<<"total trj number is "<<last_size<<endl;
				et = cvGetTickCount();
				printf("轨迹数量控制模块消耗时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.);
			}
			else
			{
				//重大修改啊！！！修正了drawmatches时每次都显示同一帧的问题
				//****************************************************用mat而不是IplImage，防止内存泄露****************************************************//
				Mat crop_cur_copy = crop_cur.clone();
				//又一个重大修改，解决了大问题，因为cvWarpPerspective函数需要调用nebor_frames，所以，这个队列也不能有重复帧出现
				Mat cur_copy(frame_cur, true);
				nebor_frames.push_back(cur_copy);	//未进行到gsw_2+1帧时，直接压入队列
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