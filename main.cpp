//
//  main.cpp
//  PSOQuatGraphSeg
//
//  Created by Giorgio on 25/10/15.
//  Copyright (c) 2015 Giorgio. All rights reserved.

#include <iostream>

//#include "/Users/giorgio/Documents/MATLAB/mypc.h"

#include "opencv2/core/core.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/calib3d/calib3d.hpp"
//#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include "opencv2/opencv.hpp"

//#include <math.h>
#include <string>
//#include <GLUT/glut.h>
//#include <GLUT/glew.h>
//#ifdef __APPLE__
//#include <GL/glew.h>//include before GL/glut.h and it contains GL/gl.h and GL/glu.h
//#include <GLUT/glut.h>
//#else
////#include <GL/gl.h>
//#include <GL/glew.h>//include before GL/glut.h and it contains GL/gl.h and GL/glu.h
//#include <GL/glut.h>
//#endif

//#include <cstdio>

// Include GLM
//#include <glm/glm.hpp>
//#include <glm/gtc/matrix_transform.hpp>
//#include "model.h"
//#include <gsl/gsl_rng.h>
#include "pso_class_quaternions.h"

//#include <boost/shared_ptr.hpp>
//#include <boost/lexical_cast.hpp>
//
//#include <fstream>
//#include <time.h>
//
//#include <algorithm>    // std::min_element, std::max_element

#include "GraphCannySeg.h"

#include "pso_opengl_render.h"

#include <boost/thread.hpp>


#include <assimp/config.h>
#include <assimp/mesh.h>



//#define FX 549.8673626911883f
//#define FY 550.4240710445896f
//#define CX 323.6734992813105f
//#define CY 230.2912939215161f
//ACCV
//#define FX 572.41140
//#define FY 573.57043
//#define CX 325.26110
//#define CY 242.04899

//rutgers
//#define FX 575.8157348632812
//#define FY 575.8157348632812
//#define CX 319.5
//#define CY 239.5

//challenge
#define FX 571.9737
#define FY 571.0073
#define CX 319.5000
#define CY 239.5000

//articulated
//#define FX 525.0f
//#define FY 525.0f
//#define CX 320.0f
//#define CY 240.0f

#define tomm_ 1000.0f
#define tocast_ 0.5f

//Rutgers
//double fx=575.8157348632812;
//double fy=575.8157348632812;
//double cx=319.5;
//double cy=239.5;

//ACCV
//double fx=572.41140;
//double fy=573.57043;
//double cx = 325.26110;
//double cy = 242.04899;

//CHALLENGE 1 DATASET
double fx = 571.9737;
double fy = 571.0073;
double cx = 319.5000;
double cy = 239.5000;

//articulated
//double fx = 525.0f;
//double fy = 525.0f;
//double cx = 320.0f;
//double cy =240.0f;



float k_vec[9] = {static_cast<float>(fx), 0, static_cast<float>(cx), 0, static_cast<float>(fy), static_cast<float>(cy), 0.f,0.f,1.f};

/*Segmentation Params*/
//params challenge
int k=38; //50;  //50 x  challenge         //67;//30; //0.003 /10000
int kx=5430; //2000;//2.0f;
int ky=30; //30;//0.03f;
int ks=50;//50;//0.05f;
float kdv=4.5f;
float kdc=0.1f;
float min_size=500.0f;
float sigma=0.8f;
float max_ecc = 0.978f;
float max_L1 = 3800.0f;
float max_L2 = 950.0f;

//params rutgers
//int k=50000;  //50 x  challenge         //67;//30; //0.003 /10000
//int kx=1050; //2000;//2.0f;
//int ky=1500; //30;//0.03f;
//int ks=500;//50;//0.05f;
//float kdv=4.5f;
//float kdc=0.1f;
//float min_size=500.0f;
//float sigma=0.8f;
//float max_ecc = 0.978f;
//float max_L1 = 3800.0f;
//float max_L2 = 950.0f;


int DTH = 30; //[mm]
int plusD = 8; //7; //for depth boundary
int point3D = 5; //10//for contact boundary
int g_angle = 162; //140;//148;//2.f/3.f*M_PI;
int l_angle = 56; //M_PI/3.f;
int Lcanny = 50;
int Hcanny = 75;
int FarObjZ = 1800; //875;//1800; //[mm]

std::string trackBarsWin = "Trackbars";
//Segmentation Results
GraphCanny::GraphCannySeg<GraphCanny::hsv>* gcs=0;
std::vector<GraphCanny::SegResults> vecSegResult;
//used to load rgb and depth images fromt the dataset
cv::Mat kinect_rgb_img;
cv::Mat kinect_depth_img_mm;
cv::Mat imageMask;

// for loading ACCV dataset
cv::Mat loadDepth( std::string a_name )
{
    cv::Mat lp_mat;
    std::ifstream l_file(a_name.c_str(),std::ofstream::in|std::ofstream::binary );
    
    if( l_file.fail() == true )
    {
        printf("cv_load_depth: could not open file for writing!\n");
        return lp_mat;
    }
    int l_row;
    int l_col;
    
    l_file.read((char*)&l_row,sizeof(l_row));
    l_file.read((char*)&l_col,sizeof(l_col));
    
    IplImage * lp_image = cvCreateImage(cvSize(l_col,l_row),IPL_DEPTH_16U,1);
    
    for(int l_r=0;l_r<l_row;++l_r)
    {
        for(int l_c=0;l_c<l_col;++l_c)
        {
            l_file.read((char*)&CV_IMAGE_ELEM(lp_image,unsigned short,l_r,l_c),sizeof(unsigned short));
        }
    }
    l_file.close();
    
    lp_mat= cv::Mat(lp_image);
    return lp_mat;
}





void on_trackbar( int, void* )
{
    
    if(gcs)
      delete gcs;
    
    float kfloat = (float)k/10000.f;
    float kxfloat = (float)kx/1000.f;
    float kyfloat = (float)ky/1000.f;
    float ksfloat = (float)ks/1000.f;
    float gafloat = ((float)g_angle)*deg2rad;
    float lafloat = ((float)l_angle)*deg2rad;
    float lcannyf = (float)Lcanny/1000.f;
    float hcannyf = (float)Hcanny/1000.f;
    //GraphCanny::GraphCannySeg<GraphCanny::hsv> gcs(kinect_rgb_img, kinect_depth_img_mm, sigma, kfloat, min_size, kxfloat, kyfloat, ksfloat,k_vec,lcannyf,hcannyf,kdv, kdc,max_ecc,max_L1,max_L2,(uint16_t)DTH,(uint16_t)plusD,(uint16_t)point3D,gafloat,lafloat,(float)FarObjZ);
    gcs = new GraphCanny::GraphCannySeg<GraphCanny::hsv>(kinect_rgb_img, kinect_depth_img_mm, sigma, kfloat, min_size, kxfloat, kyfloat, ksfloat,k_vec,lcannyf,hcannyf,kdv, kdc,max_ecc,max_L1,max_L2,(uint16_t)DTH,(uint16_t)plusD,(uint16_t)point3D,gafloat,lafloat,(float)FarObjZ);
    gcs->run();
    
    vecSegResult = gcs->vecSegResults;
    
    //Just for debug !!
    
//    for(int i=0;i<gcs.vecSegResults.size();++i)
//    {
//        cv::imshow("Single Debug",gcs.vecSegResults[i].clusterRGB);
//        
//        gcs.visualizeColorMap(gcs.vecSegResults[i].clusterDepth, "Single Debug DEPTH", 5,false);
//        cv::waitKey(0);
//    }
    
   
    
    
    //text.zeros(480, 640,CV_8UC1);
    cv::Mat text = cv::Mat::zeros(230, 640,CV_8UC1);
    char text_[200]={};
    sprintf(text_, "DTH: %d plusD: %d point3D: %d",DTH,plusD,point3D);
    std::string tstring(text_);
    std::cout<<tstring<<"\n";
    cv::putText(text, tstring, cv::Point(50,50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255));
    
    sprintf(text_, "K: %f Kx: %.2f Ky: %f Ks: %f",kfloat,kxfloat,kyfloat,ksfloat);
    tstring = std::string(text_);
    std::cout<<tstring<<"\n";
    cv::putText(text, tstring, cv::Point(50,100), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255));
    
    sprintf(text_, "G_angle: %d  L_angle: %d  Zmax: %d",g_angle,l_angle, FarObjZ);
    tstring = std::string(text_);
    std::cout<<tstring<<"\n";
    cv::putText(text, tstring, cv::Point(50,150), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255));
    
    sprintf(text_, "Low Canny : %f  High Canny: %f",lcannyf,hcannyf);
    tstring = std::string(text_);
    std::cout<<tstring<<"\n";
    cv::putText(text, tstring, cv::Point(50,200), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255));
    
    cv::imshow( trackBarsWin, text );
    cv::waitKey(5);
    
}

void initTrackbarsSegmentation()
{
    /* First Load the RGB and DEPTH */
    //Open Images

    //CHALLENGE DATASET-1
    
//    std::string rgb_name= "img_1164.png";//"img_163.png";
//    std::string obj_name= "Coffee_Cup";//"Shampoo";
//    std::string rgb_file_path = "/Users/morpheus/Downloads/img_1164.png";
//    std::string depth_file_path = "/Users/morpheus/Downloads/img_1164_depth.png";
    
    std::string rgb_file_path = "/Users/morpheus/Dropbox/PSO/luaCNN/j6WP0/rgbj6WP00.png";
    std::string depth_file_path = "/Users/morpheus/Dropbox/PSO/luaCNN/j6WP0/depthj6WP00.png";

    
    
    
//    cv::Mat imageMask = cv::Mat();
    
    std::string rgb_name= "img_1164.png";
    std::string obj_name= "Shampoo";
//    std::string rgb_file_path = "/Users/morpheus/Downloads/"+obj_name+"/RGB/"+rgb_name;
//    std::string depth_file_path = "/Users/morpheus/Downloads/"+obj_name+"/Depth/"+rgb_name;
    cv::Mat imageMask = cv::Mat();
 
     rgb_name= "img_134.png";
    obj_name= "Shampoo";
    rgb_name= "img_228.png";
    obj_name= "Milk";
//    rgb_file_path = "/Users/morpheus/Downloads/0000.png";
//    depth_file_path = "/Users/morpheus/Downloads/0000_depth.png";
    
    
    
    
    rgb_file_path = "/Volumes/HD-PNTU3/datasets/Coffee_Cup_small/RGB/img_019.png";
    depth_file_path = "/Volumes/HD-PNTU3/datasets/Coffee_Cup_small/Depth/img_019.png";
    
    std::string mask_file_path  = "/Volumes/HD-PNTU3/datasets/Coffee_Cup_small/RGB/img_019_mask1.png";
    
//        rgb_file_path = "/Volumes/HD-PNTU3/datasets/Articulated_Objects_BMVC15/test/Cupboard_Seq_2/rgb_noseg/00333.png";
//        depth_file_path = "/Volumes/HD-PNTU3/datasets/Articulated_Objects_BMVC15/test/Cupboard_Seq_2/depth_filled/00333.png";
//    
//      std::string mask_file_path  = "/Volumes/HD-PNTU3/datasets/Articulated_Objects_BMVC15/test/Cupboard_Seq_2/00333_mask.png";

    //    rgb_file_path = "/Volumes/HD-PNTU3/datasets/Articulated_Objects_BMVC15/test/Laptop_Seq_2/rgb_noseg/00200.png";
//    depth_file_path = "/Volumes/HD-PNTU3/datasets/Articulated_Objects_BMVC15/test/Laptop_Seq_2/depth_filled/00200.png";
//    std::string mask_file_path  = "/Volumes/HD-PNTU3/datasets/Articulated_Objects_BMVC15/test/Laptop_Seq_2/mask/00200_mask.png";
    
//    rgb_file_path = "/Volumes/HD-PNTU3/datasets/Articulated_Objects_BMVC15/test/Laptop_Seq_2/rgb_noseg/00060.png";
//    depth_file_path = "/Volumes/HD-PNTU3/datasets/Articulated_Objects_BMVC15/test/Laptop_Seq_2/depth_filled/00060.png";
//    std::string mask_file_path  = "/Volumes/HD-PNTU3/datasets/Articulated_Objects_BMVC15/test/Laptop_Seq_2/mask/00060_mask.png";
    
//    rgb_file_path = "/Volumes/HD-PNTU3/datasets/Articulated_Objects_BMVC15/test/Laptop_Seq_2/rgb_noseg/00150.png";
//    depth_file_path = "/Volumes/HD-PNTU3/datasets/Articulated_Objects_BMVC15/test/Laptop_Seq_2/depth_filled/00150.png";
//    std::string mask_file_path  = "/Volumes/HD-PNTU3/datasets/Articulated_Objects_BMVC15/test/Laptop_Seq_2/mask/00150_mask.png";
    
//    rgb_file_path = "/Volumes/HD-PNTU3/datasets/Articulated_Objects_BMVC15/test/Laptop_Seq_1/rgb_noseg/00929.png";
//    depth_file_path = "/Volumes/HD-PNTU3/datasets/Articulated_Objects_BMVC15/test/Laptop_Seq_1/depth_filled/00929.png";
//    std::string mask_file_path  = "/Volumes/HD-PNTU3/datasets/Articulated_Objects_BMVC15/test/Laptop_Seq_1/mask/00929_mask.png";
    
    
//    rgb_file_path = "/Volumes/HD-PNTU3/datasets/ACCV/eggbox/data/color61.jpg";
//    depth_file_path = "/Volumes/HD-PNTU3/datasets/ACCV/eggbox/data/depth61.dpt";
//    rgb_file_path = "/Volumes/HD-PNTU3/datasets/Milk/RGB/img_261.png";
//    depth_file_path = "/Volumes/HD-PNTU3/datasets/Milk/Depth/img_261.png";
    

//     rgb_file_path = "/Users/morpheus/Downloads/temptar/run_2/munchkin_white_hot_duck_bath_toy-image-J-2-1-0.png";
//    depth_file_path = "/Users/morpheus/~/Downloads/temptar/run_2/munchkin_white_hot_duck_bath_toy-depth-J-2-1-0.png";

//    rgb_file_path = "/Volumes/HD-PNTU3/datasets/"+obj_name+"/RGB/"+rgb_name;
//    depth_file_path = "/Volumes/HD-PNTU3/datasets/"+obj_name+"/Depth/"+rgb_name;
    /*
     //RGB-D DATASET
     std::string png_name= "table_1_108";//"img_163.png";
     
     std::string rgb_file_path = "/Users/giorgio/Downloads/rgbd-scenes_all/table/table_1/"+png_name+".png";
     std::string depth_file_path = "/Users/giorgio/Downloads/rgbd-scenes_all/table/table_1/"+png_name+"_depth.png";
     cv::Mat imageMask = cv::Mat();
     */
    kinect_rgb_img = cv::imread(rgb_file_path);//,cv::IMREAD_UNCHANGED);
    std::cout << "ok\n";
    kinect_depth_img_mm = cv::imread(depth_file_path,cv::IMREAD_UNCHANGED);// in mm   loadDepth(depth_file_path);  //
    std::cout << "ok\n";
    cv::imshow("kinect_rgb_img",kinect_rgb_img);
    cv::imshow("kinect_depth_img_mm",kinect_depth_img_mm);
    
    /*Create the TrackBars for Segmentation Params*/
//    cv::namedWindow(trackBarsWin,0);
//    
//    cv::createTrackbar("k", trackBarsWin, &k, 1000,on_trackbar);
//    cv::createTrackbar("kx", trackBarsWin, &kx, 10000,on_trackbar);
//    cv::createTrackbar("ky", trackBarsWin, &ky, 1000,on_trackbar);
//    cv::createTrackbar("ks", trackBarsWin, &ks, 1000,on_trackbar);
//    cv::createTrackbar("DTH", trackBarsWin, &DTH, 100,on_trackbar);
//    cv::createTrackbar("plusD", trackBarsWin, &plusD, 100,on_trackbar);
//    cv::createTrackbar("Point3D", trackBarsWin, &point3D, 100,on_trackbar);
//    cv::createTrackbar("G Angle", trackBarsWin, &g_angle, 180,on_trackbar);
//    cv::createTrackbar("L Angle", trackBarsWin, &l_angle, 180,on_trackbar);
//    cv::createTrackbar("H Canny th", trackBarsWin, &Hcanny, 100,on_trackbar);
//    cv::createTrackbar("L Canny th", trackBarsWin, &Lcanny, 100,on_trackbar);
//    cv::createTrackbar("FarObjZ", trackBarsWin, &FarObjZ, 2500,on_trackbar);
//    
//    on_trackbar( 0, 0 );
//    
//    /// Wait until user press some key
//    cv::waitKey(0);
    
    
    
    
    // added by STE, puts mask in vecSegCluster[0]
    imageMask= cv::imread(mask_file_path ,0);
    cv::imshow("mask",imageMask);
    bitwise_not(imageMask,imageMask);
    
    
    //try AABB
    std::vector<cv::Point2f> bbpp;
    
    for (int i=0; i<640;++i)
        for(int j=0;j<480;++j) {
            if(imageMask.at<unsigned char>(j,i)>0)
            {		cv::Point2f p2(i,j);
                bbpp.push_back(p2);
            }
        }
    cv::Rect bb = cv::boundingRect(bbpp);
    
    cv::Mat_<float> K_ = cv::Mat_<float>(3,3,&k_vec[0]);
    cv::Mat K_inv = K_.inv();
    
    cv::Mat_<float> XYZ;
    int centx=(bb.br().x- bb.tl().x)/2+bb.tl().x;
    int centy=(bb.br().y- bb.tl().y)/2+bb.tl().y;
    printf("create centroid3D\n");
    gcs->projectPixel2CameraRF(K_inv,
                          centx,
                          centy,
                          kinect_depth_img_mm.at<uint16_t>(centy,centx),
                          XYZ);
    XYZ(2)=kinect_depth_img_mm.at<uint16_t>(220,310);
    cv::Point3f centroid3D_(XYZ(0),XYZ(1),XYZ(2));
    printf("centroid: %f %f %f\n",XYZ(0),XYZ(1),XYZ(2));
    cv::Mat clusterDepth_(480,640,CV_16U);
    uint16_t* tempDepthPtr = clusterDepth_.ptr<uint16_t>(0);
    uint16_t* kinectDepthPtr = kinect_depth_img_mm.ptr<uint16_t>(0);
    unsigned char* maskDepthPtr = imageMask.ptr<unsigned char>(0);
    printf("fill cluster depth\n");
    int numPoints=bbpp.size();
    float sum=0;
    for(int idd=0;idd<640*480;++idd)
    {
        if( maskDepthPtr[idd]>0)
        {tempDepthPtr[idd] = kinectDepthPtr[idd];sum+=(float)kinectDepthPtr[idd];}
        else
            tempDepthPtr[idd]=0;
    }
    std::vector<cv::Point3i> pxs_;
    cv::Mat clusterRGB_;
    
    centroid3D_.z= (sum/numPoints); //+200;// per cupboard
    centroid3D_.x=0;
    centroid3D_.y=0;
    //centroid3D_.z=900;
    printf("mean centroid z: %f\n",centroid3D_.z);

    //TODO: HARDCODED AABB of the segmented cluster Laptop img 00341.png
    GraphCanny::SegResults temp(centroid3D_, centroid3D_, cv::Point2i(centx,centy),0,0,0, numPoints,pxs_, clusterRGB_,clusterDepth_,bb);

    float kfloat = (float)k/10000.f;
    float kxfloat = (float)kx/1000.f;
    float kyfloat = (float)ky/1000.f;
    float ksfloat = (float)ks/1000.f;
    float gafloat = ((float)g_angle)*deg2rad;
    float lafloat = ((float)l_angle)*deg2rad;
    float lcannyf = (float)Lcanny/1000.f;
    float hcannyf = (float)Hcanny/1000.f;
    //GraphCanny::GraphCannySeg<GraphCanny::hsv> gcs(kinect_rgb_img, kinect_depth_img_mm, sigma, kfloat, min_size, kxfloat, kyfloat, ksfloat,k_vec,lcannyf,hcannyf,kdv, kdc,max_ecc,max_L1,max_L2,(uint16_t)DTH,(uint16_t)plusD,(uint16_t)point3D,gafloat,lafloat,(float)FarObjZ);
    gcs = new GraphCanny::GraphCannySeg<GraphCanny::hsv>(kinect_rgb_img, kinect_depth_img_mm, sigma, kfloat, min_size, kxfloat, kyfloat, ksfloat,k_vec,lcannyf,hcannyf,kdv, kdc,max_ecc,max_L1,max_L2,(uint16_t)DTH,(uint16_t)plusD,(uint16_t)point3D,gafloat,lafloat,(float)FarObjZ);
    
    gcs->vecSegResults.push_back(temp);
    vecSegResult=gcs->vecSegResults;
    // fine added by ste
    //cv::waitKey();

}

void initPSOClass(PSOClass*& PSO_,
                  int IDX_cluster)
{
    /*** HereAfter we use the result of the segmentation ***/
    //Pick a Cluster : eg. cluster 1
    
    //TODO: For fast test we copy the result to the already present matrices....to be fixed
    //int IDX_cluster=0;
    
    /*
    GraphCanny::GraphCannySeg<GraphCanny::hsv>::convertDepth2ColorMap(vecSegResult[IDX_cluster].clusterDepth,SegResDepthRGB);
    vecSegResult[IDX_cluster].clusterRGB.copyTo(SegResRGB);
    //conversion from mm to m
    vecSegResult[IDX_cluster].clusterDepth.convertTo(SegResDepthM, CV_32F,1.e-3f);
    */


    //INIT THE QUATERNION PSO
    //PSO CLASS
    
    std::vector<double> Dimbounds;
    float t_bound = 0.04f;   // 0.2 x artic //0.04f; //in m
    float t_bound_z = 0.1; //0.1f;//0.1f //in m
    //tx,ty,tz in m -> so we need to convert mm->m
    Dimbounds.push_back(static_cast<double>(vecSegResult[IDX_cluster].centroid3D.x*0.001f-t_bound) );
    Dimbounds.push_back(static_cast<double>(vecSegResult[IDX_cluster].centroid3D.x*0.001f+t_bound) );
    Dimbounds.push_back(static_cast<double>(vecSegResult[IDX_cluster].centroid3D.y*0.001f-t_bound) );
    Dimbounds.push_back(static_cast<double>(vecSegResult[IDX_cluster].centroid3D.y*0.001f+t_bound) );
    Dimbounds.push_back(static_cast<double>(vecSegResult[IDX_cluster].centroid3D.z*0.001f-t_bound_z) );
    Dimbounds.push_back(static_cast<double>(vecSegResult[IDX_cluster].centroid3D.z*0.001f+t_bound_z) );
    //quat
    //q0
    Dimbounds.push_back(0.001);
    Dimbounds.push_back(1.0);
    //q1
    Dimbounds.push_back(-1.0);
    Dimbounds.push_back(1.0);
    //q2
    Dimbounds.push_back(-1.0);
    Dimbounds.push_back(1.0);
    //q3
    Dimbounds.push_back(-1.0);
    Dimbounds.push_back(1.0);
    //alfa0_ articulated 1DOF
//    Dimbounds.push_back(0.0);
//    Dimbounds.push_back(0.33);
    //Dimbounds.push_back(20.0);
    //Dimbounds.push_back(40.0);   //20-40 or 70-100
    Dimbounds.push_back(30.0);
    Dimbounds.push_back(80.0);
    
    
    
//    //reitni
//    
////    0.026750
////    0.202456
////    2.021780
////    0.983022
////    0.012499
////    0.141231
////    -0.116470
//    
//    
//    t_bound=0.1;
//    t_bound_z = 0.1;
//    vecSegResult[IDX_cluster].centroid3D.x=0.026750;
//    vecSegResult[IDX_cluster].centroid3D.y=0.202456;
//    vecSegResult[IDX_cluster].centroid3D.z=2.021780;
//        Dimbounds.push_back(static_cast<double>(vecSegResult[IDX_cluster].centroid3D.x-t_bound) );
//        Dimbounds.push_back(static_cast<double>(vecSegResult[IDX_cluster].centroid3D.x+t_bound) );
//        Dimbounds.push_back(static_cast<double>(vecSegResult[IDX_cluster].centroid3D.y-t_bound) );
//        Dimbounds.push_back(static_cast<double>(vecSegResult[IDX_cluster].centroid3D.y+t_bound) );
//        Dimbounds.push_back(static_cast<double>(vecSegResult[IDX_cluster].centroid3D.z-t_bound_z) );
//        Dimbounds.push_back(static_cast<double>(vecSegResult[IDX_cluster].centroid3D.z+t_bound_z) );
//    double pert=0;
//    Dimbounds.push_back(0.983022-pert);    Dimbounds.push_back(0.983022+pert);
//    Dimbounds.push_back(0.012499-pert);    Dimbounds.push_back(0.012499+pert);
//    Dimbounds.push_back(0.141231-pert);    Dimbounds.push_back(0.141231+pert);
//    Dimbounds.push_back(-0.116470-pert);    Dimbounds.push_back(-0.116470+pert);
//    
//    Dimbounds.push_back(0.095964-0.1);    Dimbounds.push_back(0.095964+0.1);
//    
    
    
    
    
    
    
    
    
    double Tss = 0.05; //0.05; //0.3
    double c1__ = 1.496;
    double c2__ = 1.496;
    unsigned int dynamicInertia = PSOClass::PSO_W_LIN_DEC;//PSOClass::PSO_W_CONST;//
    unsigned int nhoodtype = PSOClass::PSO_NHOOD_GLOBAL;// PSOClass::PSO_NHOOD_RING;
    int nhood_size = 7;//(Only for Random Topology)
    
    int maxSteps = 30; // 30
    double ErrorGoal = 0.0001;
    
    PSO_ = new PSOClass(/*Ndim,NSize,*/Dimbounds,c1__,c2__,
                    Tss,dynamicInertia,
                    nhoodtype,maxSteps,ErrorGoal,nhood_size);
    
}


struct ObjectModel_t
{
    float* x; //4 byte each: 4x4=16byte
    float* y;
    float* z;
    int* idx;
};

//
inline void recursiveVertexIndexLoad(const struct aiScene *sc,
                                              const aiNode* nd, ObjectModel_t& obj)
{
    static unsigned int IDX=0;
    static unsigned int IDXIndex=0;
    for (int n=0; n < nd->mNumMeshes; ++n)
    {
        const struct aiMesh* mesh = sc->mMeshes[nd->mMeshes[n]];
        //printf("%d \n",mesh->mNumVertices);
        for (int t = 0; t < mesh->mNumVertices; ++t)
        {
            aiVector3D tmp = mesh->mVertices[t];
            //printf("%f %f %f \n",tmp.x,tmp.y,tmp.z);
            obj.x[IDX] = tmp.x;
            obj.y[IDX] = tmp.y;
            obj.z[IDX] = tmp.z;
            ++IDX;
        }
        //
        for (int t = 0; t < mesh->mNumFaces; ++t)
        {
            const struct aiFace* face = &mesh->mFaces[t];
            
            for (int i = 0; i < face->mNumIndices; i++)
            {
                //int index = face->mIndices[i];

                obj.idx[IDXIndex] = (int)face->mIndices[i];

                //printf("IDXIndex: %d ;;",IDXIndex);
                //objectIndexes.push_back(index);
                ++IDXIndex;
                //objectPoints.push_back(Point3f(mesh->mVertices[index].x,mesh->mVertices[index].y,mesh->mVertices[index].z));
            }
        }
        //
    }
    for (int n = 0; n < nd->mNumChildren; ++n)
        recursiveVertexIndexLoad(sc, nd->mChildren[n],obj);
    
}
//

inline void recursiveVertexLoad(const struct aiScene *sc,
                                         const aiNode* nd, ObjectModel_t& obj)
{
    static unsigned int IDX=0;
    for (int n=0; n < nd->mNumMeshes; ++n)
    {
        const struct aiMesh* mesh = sc->mMeshes[nd->mMeshes[n]];
        //printf("%d \n",mesh->mNumVertices);
        for (int t = 0; t < mesh->mNumVertices; ++t)
        {
            aiVector3D tmp = mesh->mVertices[t];
            //printf("%f %f %f \n",tmp.x,tmp.y,tmp.z);
            obj.x[IDX] = tmp.x;
            obj.y[IDX] = tmp.y;
            obj.z[IDX] = tmp.z;
            ++IDX;
        }
    }
    for (int n = 0; n < nd->mNumChildren; ++n)
        recursiveVertexLoad(sc, nd->mChildren[n],obj);
    
}
inline void getVertexNumber(const struct aiScene *sc,
                                     const aiNode* nd, int& VertxNum)
{
    for (int n=0; n < nd->mNumMeshes; ++n)
    {
        const struct aiMesh* mesh = sc->mMeshes[nd->mMeshes[n]];
        //printf("%d \n",mesh->mNumVertices);
        VertxNum += mesh->mNumVertices;
    }
    for (int n = 0; n < nd->mNumChildren; ++n)
        getVertexNumber(sc, nd->mChildren[n],VertxNum);
}
inline void getVertexFaceNumber(const struct aiScene *sc,
                                         const aiNode* nd, int& VertxNum)
{
    for (int n=0; n < nd->mNumMeshes; ++n)
    {
        const struct aiMesh* mesh = sc->mMeshes[nd->mMeshes[n]];
        for (int t = 0; t < mesh->mNumFaces; ++t)
        {
            const struct aiFace* face = &mesh->mFaces[t];
            
            for (int i = 0; i < face->mNumIndices; ++i)
            {
                ++VertxNum;
            }
        }
        
    }
    for (int n = 0; n < nd->mNumChildren; ++n)
        getVertexFaceNumber(sc, nd->mChildren[n],VertxNum);
}

inline void quat_vect_crossPSOstateVect(const float* quat, const float* vec, float* qXv)
{
    /* qXv = cross(quat.xyz, vec) */
    
    qXv[0] = quat[q2_]*vec[2]-quat[q3_]*vec[1];
    qXv[1] =-(quat[q1_]*vec[2]-quat[q3_]*vec[0]);
    qXv[2] = quat[q1_]*vec[1]-quat[q2_]*vec[0];
    
}

inline void quatVectRotationPSOstateVect(const float* quat, const float* vec, float* pvec)
{
    
    /*	t = 2 * cross(q.xyz, v)
     v' = v + q.w * t + cross(q.xyz, t)
     */
    float t[3];
    float k[3];
    quat_vect_crossPSOstateVect(quat,vec,&t[0]);
    t[0] = 2.0f*t[0];t[1] = 2.0f*t[1];t[2] = 2.0f*t[2];
    
    quat_vect_crossPSOstateVect(quat,t,&k[0]);
    
    pvec[0] = vec[0] + quat[q0_]*t[0] + k[0];
    pvec[1] = vec[1] + quat[q0_]*t[1] + k[1];
    pvec[2] = vec[2] + quat[q0_]*t[2] + k[2];
    
    
    
}

inline void projectModelPointsToPixels(const float* pso_pose_vec, const float* point, uint16_t* UVz)
{
    /* input:
     * @pso_pose_vec: is the particle state vector:
     * 				  pose[tx,ty,tz,q0,q1,q2,q3] in [meter, unit_quat]
     * @point: is a 3D point [in meter]
     * output:
     * @UVz: is the normalized pixel coordinate UV of
     * 		 the projected point and its depth value Z in millimeters.
     */
    /* UVZ = K*T*XYZ; UV = UVZ/Z; */
    //Rotation
    float pprime[3];
    quatVectRotationPSOstateVect(pso_pose_vec,point,&pprime[0]);
    //Translation
    pprime[tx_] += pso_pose_vec[tx_];
    pprime[ty_] += pso_pose_vec[ty_];
    pprime[tz_] += pso_pose_vec[tz_];
    //Projection K and normalize
    /*	|cx*z + fx*x	|		|cx + fx*(x/z)|
     |cy*z + fy*y|  ==> 	|cy + fy*(y/z)|
     |z		    |		|     1		  |
     */
    
    UVz[tx_] = static_cast<uint16_t>( CX + FX*(pprime[tx_]/pprime[tz_]) + tocast_ );
    UVz[ty_] = static_cast<uint16_t>( CY + FY*(pprime[ty_]/pprime[tz_]) + tocast_ );
    UVz[tz_] = static_cast<uint16_t>(tomm_*pprime[tz_] + tocast_);//in mm
    
    return;
}
inline void projectModelPointsToPixels(const float* pso_pose_vec, const float* point, float* UVz)
{
    /* input:
     * @pso_pose_vec: is the particle state vector:
     * 				  pose[tx,ty,tz,q0,q1,q2,q3] in [meter, unit_quat]
     * @point: is a 3D point [in meter]
     * output:
     * @UVz: is the normalized pixel coordinate UV of
     * 		 the projected point and its depth value Z in millimeters.
     */
    /* UVZ = K*T*XYZ; UV = UVZ/Z; */
    //Rotation
    float pprime[3];
    quatVectRotationPSOstateVect(pso_pose_vec,point,&pprime[0]);
    //Translation
    pprime[tx_] += pso_pose_vec[tx_];
    pprime[ty_] += pso_pose_vec[ty_];
    pprime[tz_] += pso_pose_vec[tz_];
    //Projection K and normalize
    /*	|cx*z + fx*x	|		|cx + fx*(x/z)|
     |cy*z + fy*y|  ==> 	|cy + fy*(y/z)|
     |z		    |		|     1		  |
     */
    
    UVz[tx_] =  CX + FX*(pprime[tx_]/pprime[tz_]) + tocast_ ;
    UVz[ty_] =  CY + FY*(pprime[ty_]/pprime[tz_]) + tocast_ ;
    UVz[tz_] = tomm_*pprime[tz_] + tocast_;//in mm
    
    return;
}

inline float min3(const float &a, const float &b, const float &c)
{ return std::min(a, std::min(b, c)); }

inline float max3(const float &a, const float &b, const float &c)
{ return std::max(a, std::max(b, c)); }

const int imageWidth = 640;
const int imageHeight = 480;

inline
float edgeFunction(
        const float* a, const float* b, const float* c)
{
    return(
        (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0]));
}
inline void printVector(const float* w, int size)
{
    printf("[ ");
    for (int i=0; i<size; ++i) {
        if (i==size-1) {
            printf("%f ",w[i]);
        }
        else
            printf("%f, ",w[i]);
    }
    printf("]\n");
}


int main( int argc, char *argv[] )
{
    if (argc != 4) {
        
        printf("usage: %s c1 c2 id\n",argv[0]);
        
        return -1;
    }
    
    
    /* PROVA RENDER */
    /*LOAD THE OBJ MODEL*/
    
//    aiPropertyStore* iStore = aiCreatePropertyStore();
//    aiSetImportPropertyInteger(iStore,AI_CONFIG_PP_RVC_FLAGS,
//                               aiComponent_NORMALS | aiComponent_TANGENTS_AND_BITANGENTS  |
//                               aiComponent_COLORS | aiComponent_TEXCOORDS | aiComponent_BONEWEIGHTS |
//                               aiComponent_ANIMATIONS | aiComponent_TEXTURES | aiComponent_LIGHTS |
//                               aiComponent_CAMERAS | aiComponent_MATERIALS );
//    aiSetImportPropertyInteger(iStore,AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_POINT | aiPrimitiveType_LINE);
////    std::string path_to_obj("/Users/giorgio/Documents/PSOQuatGraphSeg/shampoo_plain_centered/shampoo_plain_centered.obj");
//    std::string path_to_obj("/Users/giorgio/Documents/PSOQuatGraphSeg/shampoo_plain_centered/shampoo_plain_centered.obj");//"/Users/giorgio/Documents/PSOQuatGraphSeg/shampoo60000/shampoo_60000vtx.obj");
//    const aiScene* scene = aiImportFileExWithProperties(//
//        path_to_obj.c_str(),
//        (aiProcess_JoinIdenticalVertices |
//         aiProcess_RemoveComponent |
//         aiProcess_FindDegenerates |
//         aiProcess_SortByPType |
//         aiProcess_FindInvalidData |
//         aiProcess_GenSmoothNormals |
//         aiProcess_OptimizeGraph |
//         aiProcess_OptimizeMeshes),NULL,iStore);
////    const aiScene* scene = aiImportFile(path_to_obj.c_str(),aiProcessPreset_TargetRealtime_Quality);
//
//    
//    if(!scene)
//    {
//        std::cout<<"error !scene\n";
//        return -1;
//    }
//    const aiNode* nd = scene->mRootNode;
//    if(!nd)
//    {
//        std::cout<<"error !nd\n";
//        return -1;
//    }
//    //Usually numFaceVertx ~= 3*numVertx
//    int numVertx=0;
//    int numFaceVertx=0;
//    getVertexFaceNumber(scene,nd,numFaceVertx);
//    printf("numFaceVertex= %d\n",numFaceVertx);
//    getVertexNumber(scene,nd,numVertx);
//    printf("numVertex= %d\n",numVertx);
//    ObjectModel_t obj;
//    obj.x = (float*)calloc(numVertx,sizeof(float));
//    obj.y = (float*)calloc(numVertx,sizeof(float));
//    obj.z = (float*)calloc(numVertx,sizeof(float));
//    obj.idx = (int*)calloc(numFaceVertx,sizeof(int));
//    
//    //recursiveVertexLoad(scene,nd,obj);
//    recursiveVertexIndexLoad(scene,nd,obj);
//    aiReleasePropertyStore(iStore);
////    for (int i=0; i<numFaceVertx; ++i) {
////        
////        printf("%d \n", obj.idx[i]);
////        
////    }
//    
//    float pso_pose_vec[7] = {0.0285172f,-0.16441f,0.815619f,
//        1.f,    0.0f, 0.f,0.f};//0.7071f,    0.7071f, 0.f,0.f};
//        //0.915359f,
////        0.f,//0.393179f,
////        0.f,//-0.0862865f,
////        0.f,-0.00912934f};
//    cv::Mat_<uint16_t> depthBufferMat = cv::Mat_<uint16_t>::zeros(imageHeight,imageWidth);
//    uint16_t* depthBuffer = depthBufferMat.ptr<uint16_t>(0);
////    size_t count=0;
//    
//    for(int vertexIdx=0;vertexIdx<numFaceVertx-3/*numVertx-3*/;vertexIdx+=3)
//    {
////        float Vertex0[3] ={obj.x[vertexIdx],obj.y[vertexIdx],obj.z[vertexIdx]};
////        float Vertex1[3] ={obj.x[vertexIdx+1],obj.y[vertexIdx+1],obj.z[vertexIdx+1]};
////        float Vertex2[3] ={obj.x[vertexIdx+2],obj.y[vertexIdx+2],obj.z[vertexIdx+2]};
//        float Vertex0[3] ={obj.x[obj.idx[vertexIdx]],obj.y[obj.idx[vertexIdx]],obj.z[obj.idx[vertexIdx]]};
//        float Vertex1[3] ={obj.x[obj.idx[vertexIdx+1]],obj.y[obj.idx[vertexIdx+1]],obj.z[obj.idx[vertexIdx+1]]};
//        float Vertex2[3] ={obj.x[obj.idx[vertexIdx+2]],obj.y[obj.idx[vertexIdx+2]],obj.z[obj.idx[vertexIdx+2]]};
//        
//        float UVz0[3];
//        float UVz1[3];
//        float UVz2[3];
//        projectModelPointsToPixels(pso_pose_vec, &Vertex0[0], &UVz0[0]);
//        projectModelPointsToPixels(pso_pose_vec, &Vertex1[0], &UVz1[0]);
//        projectModelPointsToPixels(pso_pose_vec, &Vertex2[0], &UVz2[0]);
//        UVz0[2] = 1.f/UVz0[2];
//        UVz1[2] = 1.f/UVz1[2];
//        UVz2[2] = 1.f/UVz2[2];
////        printVector(UVz0,3);
////        printVector(UVz1,3);
////        printVector(UVz2,3);
//        
//        //Triangle Bounding Box  AABB!!
//        float xmin = min3(UVz0[0], UVz1[0], UVz2[0]);
//        float ymin = min3(UVz0[1], UVz1[1], UVz2[1]);
//        float xmax = max3(UVz0[0], UVz1[0], UVz2[0]);
//        float ymax = max3(UVz0[1], UVz1[1], UVz2[1]);
//        
//        // the triangle is out of screen
////        if (xmin > imageWidth - 1 || xmax < 0 || ymin > imageHeight - 1 || ymax < 0) continue;
//        
//        // be careful xmin/xmax/ymin/ymax can be negative. Don't cast to uint32_t
//        //starting corner of the AABB
////        uint32_t x0 = std::max(int32_t(0), (int32_t)(std::floor(xmin)));
////        uint32_t x1 = std::min(int32_t(imageWidth) - 1, (int32_t)(std::floor(xmax)));
////        uint32_t y0 = std::max(int32_t(0), (int32_t)(std::floor(ymin)));
////        uint32_t y1 = std::min(int32_t(imageHeight) - 1, (int32_t)(std::floor(ymax)));
//        
//        float area = edgeFunction(&UVz0[0], &UVz1[0], &UVz2[0]);
//        //printf("area: %f\n",area);
//        //////
//        for (uint32_t y = ymin; y <= ymax; ++y) {
//        for (uint32_t x = xmin; x <= xmax; ++x) {
//            float pixelSample[3] = {x + 0.5f, y + 0.5f, 0};
////            printf("pixelSample[0]: %f pixelSample[1]: %f\n",pixelSample[0],pixelSample[1]);
//            //lambda: for Z and attributes interpolation
//            float w0 = edgeFunction(UVz1, UVz2, pixelSample);
//            //printf("w0: %f\n",w0);
//            float w1 = edgeFunction(UVz2, UVz0, pixelSample);
//            //printf("w1: %f\n",w1);
//            float w2 = edgeFunction(UVz0, UVz1, pixelSample);
//            //printf("w2: %f\n",w2);
//            if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
//                //printf("w0\n");
//                w0 /= area;
//                w1 /= area;
//                w2 /= area;
//                float oneOverZ = UVz0[2] * w0 + UVz1[2] * w1 + UVz2[2] * w2;
//                float z = 1.f / oneOverZ;
//                // [comment]
//                // Depth-buffer test
//                // [/comment]
////                if (z < depthBuffer[y * imageWidth + x] || depthBuffer[y * imageWidth + x] == 0)
////                {
////                    depthBuffer[y * imageWidth + x] = (uint16_t)z;
////                    //printf("z: %f\n",z);
////                    
////                }
//                depthBuffer[y * imageWidth + x] = (
//                (z < depthBuffer[y * imageWidth + x] || depthBuffer[y * imageWidth + x] == 0)*(uint16_t)z) | (!(z < depthBuffer[y * imageWidth + x] || depthBuffer[y * imageWidth + x] == 0))*depthBuffer[y * imageWidth + x];
//            }
//        }
//    }
//
//        
//        /////
//        
//    }//end for triplet of vertices (triangle)
//    
//    
//    cv::Mat color_depth;
//    
//    GraphCanny::GraphCannySeg<GraphCanny::hsv>::convertDepth2ColorMap(depthBufferMat, color_depth);
//    
//    GraphCanny::GraphCannySeg<GraphCanny::hsv>::visualizeColorMap(depthBufferMat,"colorrender",5,false);
//    cv::imshow("render", color_depth);
//    //    cv::imshow("render linear", linear_color_depth);
//    cv::waitKey();
//    
//    
//    
//    return 0;
//    
////    size_t array_size=0;
////    unsigned int pxIdxVec[15000];
////    unsigned int depthValVec[15000];
//
////    for(int vertexIdx=0;vertexIdx<numVertx;++vertexIdx)
////    {
////        
////        float VertexA[3] ={obj.x[vertexIdx],obj.y[vertexIdx],obj.z[vertexIdx]};
////        
////        uint16_t UVz[3];
////        projectModelPointsToPixels(pso_pose_vec, &VertexA[0], &UVz[0]);
////        
////        unsigned int pxIdx = UVz[1]*640 + UVz[0];
////        
////        //JUST for GroundTruth
////        if(depth_b(UVz[1],UVz[0]) > UVz[2] ||
////           depth_b(UVz[1],UVz[0])==0)
////        {
////            //only if buffer==0 it's a real render!!
////            if(depth_b(UVz[1],UVz[0])==0)
////                ++count;
////            depth_b(UVz[1],UVz[0]) = UVz[2];
////        }
//// /*
////        if(vertexIdx==0)
////        {
////            pxIdxVec[0] = pxIdx;
////            depthValVec[0] = UVz[2];
////            ++array_size;
////            continue;
////        }
////        //search
////        bool found = false;
////        for(int i=0;i<array_size;++i)
////        {
////            if(pxIdxVec[i]==pxIdx)
////            {
////                if(depthValVec[i]>UVz[2] || depthValVec[i]==0)
////                {
////                    depthValVec[i] = UVz[2];
////                }
////                found=true;
////                break;
////            }
////        }
////        if(!found)
////        {
////            //then insert
////            pxIdxVec[array_size] = pxIdx;
////            depthValVec[array_size] = UVz[2];
////            ++array_size;
////        }
////        
////   */
////        
////    }
////    printf("array_size: %zu\n",array_size);
////    printf("pixel_rendered: %zu\n",count);
////    cv::Mat color_depth;
////    
////    GraphCanny::GraphCannySeg<GraphCanny::hsv>::convertDepth2ColorMap(depth_b, color_depth);
//    
//    //test
////    cv::Mat_<uint16_t> lineradepth = cv::Mat_<uint16_t>::zeros(480,640);
////    uint16_t* lineradepthPtr = lineradepth.ptr<uint16_t>(0);
////    for(int i=0;i<array_size;++i)
////    {
////        lineradepthPtr[pxIdxVec[i]] = depthValVec[i];
////    }
////    cv::Mat linear_color_depth;
////    GraphCanny::GraphCannySeg<GraphCanny::hsv>::convertDepth2ColorMap(lineradepth, linear_color_depth);
////    
////    cv::imshow("render", color_depth);
//////    cv::imshow("render linear", linear_color_depth);
////    cv::waitKey();
////    
////    
////    
////    return 0;
//    
//    /* END PROVA RENDER */
//    
//
//    
//    //OPEN yml
//    //Load RGB + DEPTH from kinect
////    cv::FileStorage fs("/Users/giorgio/Documents/Polito/PhD/Tesisti/Calib/prova0.yml",cv::FileStorage::READ);
////    cv::Mat depth_m, depth_mm;
////    //fs["depth_gray_img"]>>depth_img;
////    fs["depth_mm"]>>depth_m;
////    depth_m.convertTo(depth_mm, CV_16U,1000);
////    cv::imwrite("/Users/giorgio/Documents/Polito/PhD/Tesisti/Calib/depth_mm0.png", depth_mm);
////    
////    //fs["rgb"]>>image;
////    //image = cv::imread("../ImagesAndModels/color.jpg");
////    //fs["xxx"]>>depth_m;
////    //depth_m = cv::imread("../ImagesAndModels/depth16bits.yml");
////    fs.release();
////    exit(0);
//    
//    
//    
//    
    initTrackbarsSegmentation();
    
    PSOClass* PSO_;
    cv::Mat SegResRGB;
    cv::Mat SegResDepthM;
    cv::Mat SegResDepthRGB;
    const float offset_roll = M_PI/2; //M_PI;//-M_PI/3.;//0.f;//M_PI;
    const float offset_pitch = 0;
    const float offset_yaw = 0;//M_PI;
    std::string IDcode = "10";//argv[3];
    std::string nomefilelog = "/Users/morpheus/Desktop/QpsoAndSeg/LogFiles/log" + IDcode + ".txt";
    std::string nomefilelogFinalResult = "/Users/morpheus/Desktop/QpsoAndSeg/LogFinalResults/log" + IDcode + ".dat";
//    std::string path_to_obj("/Users/giorgio/Documents/PSOQuatGraphSeg/shampoo/shampoo_plain.obj"/*"/Users/giorgio/Documents/PSOQuatGraphSeg/coffee_cup_zdown/coffee_cup_plain.obj"*/);
    
    //std::string path_to_obj("/Users/morpheus/Downloads/good/coffee_cup/coffee_cup_plain_centered.obj");
    //std::string path_to_obj("/Users/morpheusd/Downloads/shampooDecimated/shampooDecimated1024.obj");
    //std::string path_to_obj("/Users/morpheus/Downloads/good/juicecarton3072.obj");
    //std::string path_to_obj("/Users/morpheus/Downloads/apc_main/object_models/tarball/kong_duck_dog_toy.obj");
    //std::string path_to_obj("/Users/morpheus/Downloads/milkcarton3072_orig.obj");
    //std::string path_to_obj("/Users/morpheus/Downloads/cupboard/cupboard000.obj");

    //std::string path_to_obj("/Volumes/HD-PNTU3/datasets/ACCV/duck/duck.obj");

    
    std::string path_to_obj1("/Users/morpheus/Downloads/coffeecup3072.obj");
    std::string path_to_obj2("/Users/morpheus/Downloads/good/coffee_cup/coffe_cup_plain.obj");
//    std::string path_to_obj1("/Users/morpheus/Downloads/laptop/000.obj");
//    std::string path_to_obj2("/Users/morpheus/Downloads/laptop/001.obj");

    
    int IDX_cluster = 0;
    std::cout<<"centroid IDX_cluster: "<< IDX_cluster <<" "<<vecSegResult[IDX_cluster].centroid3D<<"\n------\n";
    
    
//    //try AABB
//    std::vector<cv::Point2f> bbpp;
//    for (int i=0; i<vecSegResult[IDX_cluster].num_points; ++i) {
//        
//        cv::Point2f p2(vecSegResult[IDX_cluster].pxs[i].x,
//                       vecSegResult[IDX_cluster].pxs[i].y);
//        
//        bbpp.push_back(p2);
//        
//    }
//    cv::Rect rect_ = cv::boundingRect(bbpp);
//    cv::Mat imageRect = cv::Mat::zeros(480, 640, CV_8UC1);
//    cv::rectangle(imageRect, rect_, cv::Scalar(255));
//    printf("Rect H: %d, W: %d, Area: %d\n",rect_.height,rect_.width,rect_.area());
//    IDX_cluster=vecSegResult.size()-1;
//    for(int i=0;i<vecSegResult.size();++i)
//    {
//        cv::Rect& rect_ = vecSegResult[i].rect_aabb_;
//        cv::rectangle(vecSegResult[i].clusterRGB,rect_, cv::Scalar(255));
//        printf("idx: %d\n",i);
//        printf("Rect H: %d, W: %d, Area: %d, tl: [%d , %d], br: [%d , %d] \n",rect_.height,rect_.width,rect_.area(), rect_.tl().x, rect_.tl().y, rect_.br().x, rect_.br().y);
//        cv::imshow("rect", vecSegResult[i].clusterRGB);
//        
//        unsigned c=cv::waitKey(0);
//        if(c==13) {IDX_cluster=i; break;}
//    }
    
    
    
    
//    cv::Mat SSS = cv::Mat(480,640,CV_16U,&savedmat[0]);
//    cv::Mat depthRGBdebug;
//    GraphCanny::GraphCannySeg<GraphCanny::hsv>::convertDepth2ColorMap(SSS,depthRGBdebug);
//    cv::imshow("DEBDepthRGB", depthRGBdebug);
    
    
    
//    GraphCanny::GraphCannySeg<GraphCanny::hsv>::save1chMat2HeaderVector(
//            vecSegResult[IDX_cluster].clusterDepth,
//            vecSegResult[IDX_cluster].centroid3D,
//            "/Users/giorgio/cuda-workspace/QuatPSO/shampoosegdepth.h");
//    
    //cv::waitKey(0);
    
    initPSOClass(PSO_, IDX_cluster);
    
    //DEBUG EACH CLUSTER
   /*
    for (int i=0; i<vecSegResult.size(); ++i) {
        
        cv::imshow("DEBIDX", vecSegResult[i].clusterRGB);
        
        cv::Mat depthRGBdebug;
        GraphCanny::GraphCannySeg<GraphCanny::hsv>::convertDepth2ColorMap(vecSegResult[i].clusterDepth,depthRGBdebug);
        cv::imshow("DEBDepthRGB", depthRGBdebug);
        
        std::cout<<"i: "<<i<<"\n";
        std::cout<<"centroid: "<<vecSegResult[i].centroid3D<<"\n------\n";
        
        cv::waitKey();
        
    }
    */
    
    
    //init PSO render
    PSOrender psoR(PSO_,gcs, nomefilelog, nomefilelogFinalResult, IDcode, path_to_obj1, path_to_obj2,k_vec, offset_roll, offset_pitch, offset_yaw, IDX_cluster);
    
    
    bool pso_converged = false;
    //timing
    psoR.setStartClock();
    while(!pso_converged)
    {
        pso_converged =  psoR.display();
        
    }
    psoR.closeLogFiles();
    
    
    
    //for each particle
    for(int p=0; p<PSO_->settings.size; ++p)
    {
       
        cv::Size S = cv::Size(640,480);
        std::string fname = "/Users/morpheus/Desktop/QpsoAndSeg/movies/p" + std::to_string(p) + "_ID" +IDcode+".mov";
        cv::VideoWriter vw(fname,
                           CV_FOURCC_MACRO('m','p','4','v'),
                           5,
                           S,
                           true);
        if (!vw.isOpened())
        {
            std::cout  << "Could not open the output video for write: " << std::endl;
            return -1;
        }
        
        //for each step
        for(int i=0; i<PSO_->settings.steps; ++i)
        {
//            cv::Mat P1 = cv::Mat::zeros(480,640,CV_8UC3);
//            for(int j=0; j<psoR.mParticleTraj[i].size(); ++j)
//            {
//                P1 += (1.f/((float)psoR.mParticleTraj[i].size()))*psoR.mParticleTraj[i][j];
//            }
            
            vw.write(psoR.mParticleTraj[i][p]);
        }//end for each step
        vw.release();
    }//end for each particle
    //cv::imshow("Particle N_0 Trajectory", P1);
    
    cv::waitKey();
    
    delete PSO_;
    delete gcs;
    
    return 0;
}

