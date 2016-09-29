//
//  main.cpp
//  PSOQuatGraphSeg
//
//  Created by Giorgio on 25/10/15.
//  Copyright (c) 2015 Giorgio. All rights reserved.

#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include "opencv2/opencv.hpp"

#include <math.h>
#include <string>
#include <GLUT/glut.h>
//#include <GLUT/glew.h>
//#ifdef __APPLE__
//#include <GL/glew.h>//include before GL/glut.h and it contains GL/gl.h and GL/glu.h
//#include <GLUT/glut.h>
//#else
////#include <GL/gl.h>
//#include <GL/glew.h>//include before GL/glut.h and it contains GL/gl.h and GL/glu.h
//#include <GL/glut.h>
//#endif

#include <cstdio>

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "model.h"
//#include <gsl/gsl_rng.h>
#include "pso_class_quaternions.h"

#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>

#include <fstream>
#include <time.h>

#include <algorithm>    // std::min_element, std::max_element

#include "GraphCannySeg.h"

//#include "pso_opengl_render.h"

using namespace glm;


int width = 640;
int height = 480;
const int wXh = width*height;
double zmin = 0.1;
double zmax = 1000.0;

//RGB params ASUS TESI
//double fx = 539.203994707613;
//double fy = 536.705519252862;
//double cx = 311.401125620835;
//double cy = 241.48563618046;

//RGB ASUS GIORGIO
//double fx = 549.8673626911883;
//double fy = 550.4240710445896;
//double cx = 323.6734992813105;
//double cy = 230.2912939215161;

//CHALLENGE 1 DATASET
double fx = 571.9737;
double fy = 571.0073;
double cx = 319.5000;
double cy = 239.5000;
float k_vec[9] = {static_cast<float>(fx), 0, static_cast<float>(cx), 0, static_cast<float>(fy), static_cast<float>(cy), 0.f,0.f,1.f};



int stepImage = 10;

cv::Mat image;
cv::Mat depth_img, depth_m;
cv::Mat depth_kinect_rgb;

// Create images to copy the buffers to
cv::Mat_ < cv::Vec3b > renderedImage(height, width);


/*Distance ERROR*/
cv::Mat canny_rgb_kinect,dist_background,dist_8uc3;

/** The frame buffer object used for offline rendering */
GLuint fbo_id_;
/** The render buffer object used for offline depth rendering */
GLuint rbo_id_;
/** The render buffer object used for offline image rendering */
GLuint texture_id_;
/** Model Obj+mtl to load **/
boost::shared_ptr<Model> model_;

GLuint scene_list_;


std::ofstream logfile , logFinalResult;
clock_t tStart;


const float alfa_paramErrorDT = 1.5f;
std::string IDcode;

std::vector<std::vector<cv::Point3f> >  mGridCornersin3D;

/** FORWARD DECLARATION **/
void drawAxes(float length);
bool myInitAllThings(int argc_, char **argv_);
void clean_buffers();
void bind_buffers();
bool display();
void dummy_display();
void drawImageGrid(cv::Mat& mat,
                   std::vector<std::vector<cv::Point3f> >& GridCornersin3D,
                   float subHeight=3.f,
                   float subWidth=3.f);
/**END FORWARD DECLARATION **/

PSOClass* PSO_;

double psoc1;
double psoc2;

int main( int argc, char *argv[] )
{
    if (argc != 4) {
        
        printf("usage: %s c1 c2 id\n",argv[0]);
        
        return -1;
    }
    
    bool initStatus = myInitAllThings(argc,argv);
    if(!initStatus)
    {
        return -1;
    }
    
    //overwrite the default settings
     psoc1 = atof(argv[1]);
     psoc2 = atof(argv[2]);
     IDcode = "1";//argv[3];
    
    bool pso_converged = false;
    //timing
    tStart = clock();
    while(!pso_converged)
    {
        pso_converged =  display();
        std::cout<<PSO_->settings.step<<"\n";
        std::cout<<PSO_->mTs<<"\n";
        //cv::waitKey();
//        double ts_ = PSO_->getSamplingTime();
//        ts_*=0.4;
//        if(ts_<0.001)
//            ts_=0.001;
//        PSO_->setSamplingTime(ts_);
        //cv::waitKey();
    }
    
    //cv::waitKey();
    logfile.close();
    logFinalResult.close();
    
    //TODO: Delete the PSO double pointers
    
    return 0;
}


/*Segmentation Params*/
//params
int k=67;//30; //0.003 /10000
int kx=2000;//2.0f;
int ky=30;//0.03f;
int ks=50;//0.05f;
float kdv=4.5f;
float kdc=0.1f;
float min_size=500.0f;
float sigma=0.8f;
float max_ecc = 0.978f;
float max_L1 = 3800.0f;
float max_L2 = 950.0f;

int DTH = 30; //[mm]
int plusD = 7; //for depth boundary
int point3D = 5; //10//for contact boundary
int g_angle = 148;//2.f/3.f*M_PI;
int l_angle = 56; //M_PI/3.f;
int Lcanny = 50;
int Hcanny = 75;
int FarObjZ = 875;//1800; //[mm]

std::string trackBarsWin = "Trackbars";
cv::Mat kinect_rgb_img;
cv::Mat kinect_depth_img_mm;

//Segmentation Results
std::vector<GraphCanny::SegResults> vecSegResult;

void on_trackbar( int, void* )
{
    
    float kfloat = (float)k/10000.f;
    float kxfloat = (float)kx/1000.f;
    float kyfloat = (float)ky/1000.f;
    float ksfloat = (float)ks/1000.f;
    float gafloat = ((float)g_angle)*deg2rad;
    float lafloat = ((float)l_angle)*deg2rad;
    float lcannyf = (float)Lcanny/1000.f;
    float hcannyf = (float)Hcanny/1000.f;
    GraphCanny::GraphCannySeg<GraphCanny::hsv> gcs(kinect_rgb_img, kinect_depth_img_mm, sigma, kfloat, min_size, kxfloat, kyfloat, ksfloat,k_vec,lcannyf,hcannyf,kdv, kdc,max_ecc,max_L1,max_L2,(uint16_t)DTH,(uint16_t)plusD,(uint16_t)point3D,gafloat,lafloat,(float)FarObjZ);
    
    gcs.run();
    
    vecSegResult = gcs.vecSegResults;
    
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



bool myInitAllThings(int argc_, char **argv_)
{
    
    /* First Load the RGB and DEPTH */
    //Open Images
    
    //CHALLENGE DATASET-1
    
    std::string rgb_name= "img_1164.png";//"img_163.png";
    std::string obj_name= "Coffee_Cup";//"Shampoo";
    std::string rgb_file_path = "/Users/giorgio/Downloads/"+obj_name+"/RGB/"+rgb_name;
    std::string depth_file_path = "/Users/giorgio/Downloads/"+obj_name+"/Depth/"+rgb_name;
    cv::Mat imageMask = cv::Mat();
    
/*
    //RGB-D DATASET
     std::string png_name= "table_1_108";//"img_163.png";
     
     std::string rgb_file_path = "/Users/giorgio/Downloads/rgbd-scenes_all/table/table_1/"+png_name+".png";
     std::string depth_file_path = "/Users/giorgio/Downloads/rgbd-scenes_all/table/table_1/"+png_name+"_depth.png";
     cv::Mat imageMask = cv::Mat();
*/
    kinect_rgb_img = cv::imread(rgb_file_path,cv::IMREAD_UNCHANGED);
    kinect_depth_img_mm = cv::imread(depth_file_path,cv::IMREAD_UNCHANGED);// in mm

    
    /*Create the TrackBars for Segmentation Params*/
    cv::namedWindow(trackBarsWin,0);
    
    cv::createTrackbar("k", trackBarsWin, &k, 1000,on_trackbar);
    cv::createTrackbar("kx", trackBarsWin, &kx, 10000,on_trackbar);
    cv::createTrackbar("ky", trackBarsWin, &ky, 1000,on_trackbar);
    cv::createTrackbar("ks", trackBarsWin, &ks, 1000,on_trackbar);
    cv::createTrackbar("DTH", trackBarsWin, &DTH, 100,on_trackbar);
    cv::createTrackbar("plusD", trackBarsWin, &plusD, 100,on_trackbar);
    cv::createTrackbar("Point3D", trackBarsWin, &point3D, 100,on_trackbar);
    cv::createTrackbar("G Angle", trackBarsWin, &g_angle, 180,on_trackbar);
    cv::createTrackbar("L Angle", trackBarsWin, &l_angle, 180,on_trackbar);
    cv::createTrackbar("H Canny th", trackBarsWin, &Hcanny, 100,on_trackbar);
    cv::createTrackbar("L Canny th", trackBarsWin, &Lcanny, 100,on_trackbar);
    cv::createTrackbar("FarObjZ", trackBarsWin, &FarObjZ, 2500,on_trackbar);
    
    on_trackbar( 0, 0 );
    
    /// Wait until user press some key
    cv::waitKey(0);


    //cv::cvtColor(kinect_depth_img, depth_kinect_rgb, CV_GRAY2RGB);
    
    /*** HereAfter we use the result of the segmentation ***/
    //Pick a Cluster : eg. cluster 1

    //TODO: For fast test we copy the result to the already present matrices....to be fixed
    int IDX_cluster=0;
    GraphCanny::GraphCannySeg<GraphCanny::hsv>::convertDepth2ColorMap(vecSegResult[IDX_cluster].clusterDepth,depth_kinect_rgb);
    vecSegResult[IDX_cluster].clusterRGB.copyTo(image);
    //conversion from mm to m
    vecSegResult[IDX_cluster].clusterDepth.convertTo(depth_m, CV_32F,1.e-3f);
    
    
    //INIT THE QUATERNION PSO
    //PSO CLASS
    
    std::vector<double> Dimbounds;
    float t_bound = 0.04f;//0.04f; //in m
    float t_bound_z = 0.10f; //in m
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
    
    double Tss = 0.5;
    double c1__ = 1;//1.496;
    double c2__ = 1;//1.496;
    unsigned int dynamicInertia = PSOClass::PSO_W_LIN_DEC;//PSOClass::PSO_W_CONST;//
    unsigned int nhoodtype = PSOClass::PSO_NHOOD_GLOBAL;//PSOClass::PSO_NHOOD_RING;//;
    int nhood_size = 5;//(Only for Random Topology)
    PSO_ = new PSOClass(/*Ndim,NSize,*/Dimbounds,c1__,c2__,
                        Tss,dynamicInertia,nhoodtype/*,nhood_size*/);
    
    //IDcode = idsrc;
    std::string nomefilelog = "/Users/giorgio/Documents/PSOQuatGraphSeg/LogFiles/log" + IDcode + ".txt";
    //open the Log File
    logfile.open(nomefilelog);
    std::string nomefilelogFinalResult = "/Users/giorgio/Documents/PSOQuatGraphSeg/LogFinalResults/log" + IDcode + ".dat";
    logFinalResult.open(nomefilelogFinalResult);
    //Save all the PSO settings
    logfile << "c1 " << PSO_->settings.c1 << "\n"
    << "c2 " << PSO_->settings.c2 << "\n"
    << "ProblemDim " << PSO_->settings.dim << "\n"
    << "ParticleNumber " << PSO_->settings.size << "\n"
    << "ParticleBounds[meters][unit quat] \n"
    << "\t tx " << PSO_->settings.x_lo[0] << " , " << PSO_->settings.x_hi[0] << "\n"
    << "\t ty " << PSO_->settings.x_lo[1] << " , " << PSO_->settings.x_hi[1] << "\n"
    << "\t tz " << PSO_->settings.x_lo[2] << " , " << PSO_->settings.x_hi[2] << "\n"
    << "\t q0 " << PSO_->settings.x_lo[3] << " , " << PSO_->settings.x_hi[3] << "\n"
    << "\t q1 " << PSO_->settings.x_lo[4] << " , " << PSO_->settings.x_hi[4] << "\n"
    << "\t q2 " << PSO_->settings.x_lo[5] << " , " << PSO_->settings.x_hi[5] << "\n"
    << "\t q3 " << PSO_->settings.x_lo[6] << " , " << PSO_->settings.x_hi[6] << "\n"
    << "MaxSteps " << PSO_->settings.steps << "\n"
    << "SamplingTime Ts " << PSO_->mTs << "\n"
    << "goal_th " << PSO_->settings.goal << "\n"
    << "w_max " << PSO_->settings.w_max << "\n"
    << "w_min " << PSO_->settings.w_min << "\n"
    << "nhood_strategy " << PSO_->settings.nhood_strategy << "\n"
    << "\t PSO_NHOOD_GLOBAL " << PSO_->PSO_NHOOD_GLOBAL << "\n"
    << "\t PSO_NHOOD_RING " << PSO_->PSO_NHOOD_RING << "\n"
    << "\t PSO_NHOOD_RANDOM " << PSO_->PSO_NHOOD_RANDOM << "\n"
    << "w_strategy " << PSO_->settings.w_strategy << "\n"
    << "\t PSO_W_CONST " << PSO_->PSO_W_CONST << "\n"
    << "\t PSO_W_LIN_DEC " << PSO_->PSO_W_LIN_DEC << "\n"
    << "alfa_paramErrorDT " << alfa_paramErrorDT << "\n"
    << "/***************************************/\n\n";


    clean_buffers();
    
    // initialize GLUT
    glutInit( &argc_, argv_ );
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH );
    glutInitWindowPosition( 20, 20 );
    glutInitWindowSize( width, height );
    glutCreateWindow( "OpenGL / OpenCV Example" );
    
    
    //Init GLEW to avoid Seg. fault
    /* with MAC no needed...
     GLenum err = glewInit();
     if (GLEW_OK != err)
     {
     // Problem: glewInit failed, something is seriously wrong.
     fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
     return false;
     }
     fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));
     
     */
    /** set low level params **/
    // create a framebuffer object
    glGenFramebuffers(1, &fbo_id_);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_id_);
    
    // create a texture object
    glGenTextures(1, &texture_id_);
    glBindTexture(GL_TEXTURE_2D, texture_id_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture_id_, 0);
    
    // create a renderbuffer object to store depth info
    glGenRenderbuffers(1, &rbo_id_);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo_id_);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo_id_);
    /** END set low level params **/
    
    std::string path_to_obj("/Users/giorgio/Documents/PSOQuatGraphSeg/coffee_cup_zdown/coffee_cup_plain.obj");//("/Users/giorgio/Documents/PSOQuatGraphSeg/coffee_cup_new/coffee_cup_plain.obj");//("/Users/giorgio/Documents/PSOQuatGraphSeg/coffee_cup/coffe_cup_plain.obj");//"/Users/giorgio/Downloads/coffee_cup_blender/coffe_cup_plain.obj");//"/Users/giorgio/Downloads/good/coffee_cup_plain_mod.obj"/*"/Users/giorgio/OCVOGL_6d_pose_est/ImagesAndModels/crayola_64_ct.obj"*/);
    model_.reset(new Model());
    model_->LoadModel(path_to_obj);
    /** INIT THE ENVIRONMENT **/
    glClearColor(0.f, 0.f, 0.0f, 1.f);
    
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0); // Uses default lighting parameters
    
    glEnable(GL_DEPTH_TEST);
    
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
    glEnable(GL_NORMALIZE);
    
    GLfloat LightAmbient[]= { 1.0f, 1.0f, 1.0f, 1.0f };
    GLfloat LightDiffuse[]= { 1.0f, 1.0f, 1.0f, 1.0f };
    GLfloat LightPosition[]= { 0.0f, 0.0f, 0.7f, 1.0f };
    glLightfv(GL_LIGHT1, GL_AMBIENT, LightAmbient);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, LightDiffuse);
    glLightfv(GL_LIGHT1, GL_POSITION, LightPosition);
    glEnable(GL_LIGHT1);
    //Added to work with Challenge1 models modified to obtain the .obj
    glDisable(GL_LIGHTING);
    
    /** END INIT THE ENVIRONMENT  **/
    
    /** INIT THE PROJECTION MATRIX **/
    //set projection matrix using intrinsic camera params
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    //set projection based on OpenCV Intrinsic Matrix
    GLdouble perspMatrix[16]={2*fx/width,0,0,0,0,-2*fy/height,0,0,2*(cx/width)-1,1-2*(cy/height),(zmax+zmin)/(zmax-zmin),1,0,0,2*zmax*zmin/(zmin-zmax),0};
    glMultMatrixd(perspMatrix);
    //set viewport
    glViewport(0, 0, width, height);
    /** END INIT THE PROJECTION MATRIX **/
    
    /** Set scene_list_ to 0 **/
    scene_list_=0;
    
    // set up GUI callback functions
    glutDisplayFunc( dummy_display );
    //glutReshapeFunc( reshape );
    //glutMouseFunc( mouse );
    //glutKeyboardFunc( keyboard );
    //glutIdleFunc( idle );
    
    
    /* TEST QUAT CLASS OK !!!!! */
    /*
    double p[] = {0.9227,    0.3822,    0.0191,    0.0462};
    double q[] = {0.9587,    0.2371 ,   0.0848,    0.1324};
    double q_tilde[4]={};
    double p2[] = {0, 0, 0, 0.9227,    0.3822,    0.0191,    0.0462};
    double q2[] = {0, 0, 0, 0.9587,    0.2371 ,   0.0848,    0.1324};
    double q_tilde2[4]={};
    double q_tilde3[4]={};
    PSO_->QuatProd_pq(p, q, q_tilde);
    //should be: q_tilde = {0.7863    0.5838    0.0569    0.1943}
    PSO_->printQuat(q_tilde);
    PSO_->QuatProd_pqPSOstateVect(p2, q2, q_tilde2);
    PSO_->printQuat(q_tilde2);
    PSO_->generalQuatProduct(p,1,q,1,q_tilde3);//pXq
    PSO_->printQuat(q_tilde3);
    PSO_->generalQuatProductPSOStateVect(p2, 1, q2, 1, q_tilde2);
    PSO_->printQuat(q_tilde2);
    //cv::waitKey();
    PSO_->QuatProd_pq_1(p, q, q_tilde);
    //should be: q_tilde = {0.9830    0.1490   -0.0203   -0.1058}
    PSO_->printQuat(q_tilde);
    PSO_->QuatProd_pq_1PSOstateVect(p2, q2, q_tilde2);
    PSO_->printQuat(q_tilde2);
    PSO_->generalQuatProduct(p,1,q,-1,q_tilde3);//pXq^-1
    PSO_->printQuat(q_tilde3);
    PSO_->generalQuatProductPSOStateVect(p2, 1, q2, -1, q_tilde2);
    PSO_->printQuat(q_tilde2);
    //cv::waitKey();
    double qq[] = {0.8,-0.1,0.3,0.04};
    double qq2[] = {0,0,0,0.8,-0.1,0.3,0.04};
    PSO_->NormalizeQuaternion(qq);
    PSO_->NormalizeQuaternionPSOstateVect(qq2);
    //should be: {0.9290   -0.1161    0.3484    0.0464}
    PSO_->printQuat(qq);
    PSO_->printQuatPSOstateVect(qq2);
    //PSO_->printVector(qq2, 7);
    
    //Quat Kinematics
    double w[] = {0, 5, 8, -1};
    PSO_->printVector(w, 4);
    PSO_->printQuat(p);
    PSO_->QuatKinematics(p,w,q_tilde2);//inside-> w.*Ts
    PSO_->printQuat(q_tilde2);
    double w2[] = {0,0,0, 0, 5, 8, -1};
    double qnext[7]={};
    PSO_->QuatKinematicsPSOStateVectWithInit(p2,qnext,w2);
    PSO_->printQuatPSOstateVect(qnext);
//    PSO_->QuatKinematicsPSOStateVect(p2,w2);
//    PSO_->printQuatPSOstateVect(p2);
    cv::waitKey();
     */
    /* END TEST QUAT CLASS  OK !!!!! */
    
    
    return true;
}

const float offset_roll = -M_PI/3.;//0.f;//M_PI;
const float offset_pitch = 0.f;//M_PI;
const float offset_yaw = 0.f;//M_PI;
bool display()
{
    /** For Each Step: settings.step **/
    //TODO: PRE-INIT !!!
    cv::Mat depth_gray_img, depth_color_img, renderedImageCropped;
    
    
    //RECOMPUTE the best images file names
    std::string bestParticleRGBName = "/Users/giorgio/Documents/PSOQuatGraphSeg/BestImages/rgb_"+ IDcode + "_" + boost::lexical_cast<std::string>(PSO_->settings.step-1) + ".png";
    std::string bestParticleDepthName = "/Users/giorgio/Documents/PSOQuatGraphSeg/BestImages/depth_"+ IDcode + "_" + boost::lexical_cast<std::string>(PSO_->settings.step-1) + ".png";
    
    //UPDATE Weight FCN
    if (PSO_->settings.w_strategy==PSOClass::PSO_W_LIN_DEC)
        PSO_->setInertia( PSO_->calc_inertia_lin_dec(PSO_->settings.step, &PSO_->settings) );
    std::cout<<"W: "<<PSO_->getInertia();
    //cv::waitKey();
    //Terminate?? Best Solution Found
    if (PSO_->solution.error <= PSO_->settings.goal) {
        //Stop the timing
        ////printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
        double tElapsed = (double)(clock() - tStart)/CLOCKS_PER_SEC;
        logfile <<"ElapsedTimeWithPlotting "<< tElapsed << "\n";
        // SOLVED!!
        //printf("Goal achieved @ step %d (error=%.3e) \n", settings.step-1, PSO_->solution.error);
        logfile <<"Goal achieved @ step "<< PSO_->settings.step-1 << "error " << PSO_->solution.error <<"\n";
        //printf("Solution Vector: \n");
        logfile <<"SolutionVector\n";
        for (size_t i=0; i<PSO_->settings.dim; ++i) {
            //printf("%f \n",PSO_->solution.gbest[i]);
            logfile <<PSO_->solution.gbest[i]<<"\n";
        }
        //save the best particle image "ID_nstep.jpg";
        
        cv::imwrite(bestParticleRGBName,renderedImage*0.5 + image*0.5);
        cv::imwrite(bestParticleDepthName,renderedImage*0.5 + depth_kinect_rgb*0.5);
        
        //save the best images file path in the log file
        logfile << "BestRGBimage " << bestParticleRGBName << "\n";
        logfile << "BestDepthimage " << bestParticleDepthName << "\n";
        
        //            LogFinalResult << "c1 " << settings.c1 << "\n"
        //                           << "c2 " << settings.c2 << "\n"
        //                           << "error " << PSO_->solution.error;
        logFinalResult << PSO_->settings.c1 << "," << PSO_->settings.c2 << "," << PSO_->solution.error ;
        
        return true;
    }//end IF error < goal
    
    //Terminate?? Max Steps Reached
    if (PSO_->settings.step > PSO_->settings.steps) {
        //Max Steps Reached...
        //Stop the timing
        double tElapsed = (double)(clock() - tStart)/CLOCKS_PER_SEC;
        logfile <<"ElapsedTimeWithPlotting "<< tElapsed << "\n";
        
        //printf("Max Steps Reached @ step %d (error=%.3e) \n", PSO_->settings.step-1, PSO_->solution.error);
        logfile <<"Max Steps Reached @ step "<< PSO_->settings.step-1 << "error " << PSO_->solution.error <<"\n";
        //printf("Solution Vector: \n");
        logfile <<"SolutionVector\n";
        for (size_t i=0; i<PSO_->settings.dim; ++i) {
            //printf("%f \n",PSO_->solution.gbest[i]);
            logfile <<PSO_->solution.gbest[i]<<"\n";
        }
        
        //save the best particle image "ID_nstep.jpg";
        
        cv::imwrite(bestParticleRGBName,renderedImage*0.5 + image*0.5);
        cv::imwrite(bestParticleDepthName,renderedImage*0.5 + depth_kinect_rgb*0.5);
        
        //save the best images file path in the log file
        logfile << "BestRGBimage " << bestParticleRGBName << "\n";
        logfile << "BestDepthimage " << bestParticleDepthName << "\n";
        
        logFinalResult << PSO_->settings.c1 << "," << PSO_->settings.c2 << "," << PSO_->solution.error ;
        
        return true;
    }//end IF Max Step Reached
    ////////std::cout<<"0"<<"\n";
    
    std::cout<<"display n: "<<PSO_->settings.step<<"\n";
    logfile <<"display_n "<<PSO_->settings.step<<"\n";
    
    //printf("pso_w: %f\n",pso_w);
    logfile <<"pso_w "<<PSO_->pso_w<<"\n";
    
    // update pos_nb matrix (find best of neighborhood for all particles)
    switch (PSO_->settings.nhood_strategy)
    {
        case PSOClass::PSO_NHOOD_GLOBAL :
            // comm matrix not used
            PSO_->InformGlobal((double *)PSO_->pso_pos_nb);
            break;
        case PSOClass::PSO_NHOOD_RING :
            //Already Init-ed Static Ring Topology through COMM matrix
            //so use the general Inform Fcn
            PSO_->Inform((int*)PSO_->pso_comm, (double*)PSO_->pso_pos_nb, (double*)PSO_->pso_pos_b, (double*)PSO_->pso_fit_b, PSO_->pso_improved, &PSO_->settings);
            break;
    }
    
    
    
    
    //for each particle in the swarm //each particle has dim == 7 [tx,ty,tz,q0,q1,q2,q3]
    for (size_t idx_swarm_particle=0; idx_swarm_particle<PSO_->settings.size; ++idx_swarm_particle) {
        
        
        PSO_->updateEqsParticleByParticle(idx_swarm_particle);
        /* update particle fitness && Display GL */
        
        /** Look At Set ModelView Matrix **/
        bind_buffers();
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // show the current camera frame
        glDisable(GL_DEPTH_TEST);
        //based on the way cv::Mat stores data, you need to flip it before displaying it
        cv::Mat tempimage;
        cv::flip(image, tempimage, 0);
        glDrawPixels( tempimage.size().width, tempimage.size().height, GL_BGR, GL_UNSIGNED_BYTE, tempimage.ptr() );
        glEnable(GL_DEPTH_TEST);
        
        //you will have to set modelview matrix using extrinsic camera params
        
        /** Get [tx ty tz q0 q1 q2 q3] for each particle **/
        /** set offset since in Blender we saved the .obj with +Y UP !!!! **/
        //const float offset_roll = M_PI;
        
        cv::Mat Tmodelview = cv::Mat::zeros(4,4,cv::DataType<double>::type);
        
        PSO_->getT_ModelView(Tmodelview,idx_swarm_particle,
                             offset_roll,offset_pitch,offset_yaw);
        
        //std::cout<<"MV: "<<Tmodelview<<"\n";
        
        Tmodelview = Tmodelview.t();
        
        glMatrixMode(GL_MODELVIEW);
        
        //glPushMatrix();
        
        glLoadMatrixd((double*)Tmodelview.data);
        
        glEnable(GL_TEXTURE_2D);
        if (scene_list_ == 0)
        {
            scene_list_ = glGenLists(1);
            glNewList(scene_list_, GL_COMPILE);
            // now begin at the root node of the imported data and traverse
            // the scenegraph by multiplying subsequent local transforms
            // together on GL's matrix stack.
            //glDisable(GL_LIGHTING) ;
            model_->Draw();
            ////std::cout<<"draw...\n";
            glEndList();
        }
        
        glCallList(scene_list_);
        
        /**END Look At Set ModelView Matrix **/
        
        // Create images to copy the buffers to
        cv::Mat_ < cv::Vec3b > renderedImage(height, width);
        
        glFlush();
        
        // Get data from the OpenGL buffers
        bind_buffers();
        
        // Deal with the RGB image
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, renderedImage.ptr());
        
        cv::flip(renderedImage,renderedImage,0);
        
        cv::imshow("renderedImageEachP",renderedImage);
        cv::waitKey(5);
        //DEPTH
        cv::Mat_<float> depth(height, width);
        cv::Mat_ < uchar > depth_mask = cv::Mat_ < uchar > ::zeros(cv::Size(width, height));
        // Deal with the depth image
        glReadBuffer(GL_DEPTH_ATTACHMENT);
        glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, depth.ptr());
        
        cv::flip(depth,depth,0);
        
        float zNear = zmin, zFar = zmax;
        cv::Mat_<float>::iterator it = depth.begin(), end = depth.end();
        float max_allowed_z = zFar * 0.99;
        
        //unpacking
        unsigned int i_min = width, i_max = 0, j_min = height, j_max = 0;
        for (unsigned int j = 0; j < height; ++j)
            for (unsigned int i = 0; i < width; ++i, ++it)
            {
                //need to undo the depth buffer mapping
                //http://olivers.posterous.com/linear-depth-in-glsl-for-real
                *it = 2 * zFar * zNear / (zFar + zNear - (zFar - zNear) * (2 * (*it) - 1));
                if (*it > max_allowed_z)
                    *it = 0;
                else
                {
                    depth_mask(j, i) = 255;
                    // Figure the inclusive bounding box of the mask
                    if (j > j_max)
                        j_max = j;
                    else if (j < j_min)
                        j_min = j;
                    if (i > i_max)
                        i_max = i;
                    else if (i < i_min)
                        i_min = i;
                }
                
                
            }
        
        // Rescale the depth to be in millimeters
        cv::Mat depth_scale(cv::Size(width, height), CV_16UC1);
        depth.convertTo(depth_scale, CV_16UC1, 1e3);
        
        //depth now is meters; depth_scale now is mm
        
        // Crop the images, just so that they are smaller to write/read
        if (i_min > 0)
            --i_min;
        if (i_max < width - 1)
            ++i_max;
        if (j_min > 0)
            --j_min;
        if (j_max < height - 1)
            ++j_max;
        cv::Rect depth_rect = cv::Rect(i_min, j_min, i_max - i_min + 1, j_max - j_min + 1);
        
        if ((depth_rect.width <=0) || (depth_rect.height <= 0)) {
            //                depth_out = cv::Mat();
            //                image_out = cv::Mat();
            //                mask_out = cv::Mat();
            //TODO: SOmething
            //std::cout<<"ERROR!!! (depth_rect.width <=0) || (depth_rect.height <= 0)\n";
            
        } else {
            //depth_scale(depth_rect).copyTo(depth_out);
            renderedImage(depth_rect).copyTo(renderedImageCropped);
            //mask(depth_rect).copyTo(mask_out);
        }
        
        //DEPTH in grayscale for visualization purpouses
        double minVal, maxVal;
        
        cv::minMaxLoc(depth_scale,&minVal,&maxVal);
        depth_scale.convertTo(depth_gray_img,CV_8U,255.0/(maxVal - minVal), -minVal*255.0/(maxVal-minVal));
        //DEPTH in RGB for better visualization
        cv::cvtColor(depth_gray_img, depth_color_img, CV_GRAY2RGB);
        
        //cv::imshow("rendered DEPTH RGB", depth_color_img);
        //cv::waitKey(4);
        /*
         //Prova
         double q[4] = {0.7660,0.,0.6428,0.};
         double w[4] = {0.0, 0.05, 0.01, -0.1};
         double qnext[4] = {0.,0.,0.,0.};
         PSO_->QuatKinematics(q,w,qnext);
         //Prova
         PSO_->pso_pos[0][q0_] = 0.7660;
         PSO_->pso_pos[0][q1_] = 0.0;
         PSO_->pso_pos[0][q2_] = 0.6428;
         PSO_->pso_pos[0][q3_] = 0.0;
         
         PSO_->pso_vel[0][q0_] = 0.0;
         PSO_->pso_vel[0][q1_] = 0.05;
         PSO_->pso_vel[0][q2_] = 0.01;
         PSO_->pso_vel[0][q3_] = -0.1;
         PSO_->QuatKinematicsPSOStateVect(PSO_->pso_pos[0],PSO_->pso_vel[0]);
         */
        
        
        /*** Compute Obj Fcn ***/
        float* pt_depth = depth.ptr<float>(0);
        //std::cout<<depth<<"\n";
        //cv::waitKey();
        //Depth image from Kinect
        float* pt_depthKinect = depth_m.ptr<float>(0);
        //std::cout<<depth_m<<"\n";
        //cv::waitKey();
        size_t NrenderedPoints=0;
        
        float Zerror = 0.0f;
        
        /** For Each Rendered point in the particle: Background is skipped through the depth image**/
        for (unsigned int c = 0; c < wXh ; ++c)
        {
            
            if(*pt_depth > 0.00001f)
            {
                
                
                if (*pt_depthKinect==0)//cvIsNaN(*pt_depthKinect))
                {//Penalize render onto NaN depth kinect values
                    //++pt_depth;
                    //++pt_depthKinect;
                    //continue;
                    Zerror += 0.025;//25.0f;  //max_range squared?! //(5m)^ 2
                }
                else
                {
                    //MSE
                    //float e__ = *pt_depthKinect - *pt_depth;
                    //Zerror += std::pow(e__, 1.f/2.f);
                    Zerror +=( (*pt_depthKinect - *pt_depth)*(*pt_depthKinect - *pt_depth) );
                }
                
                ++NrenderedPoints;
                
                
            }//end if(pt_depth !=0 )
            
            ++pt_depth;
            ++pt_depthKinect;
        }//end for c
        
        if(NrenderedPoints > 0)
        {
            
            Zerror /= (float)NrenderedPoints;
            
            PSO_->pso_fit[idx_swarm_particle] = Zerror;
            
            std::cout<<"Zerror= "<<Zerror<<"\n";
            
        }
        else
        {
            PSO_->pso_fit[idx_swarm_particle] = DBL_MAX;
        }
        
        
        PSO_->updatePersonalAndGBest(idx_swarm_particle);
        
    }//end FOR each swarm particle
    
    //BEST PARTICLE
    //printf("Solution Vector: \n");
    logfile <<"SolutionVector\n";
    
    //printf("Fit: %f \n",PSO_->solution.error);
    logfile <<"Fit "<< PSO_->solution.error <<"\n";
    
    //printf("Pos:\n");
    logfile <<"Pos:\n";
    for (size_t i=0; i<PSO_->settings.dim; ++i) {
        printf("%f \n",PSO_->solution.gbest[i]);
        logfile <<PSO_->solution.gbest[i]<<"\n";
    }
    //printf("Vel:\n");
    logfile <<"Vel:\n";
    for (size_t i=0; i<PSO_->settings.dim; ++i) {
        //printf("%f \n",PSO_->solution.gbestVel[i]);
        logfile <<PSO_->solution.gbestVel[i]<<"\n";
    }
    //printf("----------|\n");
    logfile <<"----------|\n";
    
    //***** DRAW the BEST !!! *****//
    bind_buffers();
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // show the current camera frame
    glDisable(GL_DEPTH_TEST);
    //based on the way cv::Mat stores data, you need to flip it before displaying it
    cv::Mat tempimage;
    cv::flip(image, tempimage, 0);
    //glDrawPixels( tempimage.size().width, tempimage.size().height, GL_BGR, GL_UNSIGNED_BYTE, tempimage.ptr() );
    glEnable(GL_DEPTH_TEST);
    
    //you will have to set modelview matrix using extrinsic camera params
    /** Get [tx ty tz q0 q1 q2 q3] for each particle **/
    /** set offset since in Blender we saved the .obj with +Y UP !!!! **/
    //const float offset_roll = M_PI;
    
    cv::Mat Tmodelview =
    cv::Mat::zeros(4,4,cv::DataType<double>::type);
    
    PSO_->getT_ModelViewFromBest(Tmodelview,
                                 offset_roll,offset_pitch,offset_yaw);
    
    //std::cout<<"MVBest: "<<Tmodelview<<"\n";
    
    Tmodelview = Tmodelview.t();
    
    glMatrixMode(GL_MODELVIEW);
    
    //glPushMatrix();
    
    glLoadMatrixd((double*)Tmodelview.data);
    
    glEnable(GL_TEXTURE_2D);
    if (scene_list_ == 0)
    {
        scene_list_ = glGenLists(1);
        glNewList(scene_list_, GL_COMPILE);
        // now begin at the root node of the imported data and traverse
        // the scenegraph by multiplying subsequent local transforms
        // together on GL's matrix stack.
        //glDisable(GL_LIGHTING) ;
        model_->Draw();
        ////std::cout<<"draw...\n";
        glEndList();
    }
    
    glCallList(scene_list_);
    
    /**END Look At Set ModelView Matrix **/
    
    // Create images to copy the buffers to
    //cv::Mat_ < cv::Vec3b > renderedImage(height, width);
    
    glFlush();
    
    // Get data from the OpenGL buffers
    bind_buffers();
    
    // Deal with the RGB image
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, renderedImage.ptr());
    
    cv::flip(renderedImage,renderedImage,0);
    //cv::imshow("mixed",depth_kinect_rgb*0.5 + image*0.5 );
    //cv::imshow("depth_only",depth_kinect_rgb);
    
    /** DEAL with DEPTH **/
    //DEPTH
    cv::Mat_<float> depth(height, width);
    cv::Mat_ < uchar > depth_mask = cv::Mat_ < uchar > ::zeros(cv::Size(width, height));
    // Deal with the depth image
    glReadBuffer(GL_DEPTH_ATTACHMENT);
    glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, depth.ptr());
    
    cv::flip(depth,depth,0);
    
    float zNear = zmin, zFar = zmax;
    cv::Mat_<float>::iterator it = depth.begin(), end = depth.end();
    float max_allowed_z = zFar * 0.99;
    
    //unpacking
    unsigned int i_min = width, i_max = 0, j_min = height, j_max = 0;
    for (unsigned int j = 0; j < height; ++j)
        for (unsigned int i = 0; i < width; ++i, ++it)
        {
            //need to undo the depth buffer mapping
            //http://olivers.posterous.com/linear-depth-in-glsl-for-real
            *it = 2 * zFar * zNear / (zFar + zNear - (zFar - zNear) * (2 * (*it) - 1));
            if (*it > max_allowed_z)
                *it = 0;
            else
            {
                depth_mask(j, i) = 255;
                // Figure the inclusive bounding box of the mask
                if (j > j_max)
                    j_max = j;
                else if (j < j_min)
                    j_min = j;
                if (i > i_max)
                    i_max = i;
                else if (i < i_min)
                    i_min = i;
            }
            
            
        }
    
    // Rescale the depth to be in millimeters
    cv::Mat depth_scale(cv::Size(width, height), CV_16UC1);
    depth.convertTo(depth_scale, CV_16UC1, 1e3);
    
    //depth now is meters; depth_scale now is mm
    
    // Crop the images, just so that they are smaller to write/read
    if (i_min > 0)
        --i_min;
    if (i_max < width - 1)
        ++i_max;
    if (j_min > 0)
        --j_min;
    if (j_max < height - 1)
        ++j_max;
    cv::Rect depth_rect = cv::Rect(i_min, j_min, i_max - i_min + 1, j_max - j_min + 1);
    
    if ((depth_rect.width <=0) || (depth_rect.height <= 0)) {
        //                depth_out = cv::Mat();
        //                image_out = cv::Mat();
        //                mask_out = cv::Mat();
        //TODO: SOmething
        //std::cout<<"ERROR!!! (depth_rect.width <=0) || (depth_rect.height <= 0)\n";
        
    } else {
        //depth_scale(depth_rect).copyTo(depth_out);
        renderedImage(depth_rect).copyTo(renderedImageCropped);
        //mask(depth_rect).copyTo(mask_out);
    }
    
    //DEPTH in grayscale for visualization purpouses
    double minVal, maxVal;
    
    cv::minMaxLoc(depth_scale,&minVal,&maxVal);
    depth_scale.convertTo(depth_gray_img,CV_8U,255.0/(maxVal - minVal), -minVal*255.0/(maxVal-minVal));
    //DEPTH in RGB for better visualization
    cv::cvtColor(depth_gray_img, depth_color_img, CV_GRAY2RGB);
    
    cv::Mat rendPimage = renderedImage*0.5 + image*0.5;
    
    cv::imshow("best",rendPimage);
    cv::imshow("bestMIxed",renderedImage*0.5 + depth_kinect_rgb*0.5);
    
    //cv::imshow("BEST depth rendered RGB",depth_color_img);
    /** END DEAL with DEPTH **/
    
    cv::waitKey(5);
    // END DRAW the BEST !!!
    
    //Update the current step of PSO
    ++PSO_->settings.step;
    PSO_->setFirstStep(false);
    return false;//Next Step...
}//End Display


void drawImageGrid(cv::Mat& mat,
                   std::vector<std::vector<cv::Point3f> >& GridCornersin3D,
                   float subHeight,
                   float subWidth)
{
    int width=mat.size().width;
    int height=mat.size().height;
    
    std::cout<<"Grid dist height float [px]: "<< float(height)/3.0f << "\n";
    std::cout<<"Grid dist width float [px]: "<< float(width)/3.0f << "\n";
    
    int disth=static_cast<int>(float(height)/subHeight);
    int distw=static_cast<int>(float(width)/subWidth);
    
    std::cout<<"Grid dist height [px]: "<< disth << "\n";
    std::cout<<"Grid dist width [px]: "<< distw << "\n";
    
    for(int i=0;i<height;i+=disth)
        cv::line(mat,cv::Point(0,i),cv::Point(width,i),cv::Scalar(255,255,255));
    
    for(int i=0;i<width;i+=distw)
        cv::line(mat,cv::Point(i,0),cv::Point(i,height),cv::Scalar(255,255,255));
    
    //std::vector<std::vector< <cv::Point3f> > > GridCornersin3D;
    GridCornersin3D.clear();
    float depth_min = 0.6; //m
    float depth_max = 1.5; //m
    //Draw the corners of the grid
    for(int v=0;v<=height;v+=disth) //added <= to project also the last row corners
        for(int u=0;u<width;u+=distw)
        {
            if (v>=height) {
                v=height-1;
            }
            mat.at<cv::Vec3b>(v,u)[0] = 0;
            mat.at<cv::Vec3b>(v,u)[1] = 0;
            mat.at<cv::Vec3b>(v,u)[2] = 255;
            
            //Project in 3D the corners
            cv::Point3f Pmin,Pmax;
            std::vector<cv::Point3f> GridCornerMinMax;
            Pmin.x = -(depth_min*(cx - u))/fx;
            Pmin.y = -(depth_min*(cy - v))/fy;
            Pmin.z = depth_min;
            GridCornerMinMax.push_back(Pmin);
            Pmax.x = -(depth_max*(cx - u))/fx;
            Pmax.y = -(depth_max*(cy - v))/fy;
            Pmax.z = depth_max;
            GridCornerMinMax.push_back(Pmax);
            
            GridCornersin3D.push_back(GridCornerMinMax);
            
        }
    
    std::cout<<"GridCornersin3D.size(): " << GridCornersin3D.size() << "\n";
    for (int i=0; i<GridCornersin3D.size(); ++i) {
        std::cout<< "GridCorner: "<< i << " depth_min " <<GridCornersin3D[i][0] <<"\n";
        std::cout<< "GridCorner: "<< i << " depth_max " <<GridCornersin3D[i][1] <<"\n";
        std::cout<<"-----\n";
    }
    
}

void clean_buffers()
{
    if (texture_id_)
        glDeleteTextures(1, &texture_id_);
    texture_id_ = 0;
    
    // clean up FBO, RBO
    if (fbo_id_)
        glDeleteFramebuffers(1, &fbo_id_);
    fbo_id_ = 0;
    if (rbo_id_)
        glDeleteRenderbuffers(1, &rbo_id_);
    rbo_id_ = 0;
}

void bind_buffers()
{
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_id_);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo_id_);
}

// a useful function for displaying your coordinate system
void drawAxes(float length)
{
    glPushAttrib(GL_POLYGON_BIT | GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT) ;
    
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE) ;
    glDisable(GL_LIGHTING) ;
    
    glBegin(GL_LINES) ;
    glColor3f(1,0,0) ;
    glVertex3f(0,0,0) ;
    glVertex3f(length,0,0);
    
    glColor3f(0,1,0) ;
    glVertex3f(0,0,0) ;
    glVertex3f(0,length,0);
    
    glColor3f(0,0,1) ;
    glVertex3f(0,0,0) ;
    glVertex3f(0,0,length);
    glEnd() ;
    
    
    glPopAttrib() ;
}

void dummy_display(){};



