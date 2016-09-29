//
//  pso_opengl_render.h
//  PSOQuatGraphSeg
//
//  Created by Giorgio on 12/11/15.
//  Copyright (c) 2015 Giorgio. All rights reserved.
//

#ifndef __PSOQuatGraphSeg__pso_opengl_render__
#define __PSOQuatGraphSeg__pso_opengl_render__

#include <stdio.h>
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

#include "GraphCannySeg.h"

#include "pso_class_quaternions.h"

#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>

#include <fstream>
#include <time.h>

#include <algorithm>    // std::min_element, std::max_element

using namespace glm;

class PSOrender {
    
    int width;
    int height;
    int wXh;
    double zmin;
    double zmax;
    //CHALLENGE 1 DATASET
    //[ fx 0 cx; 0 fy cy; 0 0 1 ]
    double K[9];
    
    
    cv::Mat cvSegmentedClusterRGB;//image;
    cv::Mat depth_img, cvSegmentedClusterDepthM;//depth_m
    cv::Mat cvSegmentedClusterDepthRGB;//depth_kinect_rgb;
    
    // Create images to copy the buffers to
    cv::Mat_ < cv::Vec3b > renderedImage;
    
    /** The frame buffer object used for offline rendering */
    GLuint fbo_id_;
    /** The render buffer object used for offline depth rendering */
    GLuint rbo_id_;
    /** The render buffer object used for offline image rendering */
    GLuint texture_id_;
    
    /** Model Obj+mtl to load **/
    boost::shared_ptr<Model> model1_,model2_;
    
    GLuint scene_list_;

    std::ofstream fLogfile , fLogFinalResult;
    clock_t tStart;
    std::string sIDcode;
    
    //Quaternion PSO Class
    PSOClass* PSO_;
    //Graph Segmentation Class
    GraphCanny::GraphCannySeg<GraphCanny::hsv>* gcs_;
    
    std::string sNomefilelog;
    std::string sNomefilelogFinalResult;
    std::string sPath_to_obj1, sPath_to_obj2;
    
    bool bDebugLog;
    bool bDebugImgs;
    
    float fOffset_roll; //= -M_PI/3.;//0.f;//M_PI;
    float fOffset_pitch; //= 0.f;//M_PI;
    float fOffset_yaw; //= 0.f;//M_PI;
    
    int mIDX_cluster;
    
    int a,b;

    
public:
    std::vector<std::vector<cv::Mat> > mParticleTraj;
    
    
public:
    PSOrender(PSOClass*& PSO,
              GraphCanny::GraphCannySeg<GraphCanny::hsv>*& gcs,
              const std::string& nomefilelog,
              const std::string& nomefilelogFinalResult,
              const std::string& IDcode,
              const std::string& path2obj1, const std::string& path2obj2,float Kvec[9],float offset_roll,float offset_pitch,float offset_yaw, int IDX_cluster,bool debugLog=true, bool debugImg = true);

    inline void setDebugLog(bool b)
    {
        this->bDebugLog = b;
    }
    inline void setDebugImgs(bool b)
    {
        this->bDebugImgs = b;
    }
    inline void initLogFiles()
    {
        fLogfile.open(sNomefilelog);
        fLogFinalResult.open(sNomefilelogFinalResult);
        
        if(!fLogfile.is_open())
        {
            std::cout<<"fLogfile not opened\n";
            exit(EXIT_FAILURE);
        }
        if(!fLogFinalResult.is_open())
        {
            std::cout<<"fLogFinalResult not opened\n";
            exit(EXIT_FAILURE);
        }
        
        //Save all the PSO settings
        fLogfile << "c1 " << PSO_->settings.c1 << "\n"
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
        << "/***************************************/\n\n";
    }
    inline void clean_buffers()
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
    
    inline void bind_buffers()
    {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_id_);
        glBindRenderbuffer(GL_RENDERBUFFER, rbo_id_);
    }
    
    static void dummy_display() {};
    
    inline void setStartClock()
    {
        this->tStart = clock();
    }
    
    inline void closeLogFiles()
    {
        if(this->bDebugLog)
        {
            this->fLogfile.close();
            this->fLogFinalResult.close();
        }
    }
    
    inline void initGL()
    {
        clean_buffers();
        
        // initialize GLUT
        int argc_=1; char* argv_[] = {"PSO"};
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
        GLdouble perspMatrix[16]={2*K[0]/width,0,0,0,0,-2*K[4]/height,0,0,2*(K[2]/width)-1,1-2*(K[5]/height),(zmax+zmin)/(zmax-zmin),1,0,0,2*zmax*zmin/(zmin-zmax),0};
        glMultMatrixd(perspMatrix);
        //set viewport
        glViewport(0, 0, width, height);
        /** END INIT THE PROJECTION MATRIX **/
        
        /** Set scene_list_ to 0 **/
        scene_list_=0;
        
        // set up GUI callback functions
        glutDisplayFunc( dummy_display );

    }
    bool display();
};


#endif /* defined(__PSOQuatGraphSeg__pso_opengl_render__) */
