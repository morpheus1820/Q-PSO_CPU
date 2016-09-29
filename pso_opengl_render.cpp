//
//  pso_opengl_render.cpp
//  PSOQuatGraphSeg
//
//  Created by Giorgio on 12/11/15.
//  Copyright (c) 2015 Giorgio. All rights reserved.
//

#include "pso_opengl_render.h"


float sigmoid(float x, float a, float b)
{
    float exp_value;
    float return_value;
    
    /*** Exponential calculation ***/
    exp_value = exp((double)(-(x-a)/b));
    
    /*** Final sigmoid value ***/
    return_value = 1.0 / (1.0 + exp_value);
    
    return return_value;
}

PSOrender::PSOrender(PSOClass*& PSO,
                     GraphCanny::GraphCannySeg<GraphCanny::hsv>*& gcs,
                     const std::string& nomefilelog,
                     const std::string& nomefilelogFinalResult,
                     const std::string& IDcode,
                     const std::string& path2obj1, const std::string& path2obj2, float Kvec[9],float offset_roll,float offset_pitch,float offset_yaw, int IDX_cluster, bool debugLog, bool debugImg)
{
    
    a=1; b=20;

    cv::namedWindow("Params",0);
    cv::createTrackbar("a", "Params", &a, 40);
    cv::createTrackbar("b", "Params", &b, 40);
    
    
    
    this->PSO_ = PSO;
    this->gcs_ = gcs;
    this->sNomefilelog = nomefilelog;
    this->sNomefilelogFinalResult = nomefilelogFinalResult;
    this->sIDcode = IDcode;
    this->sPath_to_obj1 = path2obj1;
    this->sPath_to_obj2 = path2obj2;

    this->mIDX_cluster = IDX_cluster;
    
    this->fOffset_roll = offset_roll;
    this->fOffset_pitch = offset_pitch;
    this->fOffset_yaw = offset_yaw;
    
    
    gcs_->convertDepth2ColorMap(gcs_->vecSegResults[mIDX_cluster].clusterDepth,this->cvSegmentedClusterDepthRGB);
    gcs_->vecSegResults[mIDX_cluster].clusterRGB.copyTo(this->cvSegmentedClusterRGB);
    //conversion from mm to m
    gcs_->vecSegResults[mIDX_cluster].clusterDepth.convertTo(this->cvSegmentedClusterDepthM, CV_32F,1.e-3f);
    
    // added by STE: try using whole depth image!
//    cvSegmentedClusterRGB=gcs_->rgbimg;
//    this->cvSegmentedClusterDepthM=gcs_->depthimg;
//    this->cvSegmentedClusterDepthM.convertTo(this->cvSegmentedClusterDepthM, CV_32F,1.e-3f);
    // end added by STE
    
    this->height = this->cvSegmentedClusterDepthRGB.rows;
    this->width = this->cvSegmentedClusterDepthRGB.cols;
    this->wXh = this->height*this->width;
    
    this->zmin = 0.1;
    this->zmax = 1000.0;
    
//    clusterRGB.copyTo(this->cvSegmentedClusterRGB);
//    clusterDepthM.copyTo(this->cvSegmentedClusterDepthM);
//    clusterDepthRGB.copyTo(this->cvSegmentedClusterDepthRGB);
    
    //default
    this->bDebugLog = debugLog;
    this->bDebugImgs = debugImg;
    
    if(this->bDebugLog)
        initLogFiles();
    //Load intrinsic camera matrix K
    for(int i=0; i<9; ++i)
        this->K[i] = static_cast<double>(Kvec[i]);
    
    initGL();
    
    //init model obj
    model1_.reset(new Model());
    model1_->LoadModel(this->sPath_to_obj1);
    model2_.reset(new Model());
    model2_->LoadModel(this->sPath_to_obj2);
    
    renderedImage = cv::Mat_ < cv::Vec3b >::zeros(this->height, this->width);
    
    //dummy init //Use setStartClock() before run display()
    this->tStart = clock();  
    
}


void revoluteJoint(double a)
{
    glTranslated(0.0,-0.026,0.18);
    glTranslated(0.2, 0.0,0.0);
    glRotatef(-a,0,1,0);
    glTranslated(-0.2, 0.0, 0.0);
    
}

bool PSOrender::display()
{
    /** For Each Step: settings.step **/
    //TODO: PRE-INIT !!!
    cv::Mat depth_gray_img, depth_color_img, renderedImageCropped;
    
    
//    // added by STE
//    if(PSO_->settings.step>4){
//        PSO_->mTs=0.02; printf("Ts to 0.01!\n");}
    
    
    //UPDATE Weight FCN
    if (PSO_->settings.w_strategy==PSOClass::PSO_W_LIN_DEC)
        PSO_->setInertia( PSO_->calc_inertia_lin_dec(PSO_->settings.step, &PSO_->settings) );
    std::cout<<"W: "<<PSO_->getInertia() << "\n";
    //cv::waitKey();
    //Terminate?? Best Solution Found
    if (PSO_->solution.error <= PSO_->settings.goal) {

        //RECOMPUTE the best images file names
        std::string bestParticleRGBName = "/Users/giorgio/Documents/PSOQuatGraphSeg/BestImages/rgb_"+ sIDcode + "_" + boost::lexical_cast<std::string>(PSO_->settings.step-1) + ".png";
        std::string bestParticleDepthName = "/Users/giorgio/Documents/PSOQuatGraphSeg/BestImages/depth_"+ sIDcode + "_" + boost::lexical_cast<std::string>(PSO_->settings.step-1) + ".png";

        
        if(this->bDebugLog)
        {
            //Stop the timing
            ////printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
            double tElapsed = (double)(clock() - tStart)/CLOCKS_PER_SEC;
            fLogfile <<"ElapsedTimeWithPlotting "<< tElapsed << "\n";
            // SOLVED!!
            //printf("Goal achieved @ step %d (error=%.3e) \n", settings.step-1, PSO_->solution.error);
            fLogfile <<"Goal achieved @ step "<< PSO_->settings.step-1 << "error " << PSO_->solution.error <<"\n";
            //printf("Solution Vector: \n");
            fLogfile <<"SolutionVector\n";
            for (size_t i=0; i<PSO_->settings.dim; ++i) {
                //printf("%f \n",PSO_->solution.gbest[i]);
                fLogfile <<PSO_->solution.gbest[i]<<"\n";
            }
            //save the best images file path in the log file
            fLogfile << "BestRGBimage " << bestParticleRGBName << "\n";
            fLogfile << "BestDepthimage " << bestParticleDepthName << "\n";
            
            //            LogFinalResult << "c1 " << settings.c1 << "\n"
            //                           << "c2 " << settings.c2 << "\n"
            //                           << "error " << PSO_->solution.error;
            fLogFinalResult << PSO_->settings.c1 << "," << PSO_->settings.c2 << "," << PSO_->solution.error ;
        }
        
        //save the best particle image "ID_nstep.jpg";
        cv::imwrite(bestParticleRGBName,renderedImage*0.5 + cvSegmentedClusterRGB*0.5);
        cv::imwrite(bestParticleDepthName,renderedImage*0.5 + cvSegmentedClusterDepthRGB*0.5);
        
        return true;
    }//end IF error < goal
    
    //Terminate?? Max Steps Reached
    if (PSO_->settings.step > PSO_->settings.steps) {
        //Max Steps Reached...
        //RECOMPUTE the best images file names
        std::string bestParticleRGBName = "/Users/giorgio/Documents/PSOQuatGraphSeg/BestImages/rgb_"+ sIDcode + "_" + boost::lexical_cast<std::string>(PSO_->settings.step-1) + ".png";
        std::string bestParticleDepthName = "/Users/giorgio/Documents/PSOQuatGraphSeg/BestImages/depth_"+ sIDcode + "_" + boost::lexical_cast<std::string>(PSO_->settings.step-1) + ".png";
        if(this->bDebugLog)
        {
            double tElapsed = (double)(clock() - tStart)/CLOCKS_PER_SEC;
            fLogfile <<"ElapsedTimeWithPlotting "<< tElapsed << "\n";
            
            //printf("Max Steps Reached @ step %d (error=%.3e) \n", PSO_->settings.step-1, PSO_->solution.error);
            fLogfile <<"Max Steps Reached @ step "<< PSO_->settings.step-1 << "error " << PSO_->solution.error <<"\n";
            //printf("Solution Vector: \n");
            fLogfile <<"SolutionVector\n";
            for (size_t i=0; i<PSO_->settings.dim; ++i) {
                //printf("%f \n",PSO_->solution.gbest[i]);
                fLogfile <<PSO_->solution.gbest[i]<<"\n";
            }
            
            //save the best images file path in the log file
            fLogfile << "BestRGBimage " << bestParticleRGBName << "\n";
            fLogfile << "BestDepthimage " << bestParticleDepthName << "\n";
            
            fLogFinalResult << PSO_->settings.c1 << "," << PSO_->settings.c2 << "," << PSO_->solution.error ;
        }
        
        //save the best particle image "ID_nstep.jpg";
        cv::imwrite(bestParticleRGBName,renderedImage*0.5 + cvSegmentedClusterRGB*0.5);
        cv::imwrite(bestParticleDepthName,renderedImage*0.5 + cvSegmentedClusterDepthRGB*0.5);
        
        return true;
    }//end IF Max Step Reached
    ////////std::cout<<"0"<<"\n";
    
    std::cout<<"display n: "<<PSO_->settings.step<<"\n";
    if(this->bDebugLog)
    {
        fLogfile <<"display_n "<<PSO_->settings.step<<"\n";
    
        //printf("pso_w: %f\n",pso_w);
        fLogfile <<"pso_w "<<PSO_->pso_w<<"\n";
    }
    
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
    
    
    
    std::vector<cv::Mat> particleActualPose;
    particleActualPose.clear();
    //for each particle in the swarm //each particle has dim == 7 [tx,ty,tz,q0,q1,q2,q3]
    for (size_t idx_swarm_particle=0; idx_swarm_particle<PSO_->settings.size; ++idx_swarm_particle)
    {
        
        PSO_->updateEqsParticleByParticle(idx_swarm_particle);
        /* update particle fitness && Display GL */
        
        /** Look At Set ModelView Matrix **/
        bind_buffers();
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // show the current camera frame
        glDisable(GL_DEPTH_TEST);
        //based on the way cv::Mat stores data, you need to flip it before displaying it
        cv::Mat tempimage;
        cv::flip(cvSegmentedClusterRGB, tempimage, 0);
        glDrawPixels( tempimage.size().width, tempimage.size().height, GL_BGR, GL_UNSIGNED_BYTE, tempimage.ptr() );
        glEnable(GL_DEPTH_TEST);
        
        //you will have to set modelview matrix using extrinsic camera params
        
        /** Get [tx ty tz q0 q1 q2 q3] for each particle **/
        /** set offset since in Blender we saved the .obj with +Y UP !!!! **/
        //const float offset_roll = M_PI;
        
        cv::Mat Tmodelview = cv::Mat::zeros(4,4,cv::DataType<double>::type);
        
        
        PSO_->getT_ModelView(Tmodelview,idx_swarm_particle,
                             fOffset_roll,fOffset_pitch,fOffset_yaw);
        
        

        
        Tmodelview = Tmodelview.t();
        
        glMatrixMode(GL_MODELVIEW);
        
        //glPushMatrix();
        
        glLoadMatrixd((double*)Tmodelview.data);
        
        glEnable(GL_TEXTURE_2D);
        // part1
//        if (scene_list_ == 0)
//        {
//            scene_list_ = glGenLists(1);
//            glNewList(scene_list_, GL_COMPILE);
//            // now begin at the root node of the imported data and traverse
//            // the scenegraph by multiplying subsequent local transforms
//            // together on GL's matrix stack.
//            //glDisable(GL_LIGHTING) ;
            model1_->Draw();
            ////std::cout<<"draw...\n";
//            glEndList();
//        }
//        glCallList(scene_list_);
        
        double alfa_particle = PSO_->get1DOFArticulated(idx_swarm_particle);
        
        // linear joint
        //glTranslated(0.0, 0.0, alfa_particle);
        
        // revolute joint laptop
//        glTranslated(0.0, 0.0, -0.109);
//        glRotatef(-alfa_particle,1,0,0);
//        glTranslated(0.0, 0.0, 0.109);

        // revolute joint cabinet
  //      revoluteJoint(-alfa_particle);

        // part2

 //           model2_->Draw();

        
        /**END Look At Set ModelView Matrix **/
        
        // Create images to copy the buffers to
        cv::Mat_ < cv::Vec3b > renderedImage(height, width);
        
        glFlush();
        
//        cv::waitKey(0);
        // Get data from the OpenGL buffers
        bind_buffers();
        
        // Deal with the RGB image
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, renderedImage.ptr());
        
        cv::flip(renderedImage,renderedImage,0);
        
        cv::imshow("renderedImageEachP",renderedImage);
        particleActualPose.push_back(renderedImage);
        cv::waitKey(2);
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
        
//        printf("salvatagiio!!\n");
//        cv::imwrite("/Users/morpheus/Downloads/depth893.png",depth_scale);
        
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
        float* pt_depthKinect = cvSegmentedClusterDepthM.ptr<float>(0);
        //std::cout<<cvSegmentedClusterDepthM<<"\n";
        //cv::waitKey();
        size_t NrenderedPoints=0;
        
        size_t NClusterPoints = gcs_->vecSegResults[mIDX_cluster].num_points;
        
        float Zerror = 0.0f;
        float kinectZs = 0.0f;
        float renderZs = 0.0f;
        
        //RATIO
        size_t NumSegDepthWoutValid3dModelPixel=0;
        //RATIO TRANSPOSED
        size_t NumRenderedDepthWoutValid3dClusterPixel=0;
        
        
        /** For Each Rendered point in the particle: Background is skipped through the rendered depth image**/
        for (unsigned int c = 0; c < wXh ; ++c)
        {
            
            //RATIO INDEX
            if(*pt_depthKinect > 0.00001f)//1e-5 m == 0.01mm
            {
                //Here the pixel c is a Depth Valid pixel of the segmented cluster
                
                //Now check if the rendered depth at pixel c
                //contains a NON valid depth value
                if(*pt_depth <= 0.00001f)//1e-5 m == 0.01mm
                {
                    ++NumSegDepthWoutValid3dModelPixel;
                }
            }
            
            //RADIO TRANSPOSED INDEX
            if(*pt_depth > 0.00001f)//1e-5 m == 0.01mm
            {
                //Here the pixel c is a Depth Valid pixel of the rendered obj
                
                //Now check if the  segmented cluster depth at pixel c
                //contains a NON valid depth value
                if(*pt_depthKinect <= 0.00001f)//1e-5 m == 0.01mm
                {
                    ++NumRenderedDepthWoutValid3dClusterPixel;
                }
            }
            
            if(*pt_depth > 0.00001f)//rendered depth image
            {
                //The pixel c is, here, a rendered pixel of
                //the model
                
                if (*pt_depthKinect <= 0.00001f )//==0)//cvIsNaN(*pt_depthKinect))
                {//Penalize render onto NaN depth kinect values
                    //++pt_depth;
                    //++pt_depthKinect;
                    //continue;
                    Zerror += 0.;//0.5;//0.025;//25.0f;  //max_range squared?! //(5m)^ 2
                }
                else
                {
                    //MSE
                    //float e__ = *pt_depthKinect - *pt_depth;
                    //Zerror += std::pow(e__, 1.f/2.f);
                    Zerror +=( (*pt_depthKinect - *pt_depth)*(*pt_depthKinect - *pt_depth) );
                    
                    kinectZs+=*pt_depthKinect;
                    renderZs+=*pt_depth;
                    
                }
                
                ++NrenderedPoints;
                
                
            }//end if(pt_depth !=0 )
            
            ++pt_depth;
            ++pt_depthKinect;
        }//end for c
        
        float ratio_ = ((float)NumSegDepthWoutValid3dModelPixel)/((float)NClusterPoints);
        float ratio_t = ((float)NumRenderedDepthWoutValid3dClusterPixel)/((float)NrenderedPoints);
        
        kinectZs /= ((float)NClusterPoints);
        renderZs /= ((float)NrenderedPoints);
        
        if(NrenderedPoints > 0)
        {
            //MSE
            Zerror /= (float)NrenderedPoints;
            //NMSE
            float NMSE = Zerror/(kinectZs);//*renderZs);
            
//            std::cout<<"Zerror= "<<Zerror<<"\n";
//            std::cout<<"NMSE= "<<NMSE<<"\n";
//            std::cout<<"ratio_ = "<<ratio_<<"\n";
//            std::cout<<"ratio_t = "<<ratio_t<<"\n";
        
            
            
            //between [0-1]
            float sum_ratio_ = (ratio_+ratio_t)/(double)b; //20.f;   // modificato da ste, prima era /20.0f
            //std::cout<<"(ratio_+ratio_t)/2 = "<<sum_ratio_<<"\n";
            
            //aggiunto da ste
           // Zerror=sigmoid(Zerror,-0.0001,0.005);
            
            PSO_->zerrorvec[idx_swarm_particle]=Zerror;
            PSO_->ratiovec[idx_swarm_particle]=sum_ratio_;
            
            Zerror /=(double)a;
            
            Zerror += sum_ratio_;
            
            PSO_->pso_fit[idx_swarm_particle] = Zerror;
            //std::cout<<"ZerrorTot= "<<Zerror<<"\n";
            
        }
        else
        {
            PSO_->pso_fit[idx_swarm_particle] = DBL_MAX;
        }
        
        
        PSO_->updatePersonalAndGBest(idx_swarm_particle);
        
        
    }//end FOR each swarm particle
    
    //Update the Particle Trajectory
    mParticleTraj.push_back(particleActualPose);
    
    
//    for (size_t i=0; i<PSO_->settings.dim; ++i) {
//        printf("%f \n",PSO_->solution.gbest[i]);
//        fLogfile <<PSO_->solution.gbest[i]<<"\n";
//    }
    //printf("------------------------\n");

    
    //BEST PARTICLE
    if(this->bDebugLog)
    {
        printf("Solution Vector step %d: \n",PSO_->settings.step);
        fLogfile <<"SolutionVector\n";
        
        
        printf("Fit: %f (zerror %f, ratio %f)",PSO_->solution.error, PSO_->solution.zerror,
                                                                PSO_->solution.ratio);
        fLogfile <<"Fit "<< PSO_->solution.error <<"\n";
        
        printf("Pos:\n");
        fLogfile <<"Pos:\n";
        for (size_t i=0; i<PSO_->settings.dim; ++i) {
            printf("%f \n",PSO_->solution.gbest[i]);
            fLogfile <<PSO_->solution.gbest[i]<<"\n";
        }
        //printf("Vel:\n");
        fLogfile <<"Vel:\n";
        for (size_t i=0; i<PSO_->settings.dim; ++i) {
            //printf("%f \n",PSO_->solution.gbestVel[i]);
            fLogfile <<PSO_->solution.gbestVel[i]<<"\n";
        }
        //printf("----------|\n");
        //printf("Joints: %f\n", PSO_->get1DOFArticulated(PSO_->solution.gbest[alfa0_]));
        fLogfile <<"----------|\n";
    }
    //***** DRAW the BEST !!! *****//
    bind_buffers();
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // show the current camera frame
    glDisable(GL_DEPTH_TEST);
    //based on the way cv::Mat stores data, you need to flip it before displaying it
    cv::Mat tempimage;
    cv::flip(cvSegmentedClusterRGB, tempimage, 0);
    //glDrawPixels( tempimage.size().width, tempimage.size().height, GL_BGR, GL_UNSIGNED_BYTE, tempimage.ptr() );
    glEnable(GL_DEPTH_TEST);
    
    //you will have to set modelview matrix using extrinsic camera params
    /** Get [tx ty tz q0 q1 q2 q3] for each particle **/
    /** set offset since in Blender we saved the .obj with +Y UP !!!! **/
    //const float offset_roll = M_PI;
    
    cv::Mat Tmodelview =
    cv::Mat::zeros(4,4,cv::DataType<double>::type);
    
    PSO_->getT_ModelViewFromBest(Tmodelview,
                                 fOffset_roll,fOffset_pitch,fOffset_yaw);
    
    //std::cout<<"MVBest: "<<Tmodelview<<"\n";
    
    Tmodelview = Tmodelview.t();
    
    glMatrixMode(GL_MODELVIEW);
    
    //glPushMatrix();
    
    glLoadMatrixd((double*)Tmodelview.data);
    
    glEnable(GL_TEXTURE_2D);

        model1_->Draw();
    double alfa_particle = PSO_->solution.gbest[alfa0_];
    
    // linear joint
    //glTranslated(0.0, 0.0, alfa_particle);

    // revolute joint laptop
//    glTranslated(0.0, 0.0, -0.109);
//    glRotatef(-alfa_particle,1,0,0);
//    glTranslated(0.0, 0.0, 0.109);
  
    // revolute joint cabinet
    
//    revoluteJoint(-alfa_particle);
//    
//    
//    model2_->Draw();
    
    
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
    //cv::imshow("mixed",cvSegmentedClusterDepthRGB*0.5 + cvSegmentedClusterRGB*0.5 );
    //cv::imshow("depth_only",cvSegmentedClusterDepthRGB);
    
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
    //cv::imshow("depth_scale",depth_gray_img);

//    cv::Mat_<float> PC = cv::Mat_<float>::zeros(0,3);
//    gcs_->getPointCloud(depth_scale, PC);
//    std::cout<<"c: "<<PC.cols<<" r: "<< PC.rows<<"\n";
//    gcs_->save1chMat2MatlabCSV(PC,"/Users/giorgio/Documents/MATLAB/mypc.csv");
//    cv::waitKey();
    
    //DEPTH in RGB for better visualization
    cv::cvtColor(depth_gray_img, depth_color_img, CV_GRAY2RGB);
    
    cv::Mat rendPimage = renderedImage*0.5 + cvSegmentedClusterRGB*0.5;
    
    cv::imshow("best",rendPimage);
    cv::imshow("bestMIxed",renderedImage*0.5 + cvSegmentedClusterDepthRGB*0.5);
    
    //cv::imshow("BEST depth rendered RGB",depth_color_img);
    /** END DEAL with DEPTH **/
    
    cv::waitKey(5);
    //exit(0);
    // END DRAW the BEST !!!
    
    //Update the current step of PSO
    ++PSO_->settings.step;
    PSO_->setFirstStep(false);
    /*
    if((PSO_->settings.step-1)==((int)(PSO_->settings.steps/2)))
    {                     // dt,   dwX, dwY, dwZ
        PSO_->reInitFromBest(0.02, 0.03, 0.3, 0.03);
        //cv::waitKey();
    }
    */
    return false;//Next Step...
}//End Display
