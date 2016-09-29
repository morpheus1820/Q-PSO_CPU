//
//  pso_class_quaternions.cpp
//
//
//  Created by Giorgio on 29/07/15.
//
//


#include <time.h> // for time()
#include <math.h> // for cos(), pow(), sqrt() etc.
#include <float.h> // for DBL_MAX
#include <string.h> // for mem*
#include <iostream>
#include <vector>
#include <stdlib.h>     /* exit, EXIT_FAILURE */

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"

#include <gsl/gsl_rng.h>

#include "pso_class_quaternions.h"

//Costructor
PSOClass::PSOClass(/*int Ndim, int NSize,*/ std::vector<double> Dimbounds,
                   double c1_, double c2_,double Ts,
                   unsigned int dynamicInertia,
                   unsigned int nhoodtype, int maxSteps,
                   double ErrorGoal,
                   int nhood_size,
                   double w_max_, double w_min_)
{
    
    // Dimbounds.size() == 2*Ndim = 14  --> Dimbounds[0] = tx_low; Dimbounds[1] = tx_high
    //                                      Dimbounds[2] = ty_low; Dimbounds[3] = ty_high
    //                                      Dimbounds[4] = tz_low; Dimbounds[5] = tz_high
    //                                      Dimbounds[6] = q0_low; Dimbounds[7] = q0_high
    //                                      Dimbounds[8] = q1_low; Dimbounds[9] = q1_high
    //                                      Dimbounds[10] = q2_low; Dimbounds[11] = q2_high
    //                                      Dimbounds[12] = q3_low; Dimbounds[13] = q3_high
    
    
    if (Dimbounds.size() != (2*Ndim) )
    {
        std::cout<<"Error Dimbounds.size() != 2*Ndim"<<"\n";
        exit(EXIT_FAILURE);
    }
    
    mTs = Ts;
    pso_improved = 0;
    pso_first_step = true;
    
    /** INIT PSO SETTINGS **/
    settings.dim = Ndim;//Problem dimension tx,ty,tz,q0,q1,q2,q3
    //x_lo && x_hi for each dimension
    settings.x_lo = (double*)malloc(settings.dim * sizeof(double));
    settings.x_hi = (double*)malloc(settings.dim * sizeof(double));
    
    //INIT
    //    pso_pos.resize(NSize);
    //    for (int i=0; i<pso_pos.size(); ++i) {
    //        pso_pos[i].resize(Ndim,0.0);
    //    }
    //
    //    pso_vel.resize(NSize);
    //    for (int i=0; i<pso_vel.size(); ++i) {
    //        pso_vel[i].resize(Ndim,0.0);
    //    }
    //
    //    pso_pos_b.resize(NSize);
    //    for (int i=0; i<pso_pos_b.size(); ++i) {
    //        pso_pos_b[i].resize(Ndim,0.0);
    //    }
    //
    //    pso_fit.resize(Ndim,0.0);
    //    pso_fit_b.resize(Ndim,0.0);
    //
    //    pso_pos_nb.resize(NSize);
    for (int i=0; i<NSize; ++i)
        for (int j=0; j<Ndim; ++j){
            pso_pos_nb[i][j]=0.0;
        }
    //
    //    pso_comm.resize(NSize);
    // for (int i=0; i<pso_comm.size(); ++i) {
    //     pso_comm[i].resize(NSize,0);
    // }
    
    //tx
    settings.x_lo[0] = Dimbounds[0];
    settings.x_hi[0] = Dimbounds[1];
    //ty
    settings.x_lo[1] = Dimbounds[2];
    settings.x_hi[1] = Dimbounds[3];
    //tz
    settings.x_lo[2] = Dimbounds[4];
    settings.x_hi[2] = Dimbounds[5];
    //q0 q.w
    settings.x_lo[q0_] = Dimbounds[6];
    settings.x_hi[q0_] = Dimbounds[7];
    //q1
    settings.x_lo[q1_] = Dimbounds[8];
    settings.x_hi[q1_] = Dimbounds[9];
    //q2
    settings.x_lo[q2_] = Dimbounds[10];
    settings.x_hi[q2_] = Dimbounds[11];
    //q3
    settings.x_lo[q3_] = Dimbounds[12];
    settings.x_hi[q3_] = Dimbounds[13];
    //articulated 1DOF
    settings.x_lo[alfa0_] = Dimbounds[14];
    settings.x_hi[alfa0_] = Dimbounds[15];
    
    settings.goal = ErrorGoal;
    
    settings.size = NSize;//pso_calc_swarm_size(settings.dim);
    //printf("Particle Swarm Size: %d: \n",settings.size);
    
    settings.print_every = 1000;
    settings.steps = maxSteps;
    settings.step = 0;
    settings.c1 = c1_;
    settings.c2 = c2_;
    settings.w_max = w_max_;
    settings.w_min = w_min_;
    
    settings.clamp_pos = 1;
    settings.nhood_strategy = nhoodtype;//PSO_NHOOD_GLOBAL;//PSO_NHOOD_RING;
    settings.nhood_size = nhood_size;
    settings.w_strategy = dynamicInertia;//PSO_W_CONST;//PSO_W_LIN_DEC;
    
    /** END INIT PSO SETTINGS **/
    
    /** allocate memory for the best position buffer **/
    solution.gbest = (double*)malloc(settings.dim * sizeof(double));
    /** allocate memory for the best velocity buffer **/
    solution.gbestVel = (double*)malloc(settings.dim * sizeof(double));
    
    
    //    std::random_device randDev;
    //    boost::shared_ptr<std::mt19937> genPtr;//(randDev());
    //    boost::shared_ptr<std::uniform_real_distribution<double> > UniDisPtr;//(1, 2);
    genPtr.reset(new std::mt19937(randDev()));
    UniDisPtr.reset(new std::uniform_real_distribution<double>(0,1));//[0,1)
    
    // SELECT APPROPRIATE NHOOD UPDATE FUNCTION
    switch (settings.nhood_strategy)
    {
        case PSO_NHOOD_GLOBAL :
            // comm matrix not used
            //inform_fun = inform_global;
            break;
        case PSO_NHOOD_RING :
            init_comm_ring((int *)pso_comm, &settings);
            //OK!
            for (int i=0; i<NSize; ++i) {
                printVector((int *)pso_comm[i], NSize);
            }
            //cv::waitKey();
            //inform_fun = inform_ring;
            break;
            //     case PSO_NHOOD_RANDOM :
            //     init_comm_random((int *)pso_comm, &settings);
            //     inform_fun = inform_random;
            //     break;
    }
    // INITIALIZE SOLUTION
    solution.error = DBL_MAX;
    
    double pso_a, pso_b; // for matrix initialization
    double x0, x1 ,x2, th1, th2, r1, r2; //from uniform random quaternion init
    double vdot=3.;
    //double quat[4]; int id_quat=0;
    // SWARM INITIALIZATION
    // for each particle
    for (int i_=0; i_<settings.size; ++i_) {
        // for each dimension
        for (int d_=0; d_<settings.dim; ++d_) {
            // generate two numbers within the specified range
            if (d_<3) {//tx ty tz
                pso_a = settings.x_lo[d_] + (settings.x_hi[d_] - settings.x_lo[d_]) * ((*UniDisPtr)(*genPtr));
                pso_b = settings.x_lo[d_] + (settings.x_hi[d_] - settings.x_lo[d_]) * ((*UniDisPtr)(*genPtr));
                // initialize pose
                pso_pos[i_][d_] = pso_a;
                // best pose is the same
                pso_pos_b[i_][d_] = pso_a;
                // initialize velocity
                pso_vel[i_][d_] = -vdot + (2.*vdot) * ((*UniDisPtr)(*genPtr));//(pso_a-pso_b) / 2.;
                
                
            }
            else if(d_>100)//Never Executed //3 to 6 Random Generator. I USE the perturbation Quat kinematic below
            {//Quat qw qx qy qz
                /* quat[id_quat] = settings.x_lo[d_] + (settings.x_hi[d_] - settings.x_lo[d_]) * \
                 gsl_rng_uniform((gsl_rng*)settings.rng);
                 ++id_quat;
                 if (d_==6)
                 {
                 //normalize the random quaternion
                 double qnorm = sqrt(quat[0]*quat[0] + quat[1]*quat[1] +
                 quat[2]*quat[2] + quat[3]*quat[3]);
                 
                 for (int dd_=3; dd_<7; ++dd_)
                 pso_pos[i_][dd_] = quat[dd_-3]/qnorm;
                 }
                 //angular velocity w = [0;wx;wy;wz]
                 if (d_==3) {
                 pso_vel[i_][d_] = 0.0;
                 }
                 else
                 {   pso_a = settings.x_lo[d_] + (settings.x_hi[d_] - settings.x_lo[d_]) * \
                 gsl_rng_uniform((gsl_rng*)settings.rng);
                 pso_b = settings.x_lo[d_] + (settings.x_hi[d_] - settings.x_lo[d_]) * \
                 gsl_rng_uniform((gsl_rng*)settings.rng);
                 pso_vel[i_][d_] = (pso_a-pso_b) / 2.;
                 }
                 
                 */
                //See K. Shoemake Uniform random rotations.
                x0 = ((*UniDisPtr)(*genPtr));//[0,1) TODO: Giusto?? o 1 deve essere compreso??
                x1 = ((*UniDisPtr)(*genPtr));
                x2 = ((*UniDisPtr)(*genPtr));
                
                th1 = 2*M_PI*x1;
                th2 = 2*M_PI*x2;
                
                r1 = sqrt(1.0-x0);
                r2 = sqrt(x0);
                switch (d_) {
                    case q0_://q0
                    {
                        pso_pos[i_][d_] = cos(th2)*r2;
                        // best pose is the same
                        pso_pos_b[i_][d_] = cos(th2)*r2;
                        //angular velocity w = [0;wx;wy;wz]
                        pso_vel[i_][d_] = 0.0;
                    }
                        break;
                    case q1_://q1
                    {
                        pso_pos[i_][d_] = sin(th1)*r1;
                        // best pose is the same
                        pso_pos_b[i_][d_] = sin(th1)*r1;
                        
                        //angular velocity w = [0;wx;wy;wz]
                        //Between [-1;1]
                        pso_a = settings.x_lo[d_] + (settings.x_hi[d_] - settings.x_lo[d_]) * \
                        ((*UniDisPtr)(*genPtr));
                        pso_b = settings.x_lo[d_] + (settings.x_hi[d_] - settings.x_lo[d_]) * \
                        ((*UniDisPtr)(*genPtr));
                        pso_vel[i_][d_] = (pso_a-pso_b) / 2.;
                    }
                        break;
                    case q2_://q2
                    {
                        pso_pos[i_][d_] = cos(th1)*r1;
                        // best pose is the same
                        pso_pos_b[i_][d_] = cos(th1)*r1;
                        
                        //angular velocity w = [0;wx;wy;wz]
                        //Between [-1;1]
                        pso_a = settings.x_lo[d_] + (settings.x_hi[d_] - settings.x_lo[d_]) * \
                        ((*UniDisPtr)(*genPtr));
                        pso_b = settings.x_lo[d_] + (settings.x_hi[d_] - settings.x_lo[d_]) * \
                        ((*UniDisPtr)(*genPtr));
                        pso_vel[i_][d_] = (pso_a-pso_b) / 2.;
                    }
                        break;
                    case q3_://q3
                    {
                        pso_pos[i_][d_] = sin(th2)*r2;
                        // best pose is the same
                        pso_pos_b[i_][d_] = sin(th2)*r2;
                        
                        //angular velocity w = [0;wx;wy;wz]
                        //Between [-1;1]
                        pso_a = settings.x_lo[d_] + (settings.x_hi[d_] - settings.x_lo[d_]) * \
                        ((*UniDisPtr)(*genPtr));
                        pso_b = settings.x_lo[d_] + (settings.x_hi[d_] - settings.x_lo[d_]) * \
                        ((*UniDisPtr)(*genPtr));
                        pso_vel[i_][d_] = (pso_a-pso_b) / 2.;
                    }
                        break;
                        
                    default:
                        break;
                }//end switch
                
            }//end else 3 to 6
            
        }//end for d_ : dim
        
        
        //INIT ARTICULATED 1DOF
        pso_a = settings.x_lo[alfa0_] + (settings.x_hi[alfa0_] - settings.x_lo[alfa0_]) * ((*UniDisPtr)(*genPtr));
        pso_b = settings.x_lo[alfa0_] + (settings.x_hi[alfa0_] - settings.x_lo[alfa0_]) * ((*UniDisPtr)(*genPtr));
        // initialize pose
        pso_pos[i_][alfa0_] = pso_a;
        // best pose is the same
        pso_pos_b[i_][alfa0_] = pso_a;
        // initialize velocity
        pso_vel[i_][alfa0_] = -vart_ + (2.*vart_) * ((*UniDisPtr)(*genPtr));//(pso_a-pso_b) / 2.;
        
        
        
        
        printf("Linear Position & Velocity\n");
        printLinearPositionPSOstateVect(pso_pos[i_],i_);
        printLinearVelPSOstateVect(pso_vel[i_],i_);
        
        //Perturbate the INIT Quaterion
        double qinit[7] = {0.,0.,0.,0.,1.,0.,0.};//{0.,0.,0.,0.7071,0.,0.7071,0.};
        
        //double qinit[7] = {0.,0.,0.,0.983022,0.012499,0.141231,-0.116470};
        ////    0.026750
        ////    0.202456
        ////    2.021780
        ////    0.983022
        ////    0.012499
        ////    0.141231
        ////    -0.116470
        
        
        double wpert[7] = {0.,0.,0.,0.,0.,0.,0.};
        double aminX = -10.0;//-(2.*(double)i_/10.);
        double amaxX = 10.0;//+(2.*(double)i_/10.);
        double aminY = -10.0;
        double amaxY = 10.0;
        double aminZ = -10.0;
        double amaxZ = 10.0;
        //std::cout<<"amax"<<amax<<"\n";
        /* Angular Vel. Used to Perturb the Quaternion!*/
        wpert[q1_] = aminX + (amaxX-aminX)*((*UniDisPtr)(*genPtr));
        wpert[q2_] = aminY + (amaxY-aminY)*((*UniDisPtr)(*genPtr));
        wpert[q3_] = aminZ + (amaxZ-aminZ)*((*UniDisPtr)(*genPtr));
        //printVector(pso_vel[i_],7);
        
        QuatKinematicsPSOStateVectWithInit(qinit,pso_pos[i_],wpert);
        //the same for pso_pos_b
        std::copy(std::begin(pso_pos[i_]), std::end(pso_pos[i_]), std::begin(pso_pos_b[i_]));
        
        printf("Perturbed Init Quat by AngularVel\n");
        printQuatPSOstateVect(pso_pos[i_],i_);
        printAngVelPSOstateVect(wpert,i_);
        printf("#################################\n\n");
        /* Angular Vel. used as initial vel of the particle!*/
        wpert[q1_] = aminX + (amaxX-aminX)*((*UniDisPtr)(*genPtr));
        wpert[q2_] = aminY + (amaxY-aminY)*((*UniDisPtr)(*genPtr));
        wpert[q3_] = aminZ + (amaxZ-aminZ)*((*UniDisPtr)(*genPtr));
        
        std::copy(wpert+q0_, std::end(wpert), pso_vel[i_]+q0_);
        
        /*
         //Normalize Quat actual
         NormalizeQuaternionPSOstateVect(pso_pos[i_]);
         //Normalize Quat Best
         NormalizeQuaternionPSOstateVect(pso_pos_b[i_]);
         //Cast the quaternions in the top half hypersphere
         //iff q0_ < 0
         Cast2TopHalfHyperspherePSOstateVect(pso_pos[i_]);
         Cast2TopHalfHyperspherePSOstateVect(pso_pos_b[i_]);
         */
        printf("Initial AngularVel of the Perturbed Quat\n");
        printQuatPSOstateVect(pso_pos[i_],i_);
        printAngVelPSOstateVect(pso_vel[i_],i_);
        printf("#################################\n\n");
        
    }//end for i_: size
    //cv::waitKey();
    // initialize omega using standard value
    pso_w = PSO_INERTIA;
    
    //cv::waitKey();
}

PSOClass::~PSOClass()
{
    free(settings.x_lo);
    free(settings.x_hi);
    free(solution.gbest);
    free(solution.gbestVel);
    //    if (pso_free_rng)
    //    {
    //        free(settings.rng);
    //    }
}

void PSOClass::setFirstStep(bool fs)
{
    this->pso_first_step = fs;
}

void PSOClass::InformGlobal(double *pos_nb)
{
    //std::cout<<"InforI"<<"\n";
    if (!pso_first_step) {
        //Inform Global Fcn:
        //std::cout<<"InformMemMove: "<<pso_pos_nb[2][0]<<"\n";        // all particles have the same attractor (gbest)
        // copy the contents of gbest to pos_nb
        for (int i=0; i<settings.size; ++i)
        {
            //i*settings->dim itera fra le righe e sovrascrive le colonne (lunghezza settings->dim)
            //con il vettore solution.gbest di lunghezza settings->dim
            
            memmove((void *)&pos_nb[i*settings.dim], (void *)solution.gbest,
                    sizeof(double) * settings.dim);
            
        }
        // the value of improved was just used; reset it
        //std::cout<<"InformMemMove: "<<pso_pos_nb[2][0]<<"\n";
        pso_improved = 0;
    }
    //std::cout<<"InforF"<<"\n";
}


// =============
// ring topology: INIT the COMM Matrix for Static Ring Topology
// =============
// topology initialization :: this is a static (i.e. fixed) topology
void PSOClass::init_comm_ring(int *comm, pso_settings_t * settings) {
    int i;
    // reset array
    memset((void *)comm, 0, sizeof(int)*settings->size*settings->size);
    
    // choose informers
    for (i=0; i<settings->size; i++) {
        // set diagonal to 1
        comm[i*settings->size+i] = 1;
        if (i==0) {
            // look right
            comm[i*settings->size+i+1] = 1;
            // look left
            comm[(i+1)*settings->size-1] = 1;
        } else if (i==settings->size-1) {
            // look right
            comm[i*settings->size] = 1;
            // look left
            comm[i*settings->size+i-1] = 1;
        } else {
            // look right
            comm[i*settings->size+i+1] = 1;
            // look left
            comm[i*settings->size+i-1] = 1;
        }
        
    }
    
}
// ===============================================================
// general inform function :: according to the connectivity
// matrix COMM, it copies the best position (from pos_b) of the
// informers of each particle to the pos_nb matrix
void PSOClass::Inform(int *comm, double *pos_nb, double *pos_b, double *fit_b,int improved, pso_settings_t * settings)
{
    if (!pso_first_step) {
        int i, j;
        int b_n; // best neighbor in terms of fitness
        
        // for each particle
        for (j=0; j<settings->size; j++) {
            b_n = j; // self is best
            // who is the best informer??
            for (i=0; i<settings->size; i++)
                // the i^th particle informs the j^th particle
                if (comm[i*settings->size + j] && fit_b[i] < fit_b[b_n])
                    // found a better informer for j^th particle
                    b_n = i;
            // copy pos_b of b_n^th particle to pos_nb[j]
            memmove((void *)&pos_nb[j*settings->dim],
                    (void *)&pos_b[b_n*settings->dim],
                    sizeof(double) * settings->dim);
        }//end for
        pso_improved = 0;
    }//end if (!pso_first_step)
}


void PSOClass::updateEqsParticleByParticle(size_t idx_swarm_particle)
{
    
    //std::cout<<"updateEqsParticleByParticleInizio\n";
    //std::cout<<"--------\n\n";
    
    if (!pso_first_step) {
        /**Only tx ty tz !!!**/
        for (size_t d_=0; d_<q0_;++d_){//settings.dim; ++d_) {
            // calculate stochastic coefficients
            pso_rho1 = settings.c1 * ((*UniDisPtr)(*genPtr));
            pso_rho2 = settings.c2 * ((*UniDisPtr)(*genPtr));
            pso_vel[idx_swarm_particle][d_] = pso_w * pso_vel[idx_swarm_particle][d_] +	\
            pso_rho1 * (pso_pos_b[idx_swarm_particle][d_] - pso_pos[idx_swarm_particle][d_]) +	\
            pso_rho2 * (pso_pos_nb[idx_swarm_particle][d_] - pso_pos[idx_swarm_particle][d_]);
            // update position
            
            pso_pos[idx_swarm_particle][d_] += pso_vel[idx_swarm_particle][d_];
            
            // clamp position within bounds?
            if (settings.clamp_pos) {
                if (pso_pos[idx_swarm_particle][d_] < settings.x_lo[d_]) {
                    pso_pos[idx_swarm_particle][d_] = settings.x_lo[d_];
                    pso_vel[idx_swarm_particle][d_] = 0;
                    //std::cout<<"IFsettings.clamp_pos\n";
                } else if (pso_pos[idx_swarm_particle][d_] > settings.x_hi[d_]) {
                    pso_pos[idx_swarm_particle][d_] = settings.x_hi[d_];
                    pso_vel[idx_swarm_particle][d_] = 0;
                    //std::cout<<"ELSEIFsettings.clamp_pos\n";
                }
            } else {
                // enforce periodic boundary conditions
                if (pso_pos[idx_swarm_particle][d_] < settings.x_lo[d_]) {
                    
                    pso_pos[idx_swarm_particle][d_] = settings.x_hi[d_] - fmod(settings.x_lo[d_] - pso_pos[idx_swarm_particle][d_],
                                                                               settings.x_hi[d_] - settings.x_lo[d_]);
                    pso_vel[idx_swarm_particle][d_] = 0;
                    
                    
                } else if (pso_pos[idx_swarm_particle][d_] > settings.x_hi[d_]) {
                    
                    pso_pos[idx_swarm_particle][d_] = settings.x_lo[d_] + fmod(pso_pos[idx_swarm_particle][d_] - settings.x_hi[d_],
                                                                               settings.x_hi[d_] - settings.x_lo[d_]);
                    pso_vel[idx_swarm_particle][d_] = 0;
                    
                    
                }
            }
            
        }//end for <q0_ Translation component only
        
        
        //1DOF Prismatic Joint Update...
        {//////
            
            // calculate stochastic coefficients
            pso_rho1 = settings.c1 * ((*UniDisPtr)(*genPtr));
            pso_rho2 = settings.c2 * ((*UniDisPtr)(*genPtr));
            pso_vel[idx_swarm_particle][alfa0_] = pso_w * pso_vel[idx_swarm_particle][alfa0_] +	\
            pso_rho1 * (pso_pos_b[idx_swarm_particle][alfa0_] - pso_pos[idx_swarm_particle][alfa0_]) +	\
            pso_rho2 * (pso_pos_nb[idx_swarm_particle][alfa0_] - pso_pos[idx_swarm_particle][alfa0_]);
            // update position
            
            pso_pos[idx_swarm_particle][alfa0_] += pso_vel[idx_swarm_particle][alfa0_];
            
            // clamp position within bounds?
            if (settings.clamp_pos) {
                if (pso_pos[idx_swarm_particle][alfa0_] < settings.x_lo[alfa0_]) {
                    pso_pos[idx_swarm_particle][alfa0_] = settings.x_lo[alfa0_];
                    pso_vel[idx_swarm_particle][alfa0_] = 0;
                    //std::cout<<"IFsettings.clamp_pos\n";
                } else if (pso_pos[idx_swarm_particle][alfa0_] > settings.x_hi[alfa0_]) {
                    pso_pos[idx_swarm_particle][alfa0_] = settings.x_hi[alfa0_];
                    pso_vel[idx_swarm_particle][alfa0_] = 0;
                    //std::cout<<"ELSEIFsettings.clamp_pos\n";
                }
            } else {
                // enforce periodic boundary conditions
                if (pso_pos[idx_swarm_particle][alfa0_] < settings.x_lo[alfa0_]) {
                    
                    pso_pos[idx_swarm_particle][alfa0_] = settings.x_hi[alfa0_] - fmod(settings.x_lo[alfa0_] - pso_pos[idx_swarm_particle][alfa0_],
                                                                                       settings.x_hi[alfa0_] - settings.x_lo[alfa0_]);
                    pso_vel[idx_swarm_particle][alfa0_] = 0;
                    
                    
                } else if (pso_pos[idx_swarm_particle][alfa0_] > settings.x_hi[alfa0_]) {
                    
                    pso_pos[idx_swarm_particle][alfa0_] = settings.x_lo[alfa0_] + fmod(pso_pos[idx_swarm_particle][alfa0_] - settings.x_hi[alfa0_],
                                                                                       settings.x_hi[alfa0_] - settings.x_lo[alfa0_]);
                    pso_vel[idx_swarm_particle][alfa0_] = 0;
                    
                    
                }
            }
        }/////end 1DOF Prismatic Joint
        
        
        
        /** Only Quaternion !!! **/
        
        double q_tilde_cognitive[4];
        double q_tilde_social[4];
        
        
        /** COGNITIVE PART **/
        
        //Quaternion Product q_ind_best*q_actual^(-1)
        QuatProd_pq_1PSOstateVect(pso_pos_b[idx_swarm_particle],pso_pos[idx_swarm_particle],q_tilde_cognitive);
        
        Cast2TopHalfHypersphere(q_tilde_cognitive);
        
        /*
         std::cout<<"cognitive...\n";
         printQuatPSOstateVect(pso_pos_b[idx_swarm_particle],(int)idx_swarm_particle,"pbest");
         printQuatPSOstateVect(pso_pos[idx_swarm_particle],(int)idx_swarm_particle,"actual");
         printQuat(q_tilde_cognitive,(int)idx_swarm_particle,"error");
         */
        double norm_v_tilde_cognitive = sqrt(
                                             q_tilde_cognitive[1]*q_tilde_cognitive[1] +
                                             q_tilde_cognitive[2]*q_tilde_cognitive[2] +
                                             q_tilde_cognitive[3]*q_tilde_cognitive[3]
                                             );
        //        std::cout<<"norm_v_tilde_cognitive: "
        //                    <<norm_v_tilde_cognitive<<"\n";
        
        double w_tilde_cognitive[3]={0.0,0.0,0.0};
        //double scaleW = 1;
        if (norm_v_tilde_cognitive>MIN_NORM)
        {
            w_tilde_cognitive[0] =
            /*scaleW*/2.0*(q_tilde_cognitive[1] / norm_v_tilde_cognitive)*
            std::acos(q_tilde_cognitive[0]);
            
            w_tilde_cognitive[1] =
            /*scaleW*/2.0*(q_tilde_cognitive[2] / norm_v_tilde_cognitive)*
            std::acos(q_tilde_cognitive[0]);
            
            w_tilde_cognitive[2] =
            /*scaleW*/2.0*(q_tilde_cognitive[3] / norm_v_tilde_cognitive)*
            std::acos(q_tilde_cognitive[0]);
        }
        /*
         std::cout<<"w_tilde_cognitive\n";
         printVector(w_tilde_cognitive,3);
         */
        /**END COGNITIVE PART **/
        //cv::waitKey();
        
        /** SOCIAL PART **/
        
        //Quaternion Product q_global_best*q_actual^(-1)
        QuatProd_pq_1PSOstateVect(pso_pos_nb[idx_swarm_particle],pso_pos[idx_swarm_particle],q_tilde_social);
        
        Cast2TopHalfHypersphere(q_tilde_social);
        
        /*
         std::cout<<"Social...\n";
         
         printQuatPSOstateVect(pso_pos_nb[idx_swarm_particle],(int)idx_swarm_particle,
         "gbest");
         printQuatPSOstateVect(pso_pos[idx_swarm_particle],(int)idx_swarm_particle,"actual");
         printQuat(q_tilde_social,(int)idx_swarm_particle,"error");
         */
        double norm_v_tilde_social = sqrt(
                                          q_tilde_social[1]*q_tilde_social[1] +
                                          q_tilde_social[2]*q_tilde_social[2] +
                                          q_tilde_social[3]*q_tilde_social[3]
                                          );
        /*
         std::cout<<"norm_v_tilde_social: "
         <<norm_v_tilde_social<<"\n";
         */
        
        double w_tilde_social[3]={0.0,0.0,0.0};
        if (norm_v_tilde_social>MIN_NORM)
        {
            w_tilde_social[0] =
            /*scaleW*/2.0*(q_tilde_social[1] / norm_v_tilde_social)*
            std::acos(q_tilde_social[0]);
            
            w_tilde_social[1] =
            /*scaleW*/2.0*(q_tilde_social[2] / norm_v_tilde_social)*
            std::acos(q_tilde_social[0]);
            
            w_tilde_social[2] =
            /*scaleW*/2.0*(q_tilde_social[3] / norm_v_tilde_social)*
            std::acos(q_tilde_social[0]);
        }
        /*
         std::cout<<"w_tilde_social\n";
         printVector(w_tilde_social,3);
         */
        //cv::waitKey();
        /** END SOCIAL PART **/
        
        /**UPDATE the Angular Velocity**/
        pso_vel[idx_swarm_particle][q0_] = 0.0;
        /*
         std::cout<<"Before Update the Ang Vel\n";
         printAngVelPSOstateVect(pso_vel[idx_swarm_particle],(int)idx_swarm_particle);
         */
        for (int w_idx=1; w_idx<4; ++w_idx) {
            
            // calculate stochastic coefficients
            pso_rho1 = settings.c1 * ((*UniDisPtr)(*genPtr)); //((*UniDisPtr)(*genPtr));
            pso_rho2 = settings.c2 * ((*UniDisPtr)(*genPtr));
            pso_vel[idx_swarm_particle][w_idx+q0_] =
            pso_w * pso_vel[idx_swarm_particle][w_idx+q0_] +
            pso_rho1*w_tilde_cognitive[w_idx-1] +
            pso_rho2*w_tilde_social[w_idx-1];
        }
        /*
         std::cout<<"Update the Ang Vel\n";
         printAngVelPSOstateVect(pso_vel[idx_swarm_particle],(int)idx_swarm_particle);
         */
        
        /**UPDATE the Quaternions**/
        /*        double norm_pso_vel_quat = sqrt(
         pso_vel[idx_swarm_particle][q1_]*pso_vel[idx_swarm_particle][q1_]+
         pso_vel[idx_swarm_particle][q2_]*pso_vel[idx_swarm_particle][q2_]+
         pso_vel[idx_swarm_particle][q3_]*pso_vel[idx_swarm_particle][q3_]);
         
         double cosW=0.0;
         double sinW=0.0;
         double Tc=1.0;
         double domega = norm_pso_vel_quat*Tc*0.5;
         cosW = cos(domega);
         if (domega<MIN_NORM) {
         sinW=1.0;
         }
         else{
         sinW = sin(domega)/domega;
         }
         
         double qXw[4];
         
         QuatProd_pqPSOstateVect(pso_pos[idx_swarm_particle],pso_vel[idx_swarm_particle],qXw);
         
         for (int w_idx=q0_; w_idx<=q3_; ++w_idx) {
         
         pso_pos[idx_swarm_particle][w_idx] = cosW*pso_pos[idx_swarm_particle][w_idx] + 0.5*sinW*qXw[w_idx-q0_]*Tc;
         }
         NormalizeQuaternionPSOstateVect(pso_pos[idx_swarm_particle]);
         Cast2TopHalfHyperspherePSOstateVect(pso_pos[idx_swarm_particle]);
         std::cout<<"Quat Updated\n";
         printQuatPSOstateVect(pso_pos[idx_swarm_particle]);
         //cv::waitKey();
         */
        QuatKinematicsPSOStateVect(pso_pos[idx_swarm_particle],pso_vel[idx_swarm_particle]);
        //TODO: Enforce Boundary Condition???? MayBe not needed at all !!!!
        /*
         printQuatPSOstateVect(pso_pos[idx_swarm_particle],(int)idx_swarm_particle,"qnext");
         std::cout<<"PSO pos\n";
         printVector(pso_pos[idx_swarm_particle], 7);
         std::cout<<"PSO vel\n";
         printVector(pso_vel[idx_swarm_particle], 7);
         
         std::cout<<"--------\n\n";
         */
        //cv::waitKey();
        
        
    }//fine if !pso_first_step
    
    
    //std::cout<<"updateEqsParticleByParticleFine\n";
}

void PSOClass::updatePersonalAndGBest(size_t idx_swarm_particle)
{
    //std::cout<<"updatePersonalAndGBestInizio\n";
    // update personal best position?
    if (pso_fit[idx_swarm_particle] < pso_fit_b[idx_swarm_particle]) {
        pso_fit_b[idx_swarm_particle] = pso_fit[idx_swarm_particle];
        // copy contents of pos[idx_swarm_particle] to pos_b[idx_swarm_particle]
        memmove((void *)&pso_pos_b[idx_swarm_particle], (void *)&pso_pos[idx_swarm_particle],
                sizeof(double) * settings.dim);
        //std::cout<<"update personal best position?\n";
    }
    // update gbest??
    if (pso_fit[idx_swarm_particle] < solution.error)
    //if ((settings.step+1)%2==0  || settings.step>4)        // added by STE: update global only every 2 and after first
    {
        pso_improved = 1;
        printf("Updated global best\n");
        // update best fitness
        solution.error = pso_fit[idx_swarm_particle];
        
        // added by STE
        solution.zerror = zerrorvec[idx_swarm_particle];
        solution.ratio = ratiovec[idx_swarm_particle];
        
        // copy particle pos to gbest vector
        memmove((void *)solution.gbest, (void *)&pso_pos[idx_swarm_particle],
                sizeof(double) * settings.dim);
        // copy particle vel to gbestVec vector
        memmove((void *)solution.gbestVel, (void *)&pso_vel[idx_swarm_particle],
                sizeof(double) * settings.dim);
        //std::cout<<"update gbest??\n";
        
    }
    //std::cout<<"updatePersonalAndGBestFine\n";
}

void PSOClass::reInitFromBest(double delta_t, double delta_wX,double delta_wY,double delta_wZ)
{
    
    pso_improved = 0;
    pso_first_step = true;
    
    //TODO: RE-INIT BOUNDS (left unchanged now)
    
    printf("Current Best: \n");
    printVector(solution.gbest, Ndim);
    
    double pso_a, pso_b; // for matrix initialization
    
    // SWARM INITIALIZATION as perturbation of the gbest
    // for each particle
    for (int i_=0; i_<settings.size; ++i_) {
        // for each dimension
        for (int d_=0; d_<settings.dim; ++d_) {
            
            if (d_<3) {//tx ty tz
                double a = -delta_t;//0.01; //1cm
                double b = delta_t;//0.01;
                pso_a = solution.gbest[d_] + (a + (b-a)*((*UniDisPtr)(*genPtr)));
                pso_b = solution.gbest[d_] + (a + (b-a)*((*UniDisPtr)(*genPtr)));
                
                // initialize pose
                pso_pos[i_][d_] = pso_a;
                // best pose is the same
                pso_pos_b[i_][d_] = pso_a;
                // initialize velocity
                pso_vel[i_][d_] = (pso_a-pso_b) / 2.;
                
                
            }
            
        }//end for d_ dimension
        
        //Perturbate the Current Best Quaterion
        double qinit[7] = {0.,0.,0.,
            solution.gbest[q0_],
            solution.gbest[q1_],
            solution.gbest[q2_],
            solution.gbest[q3_]};//{0.,0.,0.,0.7071,0.,0.7071,0.};
        double wpert[7] = {0.,0.,0.,0.,0.,0.,0.};
        double aminX = -delta_wX;//-(2.*(double)i_/10.);
        double amaxX = delta_wX;//+(2.*(double)i_/10.);
        double aminY = -delta_wY;
        double amaxY = delta_wY;
        double aminZ = -delta_wZ;
        double amaxZ = delta_wZ;
        //std::cout<<"amax"<<amax<<"\n";
        wpert[q1_] = aminX + (amaxX-aminX)*((*UniDisPtr)(*genPtr));
        wpert[q2_] = aminY + (amaxY-aminY)*((*UniDisPtr)(*genPtr));
        wpert[q3_] = aminZ + (amaxZ-aminZ)*((*UniDisPtr)(*genPtr));
        
        QuatKinematicsPSOStateVectWithInit(qinit,pso_pos[i_],wpert);
        //the same for pso_pos_b
        std::copy(std::begin(pso_pos[i_]), std::end(pso_pos[i_]), std::begin(pso_pos_b[i_]));
        
        std::copy(wpert+q0_, std::end(wpert), pso_vel[i_]+q0_);
        
        printQuatPSOstateVect(pso_pos[i_],i_);
        
        //printQuatPSOstateVect(pso_pos_b[i_],i_);
        
        //printAngVelPSOstateVect(wpert,i_);
        printAngVelPSOstateVect(pso_vel[i_],i_);
        
    }//end i_ num particle
    
    //TODO: Deal with the Inertia Weight!!
    //Reset to max value or left unchanged???
    //if Reset to max value we need to upgrade the lin_dec_inertia since the pso steps do not reset to 0 after the reInit_from_Best() is called!!
    
    
}

//cv::Mat MV in double
void PSOClass::getT_ModelView(cv::Mat& MV, size_t idx_swarm_particle,
                              double offset_roll, double offset_pitch, double offset_yaw )
{
    
    if (MV.rows != 4 || MV.cols != 4)
    {
        std::cout<<"ERROR: (MV.rows != 4 || MV.cols != 4) \n";
        exit(EXIT_FAILURE);
    }
    
    cv::Mat Rquat = cv::Mat::zeros(3, 3, cv::DataType<double>::type);
    double* rotquat_ptr =  Rquat.ptr<double>(0);
    //***first row
    *(rotquat_ptr++) = (pso_pos[idx_swarm_particle][q2_]*pso_pos[idx_swarm_particle][q2_])*-2.0-(pso_pos[idx_swarm_particle][q3_]*pso_pos[idx_swarm_particle][q3_])*2.0+1.0;
    *(rotquat_ptr++) = pso_pos[idx_swarm_particle][q0_]*pso_pos[idx_swarm_particle][q3_]*-2.0+pso_pos[idx_swarm_particle][q1_]*pso_pos[idx_swarm_particle][q2_]*2.0;
    *(rotquat_ptr++) = pso_pos[idx_swarm_particle][q0_]*pso_pos[idx_swarm_particle][q2_]*2.0+pso_pos[idx_swarm_particle][q1_]*pso_pos[idx_swarm_particle][q3_]*2.0;
    //***second row
    *(rotquat_ptr++) = pso_pos[idx_swarm_particle][q0_]*pso_pos[idx_swarm_particle][q3_]*2.0+pso_pos[idx_swarm_particle][q1_]*pso_pos[idx_swarm_particle][q2_]*2.0;
    *(rotquat_ptr++) = (pso_pos[idx_swarm_particle][q1_]*pso_pos[idx_swarm_particle][q1_])*-2.0-(pso_pos[idx_swarm_particle][q3_]*pso_pos[idx_swarm_particle][q3_])*2.0+1.0;
    *(rotquat_ptr++) = pso_pos[idx_swarm_particle][q0_]*pso_pos[idx_swarm_particle][q1_]*-2.0+pso_pos[idx_swarm_particle][q2_]*pso_pos[idx_swarm_particle][q3_]*2.0;
    //***third row
    *(rotquat_ptr++) = pso_pos[idx_swarm_particle][q0_]*pso_pos[idx_swarm_particle][q2_]*-2.0+pso_pos[idx_swarm_particle][q1_]*pso_pos[idx_swarm_particle][q3_]*2.0;
    *(rotquat_ptr++) = pso_pos[idx_swarm_particle][q0_]*pso_pos[idx_swarm_particle][q1_]*2.0+pso_pos[idx_swarm_particle][q2_]*pso_pos[idx_swarm_particle][q3_]*2.0;
    *(rotquat_ptr++) = (pso_pos[idx_swarm_particle][q1_]*pso_pos[idx_swarm_particle][q1_])*-2.0-(pso_pos[idx_swarm_particle][q2_]*pso_pos[idx_swarm_particle][q2_])*2.0+1.0;
    
    //!!OK!!
    //    printQuatPSOstateVect(pso_pos[idx_swarm_particle]);
    //    std::cout<<"Rquat: "<<Rquat<<"\n";
    //    cv::waitKey();
    
    cv::Mat Rot(3, 3, cv::DataType<double>::type);
    RPY2Rot(Rot,offset_roll,offset_pitch,offset_yaw);
    
    cv::Mat Rot_ = Rquat * Rot;
    
    //LET'S create the Homogeneous T matrix from [ Rot_ | t ]
    double* Rot_ptr = Rot_.ptr<double>(0);
    double* MV_ptr  = MV.ptr<double>(0);
    //***first row
    *(MV_ptr++) = *(Rot_ptr++);
    *(MV_ptr++) = *(Rot_ptr++);
    *(MV_ptr++) = *(Rot_ptr++);
    *(MV_ptr++) = pso_pos[idx_swarm_particle][0]; //tx
    //***second row
    *(MV_ptr++) = *(Rot_ptr++);
    *(MV_ptr++) = *(Rot_ptr++);
    *(MV_ptr++) = *(Rot_ptr++);
    *(MV_ptr++) = pso_pos[idx_swarm_particle][1]; //ty
    //***third row
    *(MV_ptr++) = *(Rot_ptr++);
    *(MV_ptr++) = *(Rot_ptr++);
    *(MV_ptr++) = *(Rot_ptr++);
    *(MV_ptr++) = pso_pos[idx_swarm_particle][2]; //tz
    //***fourth row
    *(MV_ptr++) = 0.0;
    *(MV_ptr++) = 0.0;
    *(MV_ptr++) = 0.0;
    *(MV_ptr++) = 1.0;
}

//cv::Mat MV in double
void PSOClass::getT_ModelViewFromBest(cv::Mat& MV, double offset_roll, double offset_pitch, double offset_yaw )
{
    
    if (MV.rows != 4 || MV.cols != 4)
    {
        std::cout<<"ERROR: (MV.rows != 4 || MV.cols != 4) \n";
        exit(EXIT_FAILURE);
    }
    
    cv::Mat Rquat = cv::Mat::zeros(3, 3, cv::DataType<double>::type);
    double* rotquat_ptr =  Rquat.ptr<double>(0);
    //***first row
    *(rotquat_ptr++) = (solution.gbest[q2_]*solution.gbest[q2_])*-2.0-(solution.gbest[q3_]*solution.gbest[q3_])*2.0+1.0;
    *(rotquat_ptr++) = solution.gbest[q0_]*solution.gbest[q3_]*-2.0+solution.gbest[q1_]*solution.gbest[q2_]*2.0;
    *(rotquat_ptr++) = solution.gbest[q0_]*solution.gbest[q2_]*2.0+solution.gbest[q1_]*solution.gbest[q3_]*2.0;
    //***second row
    *(rotquat_ptr++) = solution.gbest[q0_]*solution.gbest[q3_]*2.0+solution.gbest[q1_]*solution.gbest[q2_]*2.0;
    *(rotquat_ptr++) = (solution.gbest[q1_]*solution.gbest[q1_])*-2.0-(solution.gbest[q3_]*solution.gbest[q3_])*2.0+1.0;
    *(rotquat_ptr++) = solution.gbest[q0_]*solution.gbest[q1_]*-2.0+solution.gbest[q2_]*solution.gbest[q3_]*2.0;
    //***third row
    *(rotquat_ptr++) = solution.gbest[q0_]*solution.gbest[q2_]*-2.0+solution.gbest[q1_]*solution.gbest[q3_]*2.0;
    *(rotquat_ptr++) = solution.gbest[q0_]*solution.gbest[q1_]*2.0+solution.gbest[q2_]*solution.gbest[q3_]*2.0;
    *(rotquat_ptr++) = (solution.gbest[q1_]*solution.gbest[q1_])*-2.0-(solution.gbest[q2_]*solution.gbest[q2_])*2.0+1.0;
    
    
    //!!OK!!
    //        printQuatPSOstateVect(solution.gbest);
    //        std::cout<<"Rquat: "<<Rquat<<"\n";
    //        cv::waitKey();
    
    cv::Mat Rot(3, 3, cv::DataType<double>::type);
    RPY2Rot(Rot,offset_roll,offset_pitch,offset_yaw);
    
    cv::Mat Rot_ = Rquat * Rot;
    
    //LET'S create the Homogeneous T matrix from [ Rot_ | t ]
    double* Rot_ptr = Rot_.ptr<double>(0);
    double* MV_ptr  = MV.ptr<double>(0);
    //***first row
    *(MV_ptr++) = *(Rot_ptr++);
    *(MV_ptr++) = *(Rot_ptr++);
    *(MV_ptr++) = *(Rot_ptr++);
    *(MV_ptr++) = solution.gbest[0]; //tx
    //***second row
    *(MV_ptr++) = *(Rot_ptr++);
    *(MV_ptr++) = *(Rot_ptr++);
    *(MV_ptr++) = *(Rot_ptr++);
    *(MV_ptr++) = solution.gbest[1]; //ty
    //***third row
    *(MV_ptr++) = *(Rot_ptr++);
    *(MV_ptr++) = *(Rot_ptr++);
    *(MV_ptr++) = *(Rot_ptr++);
    *(MV_ptr++) = solution.gbest[2]; //tz
    //***fourth row
    *(MV_ptr++) = 0.0;
    *(MV_ptr++) = 0.0;
    *(MV_ptr++) = 0.0;
    *(MV_ptr++) = 1.0;
}


//cv::Mat in double!!
void PSOClass::RPY2Rot(cv::Mat& Rot, double roll, double pitch, double yaw)
{
    //Rot = Rz(yaw)*Ry(pitch)*Rx(roll) starting from roll in [rad]
    double* rot_ptr =  Rot.ptr<double>(0);
    //***first row
    *(rot_ptr++) = cos(yaw)*cos(pitch);
    *(rot_ptr++) = cos(yaw)*sin(pitch)*sin(roll) - sin(yaw)*cos(roll);
    *(rot_ptr++) = cos(yaw)*sin(pitch)*cos(roll) + sin(yaw)*sin(roll);
    //***second row
    *(rot_ptr++) = sin(yaw)*cos(pitch);
    *(rot_ptr++) = sin(yaw)*sin(pitch)*sin(roll) + cos(yaw)*cos(roll);
    *(rot_ptr++) = sin(yaw)*sin(pitch)*cos(roll) - cos(yaw)*sin(roll);
    //***third row
    *(rot_ptr++) = -sin(pitch);
    *(rot_ptr++) = cos(pitch)*sin(roll);
    *(rot_ptr++) = cos(pitch)*cos(roll);
    
    /*  //cv::Mat Rot(3, 3, cv::DataType<float>::type);
     //***first row
     Rot.at<double>(0,0) = cos(yaw)*cos(pitch);
     Rot.at<double>(0,1) = cos(yaw)*sin(pitch)*sin(roll) - sin(yaw)*cos(roll);
     Rot.at<double>(0,2) = cos(yaw)*sin(pitch)*cos(roll) + sin(yaw)*sin(roll);
     //***second row
     Rot.at<double>(1,0) = sin(yaw)*cos(pitch);
     Rot.at<double>(1,1) = sin(yaw)*sin(pitch)*sin(roll) + cos(yaw)*cos(roll);
     Rot.at<double>(1,2) = sin(yaw)*sin(pitch)*cos(roll) - cos(yaw)*sin(roll);
     //***third row
     Rot.at<double>(2,0) = -sin(pitch);
     Rot.at<double>(2,1) = cos(pitch)*sin(roll);
     Rot.at<double>(2,2) = cos(pitch)*cos(roll);
     */
}
//cv::Mat in float!!
cv::Mat PSOClass::setRotMatx(cv::Mat& rVec, float roll,
                             float pitch, float yaw)
{
    //Rot = Rz(yaw)*Ry(pitch)*Rx(roll) starting from roll in [rad]
    //cv::Mat Rot(3, 3, cv::DataType<float>::type);
    cv::Mat Rot = cv::Mat::zeros(3, 3, cv::DataType<float>::type);
    float* rot_ptr =  Rot.ptr<float>(0);
    //***first row
    *(rot_ptr++) = cos(yaw)*cos(pitch);
    *(rot_ptr++) = cos(yaw)*sin(pitch)*sin(roll) - sin(yaw)*cos(roll);
    *(rot_ptr++) = cos(yaw)*sin(pitch)*cos(roll) + sin(yaw)*sin(roll);
    //***second row
    *(rot_ptr++) = sin(yaw)*cos(pitch);
    *(rot_ptr++) = sin(yaw)*sin(pitch)*sin(roll) + cos(yaw)*cos(roll);
    *(rot_ptr++) = sin(yaw)*sin(pitch)*cos(roll) - cos(yaw)*sin(roll);
    //***third row
    *(rot_ptr++) = -sin(pitch);
    *(rot_ptr++) = cos(pitch)*sin(roll);
    *(rot_ptr++) = cos(pitch)*cos(roll);
    //cv::Mat rVec(3, 1, cv::DataType<float>::type); // Rotation vector
    cv::Rodrigues(Rot, rVec);
    
    ////std::cout<<"rVec:\n"<<rVec<<"\n";
    
    return Rot;
}

