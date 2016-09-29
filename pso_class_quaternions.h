//
//  pso_class_quaternions.h
//
//
//  Created by Giorgio on 29/07/15.
//
//

#ifndef ____pso_class_quaternions__
#define ____pso_class_quaternions__

#ifndef M_PI
#define M_PI 3.141592653589793
#endif

#ifndef rad2deg
#define rad2deg 180.0f/M_PI
#endif

#ifndef deg2rad
#define deg2rad M_PI/180.0f
#endif

#define NSize 2048
#define Ndim 8 //tx,ty,tz, q0 q1 q2 q3, alfa (1DOF)

#define tx_ 0
#define ty_ 1
#define tz_ 2

#define q0_ 3
#define q1_ 4
#define q2_ 5
#define q3_ 6
#define alfa0_ 7

#define MIN_NORM 1.0e-7

//to perturb revolute articulated velocity
#define vart_ 1.0f //1.0f for laptop

#include <random>
#include <boost/shared_ptr.hpp>

//#include <stdio.h>
class PSOClass {
    
public: // CONSTANTS
    static const int PSO_MAX_SIZE = 2048; // max swarm size
    constexpr static const double PSO_INERTIA = 0.7298; // default value of w (see clerc02)
    
    
    // === NEIGHBORHOOD SCHEMES ===
    
    // global best topology
    static const unsigned int PSO_NHOOD_GLOBAL = 0;
    
    // ring topology
    static const unsigned int PSO_NHOOD_RING = 1;
    
    // Random neighborhood topology
    // **see http://clerc.maurice.free.fr/pso/random_topology.pdf**
    static const unsigned int PSO_NHOOD_RANDOM = 2;
    
    
    
    // === INERTIA WEIGHT UPDATE FUNCTIONS ===
    static const unsigned int PSO_W_CONST  = 0;
    static const unsigned int PSO_W_LIN_DEC = 1;
    
    
    static const char GOAL_REACHED = 0;
    static const char MAX_STEPS_REACHED = 1;
    static const char UNKNOWN_ERROR = -1;
    
    
public:
    // PSO SOLUTION struct -- Initialized by the user
    struct pso_result_t {
        double zerror;
        double ratio;

        double error;
        double *gbest; // should contain DIM elements!!
        double *gbestVel;// should contain DIM elements!!
        
    };
    // PSO SETTINGS struct
    struct pso_settings_t {
        
        int dim; // problem dimensionality
        double* x_lo; // lower range limit for each !!dimension should contain DIM elements!!
        double* x_hi; // higher range limit for each !!dimension should contain DIM elements!!
        double goal; // optimization goal (error threshold)
        
        int size; // swarm size (number of particles)
        int print_every; // ... N steps (set to 0 for no output)
        int steps; // maximum number of iterations
        int step; // current PSO step
        double c1; // cognitive coefficient
        double c2; // social coefficient
        double w_max; // max inertia weight value
        double w_min; // min inertia weight value
        
        int clamp_pos; // whether to keep particle position within defined bounds (TRUE)
        // or apply periodic boundary conditions (FALSE)
        int nhood_strategy; // neighborhood strategy (see PSO_NHOOD_*)
        int nhood_size; // neighborhood size (Only for Random Topology)
        int w_strategy; // inertia weight strategy (see PSO_W_*)
        
    };
    
    std::random_device randDev;
    boost::shared_ptr<std::mt19937> genPtr;//(randDev());
    boost::shared_ptr<std::uniform_real_distribution<double> > UniDisPtr;//(1, 2);
    
    
    /** PSO VARIABLES **/
    //pso settings
    pso_settings_t settings;
    // initialize GBEST solution
    pso_result_t solution;
    
    // Particles
    double pso_pos[NSize][Ndim]; // position matrix
    double pso_vel[NSize][Ndim]; // velocity matrix
    double pso_pos_b[NSize][Ndim]; // best position matrix
    double pso_fit[NSize]; // particle fitness vector
    double pso_fit_b[NSize]; // best fitness vector
    
    double zerrorvec[NSize];
    double ratiovec[NSize];
    
    // Swarm
    double pso_pos_nb[NSize][Ndim]; // what is the best informed
    // position for each particle
    int pso_comm[NSize][NSize]; // communications:who informs who
    // rows : those who inform
    // cols : those who are informed
    
    int pso_improved; // whether solution->error was improved during // the last iteration
    bool pso_first_step;
    
    double pso_rho1, pso_rho2; // random numbers (coefficients)
    double pso_w; // current omega
    
    double mTs;
    
    
public:
    //    PSOClass(/*int Ndim, int NSize,*/ std::vector<double> Dimbounds,
    //             double c1_ = 1.496, double c2_ = 1.496,double Ts=0.1,
    //             unsigned int dynamicInertia = PSO_W_LIN_DEC,
    //             unsigned int nhoodtype = PSO_NHOOD_GLOBAL,
    //             int maxSteps = 50,
    //             double ErrorGoal = 0.00005,
    //             int nhood_size = 5,//(Only for Random Topology)
    //             double w_max_ = PSO_INERTIA, double w_min_ = 0.3);
    
    PSOClass(/*int Ndim, int NSize,*/ std::vector<double> Dimbounds,
             double c1_ = 1, double c2_ = 1,double Ts=0.3,
             unsigned int dynamicInertia = PSO_W_LIN_DEC,
             unsigned int nhoodtype = PSO_NHOOD_GLOBAL,
             int nhood_size = 5,//(Only for Random Topology)
             double ErrorGoal = 0.00005, int maxSteps = 100,
             double w_max_ = PSO_INERTIA, double w_min_ = 0.3);
    
    
    ~PSOClass();
    
    //default true;
    void setFirstStep(bool fs);
    
    void InformGlobal(double *pos_nb);
    
    //TODO: fill it
    void updateEqsParticleByParticle(size_t idx_swarm_particle);
    
    void updatePersonalAndGBest(size_t idx_swarm_particle);
    
    cv::Mat setRotMatx(cv::Mat& rVec, float roll,
                       float pitch, float yaw);
    
    void reInitFromBest(double delta_t=0.01, double delta_wX=0.3,double delta_wY=0.8,double delta_wZ=0.3); //[m], [rad/s]
    
    void getT_ModelView(cv::Mat& MV, size_t idx_swarm_particle,
                        double offset_roll, double offset_pitch, double offset_yaw );
    void getT_ModelViewFromBest(cv::Mat& MV, double offset_roll, double offset_pitch, double offset_yaw );
    
    void RPY2Rot(cv::Mat& Rot, double roll, double pitch, double yaw);
    
    inline double get1DOFArticulated(size_t idx_swarm_particle)
    {
        return pso_pos[idx_swarm_particle][alfa0_];
    }
    
    inline void QuatProd_pq(const double* p, const double* q, double* q_tilde )
    {
        
        q_tilde[0] = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3];
        q_tilde[1] = p[1]*q[0] + p[0]*q[1] + p[2]*q[3] - p[3]*q[2];
        q_tilde[2] = p[2]*q[0] + p[0]*q[2] + p[3]*q[1] - p[1]*q[3];
        q_tilde[3] = p[3]*q[0] + p[0]*q[3] + p[1]*q[2] - p[2]*q[1];
    }
    inline void QuatProd_pq_1(const double* p, const double* q, double* q_tilde )
    {
        //q became the coniugate q=(s,vx,vy,vz) -> q"=(s,-vx,-vy,-vz)
        q_tilde[0] = p[0]*q[0] + p[1]*q[1] + p[2]*q[2] + p[3]*q[3];
        q_tilde[1] = p[1]*q[0] - p[0]*q[1] - p[2]*q[3] + p[3]*q[2];
        q_tilde[2] = p[2]*q[0] - p[0]*q[2] - p[3]*q[1] + p[1]*q[3];
        q_tilde[3] = p[3]*q[0] - p[0]*q[3] - p[1]*q[2] + p[2]*q[1];
    }
    
    inline void QuatProd_pqPSOstateVect(const double* p, const double* q, double* q_tilde )
    {
        q_tilde[0] = p[q0_]*q[q0_] - p[q1_]*q[q1_] - p[q2_]*q[q2_] - p[q3_]*q[q3_];
        q_tilde[1] = p[q1_]*q[q0_] + p[q0_]*q[q1_] + p[q2_]*q[q3_] - p[q3_]*q[q2_];
        q_tilde[2] = p[q2_]*q[q0_] + p[q0_]*q[q2_] + p[q3_]*q[q1_] - p[q1_]*q[q3_];
        q_tilde[3] = p[q3_]*q[q0_] + p[q0_]*q[q3_] + p[q1_]*q[q2_] - p[q2_]*q[q1_];
    }
    inline void QuatProd_pq_1PSOstateVect(const double* p, const double* q, double* q_tilde )
    {
        //q became the coniugate q=(s,vx,vy,vz) -> q"=(s,-vx,-vy,-vz)
        q_tilde[0] = p[q0_]*q[q0_] + p[q1_]*q[q1_] + p[q2_]*q[q2_] + p[q3_]*q[q3_];
        q_tilde[1] = p[q1_]*q[q0_] - p[q0_]*q[q1_] - p[q2_]*q[q3_] + p[q3_]*q[q2_];
        q_tilde[2] = p[q2_]*q[q0_] - p[q0_]*q[q2_] - p[q3_]*q[q1_] + p[q1_]*q[q3_];
        q_tilde[3] = p[q3_]*q[q0_] - p[q0_]*q[q3_] - p[q1_]*q[q2_] + p[q2_]*q[q1_];
    }
    inline void NormalizeQuaternionPSOstateVect(double *q)
    {
        double denom;
        denom = sqrt((q[q0_])*(q[q0_]) + (q[q1_])*(q[q1_]) + (q[q2_])*(q[q2_]) + (q[q3_])*(q[q3_]));
        if(denom > MIN_NORM) {
            q[q0_] = (q[q0_])/denom;
            q[q1_] = (q[q1_])/denom;
            q[q2_] = (q[q2_])/denom;
            q[q3_] = (q[q3_])/denom;
        }
        
    }
    inline void NormalizeQuaternion(double *q)
    {
        double denom;
        denom = sqrt((q[0])*(q[0]) + (q[1])*(q[1]) + (q[2])*(q[2]) + (q[3])*(q[3]));
        if(denom > MIN_NORM) {
            q[0] = (q[0])/denom;
            q[1] = (q[1])/denom;
            q[2] = (q[2])/denom;
            q[3] = (q[3])/denom;
        }
        
    }
    inline void Cast2TopHalfHyperspherePSOstateVect(double* q)
    {
        if(q[q0_]<0.0)//q0
        {
            q[q0_] = -q[q0_];
            q[q1_] = -q[q1_];
            q[q2_] = -q[q2_];
            q[q3_] = -q[q3_];
        }
        
    }
    inline void Cast2TopHalfHypersphere(double* q)
    {
        if(q[0]<0.0)//q0
        {
            q[0] = -q[0];
            q[1] = -q[1];
            q[2] = -q[2];
            q[3] = -q[3];
        }
        
    }
    inline void printQuatPSOstateVect(const double* q, int idx=-1, std::string name = "")
    {
        if (idx<0) {
            printf("q: %s [ %f, %f, %f, %f ]\n", name.c_str() ,q[q0_],q[q1_],q[q2_],q[q3_]);
        }
        else
        {
            printf("q: %s %d [ %f, %f, %f, %f ]\n",name.c_str(),idx,q[q0_],q[q1_],q[q2_],q[q3_]);
        }
        
    }
    inline void printQuat(const double* q, int idx=-1, std::string name = "")
    {
        if (idx<0) {
            printf("q: %s [ %f, %f, %f, %f ]\n",name.c_str(),q[0],q[1],q[2],q[3]);
        }
        else
        {
            printf("q: %s %d [ %f, %f, %f, %f ]\n",name.c_str(),idx,q[0],q[1],q[2],q[3]);
        }
        
    }
    
    inline void printAngVelPSOstateVect(const double* w, int idx=-1)
    {
        if (idx<0) {
            printf("w: [ %f, %f, %f, %f ]\n",w[q0_],w[q1_],w[q2_],w[q3_]);
        }
        else
        {
            printf("w: %d: [ %f, %f, %f, %f ]\n",idx,w[q0_],w[q1_],w[q2_],w[q3_]);
        }
        
    }
    
    inline void printLinearVelPSOstateVect(const double* w, int idx=-1)
    {
        if (idx<0) {
            printf("vdot: [ %f, %f, %f ]\n",w[0],w[1],w[2]);
        }
        else
        {
            printf("vdot: %d: [ %f, %f, %f ]\n",idx,w[0],w[1],w[2]);
        }
        
    }
    inline void printLinearPositionPSOstateVect(const double* w, int idx=-1)
    {
        if (idx<0) {
            printf("x: [ %f, %f, %f ]\n",w[0],w[1],w[2]);
        }
        else
        {
            printf("x: %d: [ %f, %f, %f ]\n",idx,w[0],w[1],w[2]);
        }
        
    }
    
    
    inline void printVector(const double* w, int size)
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
    inline void printVector(const int* w, int size)
    {
        printf("[ ");
        for (int i=0; i<size; ++i) {
            if (i==size-1) {
                printf("%d ",w[i]);
            }
            else
                printf("%d, ",w[i]);
        }
        printf("]\n");
    }
    inline void generalQuatProduct(const double q1[4], double s1, const double q2[4], double s2,double q[4])
    {
        //
        // s1 = direct s1= -1 inverse
        // Arguments    : const double q1[4]
        //                double s1
        //                const double q2[4]
        //                double s2
        //                double q[4]
        // Return Type  : void
        
        
        double d0;
        double q1v[3];
        double q2v[3];
        int i;
        double b_q1v;
        double b_q2v;
        double dv0[9];
        double b_q1[3];
        int i0;
        
        //
        d0 = 0.0;
        for (i = 0; i < 3; i++) {
            b_q1v = s1 * q1[i + 1];
            b_q2v = s2 * q2[i + 1];
            d0 += b_q1v * b_q2v;
            q1v[i] = b_q1v;
            q2v[i] = b_q2v;
        }
        
        dv0[0] = 0.0;
        dv0[3] = -q1v[2];
        dv0[6] = q1v[1];
        dv0[1] = q1v[2];
        dv0[4] = 0.0;
        dv0[7] = -q1v[0];
        dv0[2] = -q1v[1];
        dv0[5] = q1v[0];
        dv0[8] = 0.0;
        for (i = 0; i < 3; i++) {
            b_q1v = 0.0;
            for (i0 = 0; i0 < 3; i0++) {
                b_q1v += dv0[i + 3 * i0] * q2v[i0];
            }
            
            b_q1[i] = (q1[0] * q2v[i] + q2[0] * q1v[i]) + b_q1v;
        }
        
        q[0] = q1[0] * q2[0] - d0;
        for (i = 0; i < 3; i++) {
            q[i + 1] = b_q1[i];
        }
    }
    
    inline void generalQuatProductPSOStateVect(const double q1[7], double s1, const double q2[7], double s2, double q[4])
    {
        // s1 = direct s1= -1 inverse
        // Arguments    : const double q1[7]
        //                double s1
        //                const double q2[7]
        //                double s2
        //                double q[4]
        // Return Type  : void
        
        double d0;
        double q1v[3];
        double q2v[3];
        int i0;
        double b_q1v;
        double b_q2v;
        double dv0[9];
        double b_q1[3];
        int i1;
        
        //
        d0 = 0.0;
        for (i0 = 0; i0 < 3; i0++) {
            b_q1v = s1 * q1[4 + i0];
            b_q2v = s2 * q2[4 + i0];
            d0 += b_q1v * b_q2v;
            q1v[i0] = b_q1v;
            q2v[i0] = b_q2v;
        }
        
        dv0[0] = 0.0;
        dv0[3] = -q1v[2];
        dv0[6] = q1v[1];
        dv0[1] = q1v[2];
        dv0[4] = 0.0;
        dv0[7] = -q1v[0];
        dv0[2] = -q1v[1];
        dv0[5] = q1v[0];
        dv0[8] = 0.0;
        for (i0 = 0; i0 < 3; i0++) {
            b_q1v = 0.0;
            for (i1 = 0; i1 < 3; i1++) {
                b_q1v += dv0[i0 + 3 * i1] * q2v[i1];
            }
            
            b_q1[i0] = (q1[3] * q2v[i0] + q2[3] * q1v[i0]) + b_q1v;
        }
        
        q[0] = q1[3] * q2[3] - d0;
        for (i0 = 0; i0 < 3; i0++) {
            q[i0 + 1] = b_q1[i0];
        }
        
    }
    
    
    inline void QuatKinematics(const double* q, const double* w, double* qnext)
    {   //q[4] ; w[4] ; qnext[4];
        //w[0]=0.0 because w=[0;wx;wy;wz]
        double w_temp[4]={0};
        
        w_temp[1] = w[1]*mTs;
        w_temp[2] = w[2]*mTs;
        w_temp[3] = w[3]*mTs;
        double norm_pso_vel_quat = sqrt(w_temp[1]*w_temp[1]+
                                        w_temp[2]*w_temp[2]+
                                        w_temp[3]*w_temp[3]);
        double cosW=0.0;
        double sinW=0.0;
        //double Tc=1.0;
        double domega = norm_pso_vel_quat*0.5;
        cosW = cos(domega);
        if (domega<MIN_NORM) {
            sinW=1.0;
        }
        else{
            sinW = sin(domega)/domega;
        }
        
        double qXw[4];
        
        QuatProd_pq(q,w_temp,qXw);
        
        for (int w_idx=0; w_idx<4; ++w_idx) {
            
            qnext[w_idx] = cosW*q[w_idx] + 0.5*sinW*qXw[w_idx];
        }
        NormalizeQuaternion(qnext);
        Cast2TopHalfHypersphere(qnext);
        //std::cout<<"Quat Updated\n";
        //printQuat(qnext);
    }
    
    inline void QuatKinematicsPSOStateVect(double* q, const double* w)
    {
        //q[7] ; w[7]
        /** OverWrite q that is pso_pos[i][q0:q3]**/
        /** OverWrite w [q1:q3]**/
        //w[q0_]=0.0 because w=[vx;vy;vz;0;wx;wy;wz]
        
        double w_temp[7]={0};
        
        w_temp[q1_] = w[q1_]*mTs;
        w_temp[q2_] = w[q2_]*mTs;
        w_temp[q3_] = w[q3_]*mTs;
        
        double norm_pso_vel_quat = sqrt(w_temp[q1_]*w_temp[q1_]+
                                        w_temp[q2_]*w_temp[q2_]+
                                        w_temp[q3_]*w_temp[q3_]);
        double cosW=0.0;
        double sinW=0.0;
        //double Tc=5;
        double domega = norm_pso_vel_quat*0.5;
        cosW = cos(domega);
        if (domega<MIN_NORM) {
            sinW=1.0;
        }
        else{
            sinW = sin(domega)/domega;
        }
        
        double qXw[4];
        //q[7], w[7], qxw[4]
        QuatProd_pqPSOstateVect(q,w_temp,qXw);
        
        for (int w_idx=q0_; w_idx<=q3_; ++w_idx) {
            
            q[w_idx] = cosW*q[w_idx] + 0.5*sinW*qXw[w_idx-q0_];
        }
        NormalizeQuaternionPSOstateVect(q);
        Cast2TopHalfHyperspherePSOstateVect(q);
        //std::cout<<"Quat Updated\n";
        //printQuatPSOstateVect(q);
    }
    inline void QuatKinematicsPSOStateVectWithInit(double* qinit, double* q, const double* w)
    {
        //q[7] ; w[7]
        /** OverWrite q that is pso_pos[i][q0:q3]**/
        /** OverWrite w [q1:q3]**/
        //w[q0_]=0.0 because w=[vx;vy;vz;0;wx;wy;wz]
        
        double w_temp[7]={0};
        
        w_temp[q1_] = w[q1_]*mTs;
        w_temp[q2_] = w[q2_]*mTs;
        w_temp[q3_] = w[q3_]*mTs;
        
        double norm_pso_vel_quat = sqrt(w_temp[q1_]*w_temp[q1_]+
                                        w_temp[q2_]*w_temp[q2_]+
                                        w_temp[q3_]*w_temp[q3_]);
        double cosW=0.0;
        double sinW=0.0;
        //double Tc=1.0;
        double domega = norm_pso_vel_quat*0.5;
        cosW = cos(domega);
        if (domega<MIN_NORM) {
            sinW=1.0;
        }
        else{
            sinW = sin(domega)/domega;
        }
        
        double qXw[4];
        
        QuatProd_pqPSOstateVect(qinit,w_temp,qXw);
        
        
        for (int w_idx=q0_; w_idx<=q3_; ++w_idx) {
            
            q[w_idx] = cosW*qinit[w_idx] + 0.5*sinW*qXw[w_idx-q0_];
        }
        NormalizeQuaternionPSOstateVect(q);
        Cast2TopHalfHyperspherePSOstateVect(q);
        //std::cout<<"Quat Updated\n";
        //printQuatPSOstateVect(q);
    }
    
    inline void setSamplingTime(double Ts)
    {
        this->mTs = Ts;
    }
    inline double getSamplingTime(void)
    {
        return this->mTs;
    }
    inline void setInertia(double w)
    {
        if(w<settings.w_min)
            this->pso_w = settings.w_min;
        else if(w>settings.w_max)
            this->pso_w = settings.w_max;
        else
            this->pso_w = w;
        return;
        
        
    }
    inline double getInertia(void)
    {
        return this->pso_w;
    }
    inline double calc_inertia_lin_dec(int step, pso_settings_t *settings) {
        
        int dec_stage = 3 * settings->steps / 4;
        if (step <= dec_stage)
            return settings->w_min + (settings->w_max - settings->w_min) *	\
            (dec_stage - step) / dec_stage;
        else
            return settings->w_min;
    }
    void init_comm_ring(int *comm, pso_settings_t * settings);
    void Inform(int *comm, double *pos_nb, double *pos_b, double *fit_b,int improved, pso_settings_t * settings);
    
    
    
};





#endif /* defined(____pso_class_quaternions__) */
