#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <thread>
#include <fstream>
#include <string>
#include <iomanip>  // For formatting filenames
#include <omp.h>
#include "../include/split.h"
#include "../include/derivatives.h"

using namespace Eigen;
using namespace std;

const double GAMMA = 1.4;
const double YITA = 1E-10;

int main() {
    int n = 201;
    VectorXd x = VectorXd::LinSpaced(n, -5, 5);
    double h = 10.0 / (n - 1);
    double dt = 0.0004;
    int M = ceil(2.0 / dt);
    
    // Initialize variables
    VectorXd rho = VectorXd::Ones(n);
    VectorXd u_vec = VectorXd::Zero(n);
    VectorXd p = VectorXd::Ones(n);
    
    // Set initial conditions
    int mid = (n + 1) / 2;
    rho.segment(mid - 1, n - mid + 1).setConstant(0.125);
    p.segment(mid - 1, n - mid + 1).setConstant(0.1);
    
    VectorXd E = p / 0.4 + 0.5 * rho.cwiseProduct(u_vec.cwiseProduct(u_vec));
    double Eo = E(0);
    double Eend = E(n - 1);
    VectorXd ru = rho.cwiseProduct(u_vec);
    
    // Main iteration loop
    for (int i = 0; i < M; ++i) {
        if (i % 100 == 0) {
            // Save data for every time step
            std::string filename = "timestep_" + std::to_string(i) + ".dat";
            std::ofstream outfile(filename);

            if (outfile.is_open()) {
                // Write headers (optional)
                outfile << "Rho\tU\tE\tP\n";

                // Write data for all grid points
                for (int j = 0; j < n; ++j) {
                    outfile << rho(j) << "\t" << u_vec(j) << "\t" 
                            << E(j) << "\t" << p(j) << "\n";
                }
                outfile.close();
            } else {
                std::cerr << "Error: Unable to open file " << filename << "\n";
            }
            cout<< "Saved data for time step " << i << " to " << filename << endl;
        }

        cout<<"main_omp.cpp time step: " << i << endl;
        // RK4 - First step
        auto [ep1, ep2, ep3, en1, en2, en3] = split(rho, p, u_vec);
        //cout<< "ep1: " << ep1.transpose() << endl;
        //cout<< "en3: " << en3.transpose() << endl;


        VectorXd rk1, ruk1, Ek1;
        VectorXd rk_p, rk_n, ruk_p, ruk_n, Ek_p, Ek_n;

        omp_set_num_threads(6);

        // 并行计算导数，6 个 section
        #pragma omp parallel sections
        {
            #pragma omp section
            { rk_p = -derivative_p(ep1, h) * dt;}
            #pragma omp section
            { rk_n = -derivative_n(en1, h) * dt; }
            #pragma omp section
            { ruk_p = -derivative_p(ep2, h) * dt; }
            #pragma omp section
            { ruk_n = -derivative_n(en2, h) * dt; }
            #pragma omp section
            { Ek_p= -derivative_p(ep3, h) * dt; }
            #pragma omp section
            { Ek_n = -derivative_n(en3, h) * dt; }
        }

        // 合并结果（在串行区域完成）
        rk1 = rk_p + rk_n;
        ruk1 = ruk_p + ruk_n;
        Ek1 = Ek_p + Ek_n;
        //cout<< "rk1: " << rk1.transpose() << endl;
        //cout<< "rku1: " << ruk1.transpose() << endl;
        //cout<< "Ek1: " << Ek1.transpose() << endl;       
        VectorXd ru1 = ru + ruk1 / 2.0;
        VectorXd r1 = rho + rk1 / 2.0;
        VectorXd u1 = ru1.cwiseQuotient(r1);
        VectorXd E1 = E + Ek1 / 2.0;
        VectorXd p1 = 0.4 * (E1 - 0.5 * ru1.cwiseProduct(u1));
        
        // Apply boundary conditions
        r1(0) = 1.0;
        r1.segment(n - 3, 3).setConstant(0.125);
        u1(0) = 0.0;
        u1.segment(n - 3, 3).setConstant(0.0);
        p1(0) = 1.0;
        p1.segment(n - 3, 3).setConstant(0.1);
        
        // RK4 - Second step
        tie(ep1, ep2, ep3, en1, en2, en3) = split(r1, p1, u1);
        VectorXd rk2, ruk2, Ek2;
        #pragma omp parallel sections
        {
            #pragma omp section
            { rk_p = -derivative_p(ep1, h) * dt; }
            #pragma omp section
            { rk_n = -derivative_n(en1, h) * dt; }
            #pragma omp section
            { ruk_p = -derivative_p(ep2, h) * dt; }
            #pragma omp section
            { ruk_n = -derivative_n(en2, h) * dt; }
            #pragma omp section
            { Ek_p= -derivative_p(ep3, h) * dt; }
            #pragma omp section
            { Ek_n = -derivative_n(en3, h) * dt; }
        }

        // 合并结果（在串行区域完成）
        rk2 = rk_p + rk_n;
        ruk2 = ruk_p + ruk_n;
        Ek2 = Ek_p + Ek_n;

        VectorXd ru2 = ru + ruk2 / 2.0;
        VectorXd r2 = rho + rk2 / 2.0;
        VectorXd u2 = ru2.cwiseQuotient(r2);
        VectorXd E2 = E + Ek2 / 2.0;
        VectorXd p2 = 0.4 * (E2 - 0.5 * ru2.cwiseProduct(u2));
        
        // Apply boundary conditions
        r2(0) = 1.0;
        r2.segment(n - 3, 3).setConstant(0.125);
        u2(0) = 0.0;
        u2.segment(n - 3, 3).setConstant(0.0);
        p2(0) = 1.0;
        p2.segment(n - 3, 3).setConstant(0.1);
        
        // RK4 - Third step
        tie(ep1, ep2, ep3, en1, en2, en3) = split(r2, p2, u2);
        VectorXd rk3, ruk3, Ek3;
        #pragma omp parallel sections
        {
            #pragma omp section
            { rk_p = -derivative_p(ep1, h) * dt; }
            #pragma omp section
            { rk_n = -derivative_n(en1, h) * dt; }
            #pragma omp section
            { ruk_p = -derivative_p(ep2, h) * dt; }
            #pragma omp section
            { ruk_n = -derivative_n(en2, h) * dt; }
            #pragma omp section
            { Ek_p= -derivative_p(ep3, h) * dt; }
            #pragma omp section
            { Ek_n = -derivative_n(en3, h) * dt; }
        }

        // 合并结果（在串行区域完成）
        rk3 = rk_p + rk_n;
        ruk3 = ruk_p + ruk_n;
        Ek3 = Ek_p + Ek_n;

        VectorXd ru3 = ru + ruk3;
        VectorXd r3 = rho + rk3;
        VectorXd u3 = ru3.cwiseQuotient(r3);
        VectorXd E3 = E + Ek3;
        VectorXd p3 = 0.4 * (E3 - 0.5 * ru3.cwiseProduct(u3));
        
        // Apply boundary conditions
        r3(0) = 1.0;
        r3.segment(n - 3, 3).setConstant(0.125);
        u3(0) = 0.0;
        u3.segment(n - 3, 3).setConstant(0.0);
        p3(0) = 1.0;
        p3.segment(n - 3, 3).setConstant(0.1);
        
        // RK4 - Fourth step
        tie(ep1, ep2, ep3, en1, en2, en3) = split(r3, p3, u3);
        VectorXd rk4, ruk4, Ek4;
        #pragma omp parallel sections
        {
            #pragma omp section
            { rk_p = -derivative_p(ep1, h) * dt; }
            #pragma omp section
            { rk_n = -derivative_n(en1, h) * dt; }
            #pragma omp section
            { ruk_p = -derivative_p(ep2, h) * dt; }
            #pragma omp section
            { ruk_n = -derivative_n(en2, h) * dt; }
            #pragma omp section
            { Ek_p= -derivative_p(ep3, h) * dt; }
            #pragma omp section
            { Ek_n = -derivative_n(en3, h) * dt; }
        }

        // 合并结果（在串行区域完成）
        rk4 = rk_p + rk_n;
        ruk4 = ruk_p + ruk_n;
        Ek4 = Ek_p + Ek_n;     

        // Update variables
        rho = rho + (rk1 + 2.0 * rk2 + 2.0 * rk3 + rk4) / 6.0;
        ru = ru + (ruk1 + 2.0 * ruk2 + 2.0 * ruk3 + ruk4) / 6.0;
        u_vec = ru.cwiseQuotient(rho);
        E = E + (Ek1 + 2.0 * Ek2 + 2.0 * Ek3 + Ek4) / 6.0;
        p = 0.4 * (E - 0.5 * ru.cwiseProduct(u_vec));
        
        // Apply boundary conditions
        rho(0) = 1.0;
        u_vec(0) = 0.0;
        E(0) = Eo;
        p(0) = 1.0;
        rho.segment(n - 3, 3).setConstant(0.125);
        u_vec.segment(n - 3, 3).setConstant(0.0);
        E(n - 1) = Eend;
        p.segment(n - 3, 3).setConstant(0.1);
    
    }

    return 0;
}
