#include "../include/nov_5.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

using namespace Eigen;
using namespace std;

const double YITA = 1E-10;

VectorXd nov_5(const VectorXd& f, double h) {
    int n = f.size();
    VectorXd de = VectorXd::Zero(n + 1);
    MatrixXd wc = MatrixXd::Zero(n + 1, 3);
    MatrixXd gama = MatrixXd::Zero(n + 1, 3);
    MatrixXd is = MatrixXd::Zero(n + 1, 3);
    
    // Coefficient matrix A
    MatrixXd A = MatrixXd::Zero(n + 1, n + 1);

    // boundary conditions
    de(0) = f(0)*137/60 + f(1)*(-163)/60 + f(2)*137/60 + f(3)*(-21)/20 + f(4)*1/5;
    de(1) = f(0)*1/5 + f(1)*77/60 + f(2)*(-43)/60 + f(3)*17/60 + f(4)*(-1)/20;
    de(2) = f(0)*(-1)/20 + f(1)*9/20 + f(2)*47/60 + f(3)*(-13)/60 + f(4)*1/30;

    // A's boundary
    A(0, 0) = 1.0;
    A(1, 1) = 1.0;
    A(2, 2) = 1.0;

    // main loop for the interior points
    for (int j = 2; j < n - 2; ++j) {
        // smoothness indicator
        is(j, 0) = 13.0/12.0 * pow(f(j-2) - 2*f(j-1) + f(j), 2) + 
                   0.25 * pow(f(j-2) - 4*f(j-1) + 3*f(j), 2);
        is(j, 1) = 13.0/12.0 * pow(f(j-1) - 2*f(j) + f(j+1), 2) + 
                   0.25 * pow(f(j-1) - f(j+1), 2);
        is(j, 2) = 13.0/12.0 * pow(f(j) - 2*f(j+1) + f(j+2), 2) + 
                   0.25 * pow(3*f(j) - 4*f(j+1) + f(j+2), 2);

        // gamma weights
        gama(j, 0) = 0.05555555555555556 / (YITA + is(j, 0));
        gama(j, 1) = 0.44444444444444444 / (YITA + is(j, 1));
        gama(j, 2) = 0.5 / (YITA + is(j, 2));
        
        double sss = gama(j, 0) + gama(j, 1) + gama(j, 2);
        wc(j, 0) = gama(j, 0) / sss;
        wc(j, 1) = gama(j, 1) / sss;
        wc(j, 2) = gama(j, 2) / sss;

        // 
        de(j + 1) = (0.5*wc(j,0) ) * f(j-1) + 
                   (2.5*wc(j,0) + 1.25*wc(j,1) + 0.222222222222222*wc(j,2)) * f(j) + 
                   (0.25*wc(j,1)+1.38888888888889*wc(j,2)) * f(j+1) + 
                   (0.05555555555555556*wc(j,2)) * f(j+2);

        // A
        A(j + 1, j ) = 2.0 * wc(j, 0) + 0.5 * wc(j, 1);
        A(j + 1, j + 1)     = 1.0;
        A(j + 1, j + 2) = 0.66666666666666666666667 * wc(j, 2);
    }

    // boundary conditions for the last two points
    de(n - 1) = f(n-5)*(-1)/20 + f(n-4)*17/60 + f(n-3)*(-43)/60 + f(n-2)*77/60 + f(n-1)*1/5;
    de(n)     = f(n-5)*1/5 + f(n-4)*(-21)/20 + f(n-3)*137/60 + f(n-2)*(-163)/60 + f(n-1)*137/60;

    // A's boundary for the last two points
    A(n - 1, n - 1) = 1.0;
    A(n, n) = 1.0;

    // linear system solution
    VectorXd der = A.completeOrthogonalDecomposition().solve(de);
    /*std::string filename = "A.dat";

    // Save data for every time step
    std::ofstream outfile(filename);

    if (outfile.is_open())
    {
        // Write data for all grid points
        outfile << A(0, 0) << "," << A(0, 1) << "," << A(0, 2) << "," << A(0, 3) << A(0, 4) << ",   ," << de(0) << ",   ," << der(0) << "\n";
        outfile << A(1, 0) << "," << A(1, 1) << "," << A(1, 2) << "," << A(1, 3) << A(1, 4) << ",   ," << de(1) << ",   ," << der(1) << "\n";

        for (int j = 2; j < n - 1; ++j)
            outfile << A(j, j - 2) << "," << A(j, j - 1) << "," << A(j, j) << "," << A(j, j + 1) << "," << A(j, j + 2) << ",   ," << de(j) << ",   ," << der(j) << "\n";
        outfile << A(n - 1, n - 4) << "," << A(n - 1, n - 3) << "," << A(n - 1, n - 2) << "," << A(n - 1, n - 1) << "," << A(n - 1, n) << ",   ," << de(n - 1) << ",   ," << der(n - 1) << "\n";
        outfile << A(n, n - 4) << "," << A(n, n - 3) << "," << A(n, n - 2) << "," << A(n, n - 1) << "," << A(n, n) << ",   ," << de(n) << ",   ," << der(n) << "\n";
        outfile.close();
    }
    else
    {
        std::cerr << "Error: Unable to open file " << filename << "\n";
    }*/

    // derivative results
    VectorXd result(n);
    for (int i = 0; i < n; ++i) {
        result(i) = (der(i + 1) - der(i)) / h;
    }

    //cout<<"de: " << de(n-5) <<","<< de(n-4) <<","<< de(n-3) <<","<< de(n-2) <<","<< de(n-1) <<","<< de(n) <<endl;
    //cout<<"der: " << der(n-5) <<","<< der(n-4) <<","<< der(n-3) <<","<< der(n-2) <<","<< der(n-1) <<","<< der(n) <<endl;
    //cout<<"results: " << result(n-5) <<","<< result(n-4) <<","<< result(n-3) <<","<< result(n-2) <<","<< result(n-1) <<endl;
    //cout<<"wc99: " << wc(99,0)<<","<< wc(99,1) <<","<< wc(99,2) <<endl;
    //cout<<"wc100: " << wc(100,0)<<","<< wc(100,1) <<","<< wc(100,2) <<endl;
    //cout<<"wc101: " << wc(101,0)<<","<< wc(101,1) <<","<< wc(101,2) <<endl;
    //cout<<"f: " << f(99) <<","<< f(100) <<","<<  f(101) <<","<< f(102)<<","<< f(103) <<endl;
    //cout<< "Result: " << result.transpose() << endl;
    return result;
}

VectorXd nov_5_n(const VectorXd& f, double h) {
    int n = f.size();
    VectorXd de = VectorXd::Zero(n + 1);
    MatrixXd wc = MatrixXd::Zero(n + 1, 3);
    MatrixXd gama = MatrixXd::Zero(n + 1, 3);
    MatrixXd is = MatrixXd::Zero(n + 1, 3);
    
    // coefficient matrix A
    MatrixXd A = MatrixXd::Zero(n + 1, n + 1);

    // boundary
    de(0) = f(0)*137/60 + f(1)*(-163)/60 + f(2)*137/60 + f(3)*(-21)/20 + f(4)*1/5;
    de(1) = f(0)*1/5 + f(1)*77/60 + f(2)*(-43)/60 + f(3)*17/60 + f(4)*(-1)/20;

    // 
    A(0, 0) = 1.0;
    A(1, 1) = 1.0;

    // 
    for (int j = 1; j < n - 3; ++j) {
        // smoothness indicator
        is(j, 0) = 13.0/12.0 * pow(f(j-1) - 2*f(j) + f(j+1), 2) + 
                   0.25 * pow(f(j-1) - 4*f(j) + 3*f(j+1), 2);
        is(j, 1) = 13.0/12.0 * pow(f(j) - 2*f(j+1) + f(j+2), 2) + 
                   0.25 * pow(f(j) - f(j+2), 2);
        is(j, 2) = 13.0/12.0 * pow(f(j+1) - 2*f(j+2) + f(j+3), 2) + 
                   0.25 * pow(3*f(j+1) - 4*f(j+2) + f(j+3), 2);

        // gamma weights
        gama(j, 0) = 0.5 / (YITA + is(j, 0));
        gama(j, 1) = 0.44444444444444444 / (YITA + is(j, 1));
        gama(j, 2) = 0.05555555555555556 / (YITA + is(j, 2));
        
        double sss = gama(j, 0) + gama(j, 1) + gama(j, 2);
        wc(j, 0) = gama(j, 0) / sss;
        wc(j, 1) = gama(j, 1) / sss;
        wc(j, 2) = gama(j, 2) / sss;

        // 
        de(j + 1) = (0.05555555555555556*wc(j,0)) * f(j-1) + 
                   (1.38888888888889*wc(j,0) + 0.25*wc(j,1)) * f(j) + 
                   (0.222222222222222*wc(j,0) + 1.25*wc(j,1) + 2.5*wc(j,2)) * f(j+1) + 
                   (0.5*wc(j,2)) * f(j+2);

        // 
        A(j + 1, j)     = 0.66666666666666666666667 * wc(j, 0);
        A(j + 1, j + 1) = 1.0;
        A(j + 1, j + 2) = 0.5 * wc(j, 1) + 2.0 * wc(j, 2);
    }

    //
    de(n - 2) = f(n-5)*1/30 + f(n-4)*(-13)/60 + f(n-3)*47/60 + f(n-2)*9/20 + f(n-1)*(-1)/20;
    de(n - 1) = f(n-5)*(-1)/20 + f(n-4)*17/60 + f(n-3)*(-43)/60 + f(n-2)*77/60 + f(n-1)*1/5;
    de(n)     = f(n-5)*1/5 + f(n-4)*(-21)/20 + f(n-3)*137/60 + f(n-2)*(-163)/60 + f(n-1)*137/60;

    // 
    A(n - 2, n - 2) = 1.0;
    A(n - 1, n - 1) = 1.0;
    A(n, n) = 1.0;

    // linear system solution
    VectorXd der = A.completeOrthogonalDecomposition().solve(de);

    // 
    VectorXd result(n);
    for (int i = 0; i < n; ++i) {
        result(i) = (der(i + 1) - der(i)) / h;
    }
    //cout<< "Result: " << result.transpose() << endl;    
    return result;
}
