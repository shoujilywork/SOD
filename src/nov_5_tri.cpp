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

// 解三对角系统：a 是下对角 (长度 n-1), b 是主对角 (长度 n), c 是上对角 (长度 n-1), d 是右端项 (长度 n)
Eigen::VectorXd solveTridiagonal(const Eigen::VectorXd& a, const Eigen::VectorXd& b,
                                 const Eigen::VectorXd& c, const Eigen::VectorXd& d) {
    int n = b.size();
    Eigen::VectorXd cp(n - 1);  // modified upper diag
    Eigen::VectorXd dp(n);      // modified right-hand side
    Eigen::VectorXd x(n);       // solution

    // 初始化
    cp(0) = c(0) / b(0);
    dp(0) = d(0) / b(0);

    // 正向消去
    for (int i = 1; i < n; ++i) {
        double denom = b(i) - a(i - 1) * cp(i - 1);
        if (std::abs(denom) < 1e-14) {
            throw std::runtime_error("Zero pivot encountered in Thomas algorithm.");
        }
        if (i < n - 1) {
            cp(i) = c(i) / denom;
        }
        dp(i) = (d(i) - a(i - 1) * dp(i - 1)) / denom;
    }

    // 回代
    x(n - 1) = dp(n - 1);
    for (int i = n - 2; i >= 0; --i) {
        x(i) = dp(i) - cp(i) * x(i + 1);
    }

    return x;
}

VectorXd nov_5(const VectorXd& f, double h) {
    int n = f.size();
    VectorXd de = VectorXd::Zero(n + 1);
    MatrixXd wc = MatrixXd::Zero(n + 1, 3);
    MatrixXd gama = MatrixXd::Zero(n + 1, 3);
    MatrixXd is = MatrixXd::Zero(n + 1, 3);
    
    // 使用稠密矩阵替代稀疏矩阵
    MatrixXd A = MatrixXd::Zero(n + 1, n + 1);

    // 边界条件
    de(0) = f(0)*137/60 + f(1)*(-163)/60 + f(2)*137/60 + f(3)*(-21)/20 + f(4)*1/5;
    de(1) = f(0)*1/5 + f(1)*77/60 + f(2)*(-43)/60 + f(3)*17/60 + f(4)*(-1)/20;
    de(2) = f(0)*(-1)/20 + f(1)*9/20 + f(2)*47/60 + f(3)*(-13)/60 + f(4)*1/30;

    VectorXd a = VectorXd::Zero(n );
    VectorXd b = VectorXd::Ones(n + 1);
    VectorXd c = VectorXd::Zero(n );
    // 直接设置边界行为单位矩阵
 
    // 主循环
    for (int j = 2; j < n - 2; ++j) {
        // 计算平滑指标
        is(j, 0) = 13.0/12.0 * pow(f(j-2) - 2*f(j-1) + f(j), 2) + 
                   0.25 * pow(f(j-2) - 4*f(j-1) + 3*f(j), 2);
        is(j, 1) = 13.0/12.0 * pow(f(j-1) - 2*f(j) + f(j+1), 2) + 
                   0.25 * pow(f(j-1) - f(j+1), 2);
        is(j, 2) = 13.0/12.0 * pow(f(j) - 2*f(j+1) + f(j+2), 2) + 
                   0.25 * pow(3*f(j) - 4*f(j+1) + f(j+2), 2);

        // 计算权重
        gama(j, 0) = 0.05555555555555556 / (YITA + is(j, 0));
        gama(j, 1) = 0.44444444444444444 / (YITA + is(j, 1));
        gama(j, 2) = 0.5 / (YITA + is(j, 2));
        
        double sss = gama(j, 0) + gama(j, 1) + gama(j, 2);
        wc(j, 0) = gama(j, 0) / sss;
        wc(j, 1) = gama(j, 1) / sss;
        wc(j, 2) = gama(j, 2) / sss;

        // 导数近似
        de(j + 1) = (0.5*wc(j,0) ) * f(j-1) + 
                   (2.5*wc(j,0) + 1.25*wc(j,1) + 0.222222222222222*wc(j,2)) * f(j) + 
                   (0.25*wc(j,1)+1.38888888888889*wc(j,2)) * f(j+1) + 
                   (0.05555555555555556*wc(j,2)) * f(j+2);

        // 填充稠密矩阵A
        a(j - 1) = 2.0 * wc(j, 0) + 0.5 * wc(j, 1);
        c(j + 1) = 0.66666666666666666666667 * wc(j, 2);
    }

    // 边界条件
    de(n - 1) = f(n-5)*(-1)/20 + f(n-4)*17/60 + f(n-3)*(-43)/60 + f(n-2)*77/60 + f(n-1)*1/5;
    de(n)     = f(n-5)*1/5 + f(n-4)*(-21)/20 + f(n-3)*137/60 + f(n-2)*(-163)/60 + f(n-1)*137/60;


    // 求解线性系统
    VectorXd der = solveTridiagonal(a, b, c, de);
    
    // 计算导数
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
    
    // 使用稠密矩阵替代稀疏矩阵
    MatrixXd A = MatrixXd::Zero(n + 1, n + 1);

    // 边界条件
    de(0) = f(0)*137/60 + f(1)*(-163)/60 + f(2)*137/60 + f(3)*(-21)/20 + f(4)*1/5;
    de(1) = f(0)*1/5 + f(1)*77/60 + f(2)*(-43)/60 + f(3)*17/60 + f(4)*(-1)/20;

    // 直接设置边界行为单位矩阵
    A(0, 0) = 1.0;
    A(1, 1) = 1.0;

    // 主循环
    for (int j = 1; j < n - 3; ++j) {
        // 计算平滑指标
        is(j, 0) = 13.0/12.0 * pow(f(j-1) - 2*f(j) + f(j+1), 2) + 
                   0.25 * pow(f(j-1) - 4*f(j) + 3*f(j+1), 2);
        is(j, 1) = 13.0/12.0 * pow(f(j) - 2*f(j+1) + f(j+2), 2) + 
                   0.25 * pow(f(j) - f(j+2), 2);
        is(j, 2) = 13.0/12.0 * pow(f(j+1) - 2*f(j+2) + f(j+3), 2) + 
                   0.25 * pow(3*f(j+1) - 4*f(j+2) + f(j+3), 2);

        // 计算权重
        gama(j, 0) = 0.5 / (YITA + is(j, 0));
        gama(j, 1) = 0.44444444444444444 / (YITA + is(j, 1));
        gama(j, 2) = 0.05555555555555556 / (YITA + is(j, 2));
        
        double sss = gama(j, 0) + gama(j, 1) + gama(j, 2);
        wc(j, 0) = gama(j, 0) / sss;
        wc(j, 1) = gama(j, 1) / sss;
        wc(j, 2) = gama(j, 2) / sss;

        // 导数近似
        de(j + 1) = (0.05555555555555556*wc(j,0)) * f(j-1) + 
                   (1.38888888888889*wc(j,0) + 0.25*wc(j,1)) * f(j) + 
                   (0.222222222222222*wc(j,0) + 1.25*wc(j,1) + 2.5*wc(j,2)) * f(j+1) + 
                   (0.5*wc(j,2)) * f(j+2);

        // 填充稠密矩阵A
        A(j + 1, j)     = 0.66666666666666666666667 * wc(j, 0);
        A(j + 1, j + 1) = 1.0;
        A(j + 1, j + 2) = 0.5 * wc(j, 1) + 2.0 * wc(j, 2);
    }

    // 边界条件
    de(n - 2) = f(n-5)*1/30 + f(n-4)*(-13)/60 + f(n-3)*47/60 + f(n-2)*9/20 + f(n-1)*(-1)/20;
    de(n - 1) = f(n-5)*(-1)/20 + f(n-4)*17/60 + f(n-3)*(-43)/60 + f(n-2)*77/60 + f(n-1)*1/5;
    de(n)     = f(n-5)*1/5 + f(n-4)*(-21)/20 + f(n-3)*137/60 + f(n-2)*(-163)/60 + f(n-1)*137/60;

    // 设置边界行为单位矩阵
    A(n - 2, n - 2) = 1.0;
    A(n - 1, n - 1) = 1.0;
    A(n, n) = 1.0;

    // 求解线性系统
    VectorXd der = A.completeOrthogonalDecomposition().solve(de);

    // 计算导数
    VectorXd result(n);
    for (int i = 0; i < n; ++i) {
        result(i) = (der(i + 1) - der(i)) / h;
    }
    //cout<< "Result: " << result.transpose() << endl;    
    return result;
}
