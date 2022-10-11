#include <iostream>
#include <cmath>
#include "omp.h"
#include "vector"

std::vector<std::vector<double>> inverse_L(std::vector<std::vector<double>> const& L){
    std::vector<std::vector<double>> res(L.size());
    for (int i = 0; i < L.size(); ++i) {
        res[i] = std::vector<double>(L.size());
        for (int j = 0; j < L.size(); ++j) {
            res[i][j] = 0;
        }
        res[i][i] = 1;
    }
    for (int i = 0; i < L.size()-1; ++i) {
        for (int j = i+1; j < L.size(); ++j) {
            for (int k = 0; k < L.size(); ++k) {
                res[j][k] = res[j][k] - res[i][k]*L[j][i];
            }
        }
    }
    return res;
}

std::vector<std::vector<double>> prod(std::vector<std::vector<double>> const& left,
                                      std::vector<std::vector<double>> const& right){
    std::vector<std::vector<double>> res(left.size());
    for (int i = 0; i < left.size(); ++i) {
        res[i] = std::vector<double>(right[0].size());
        for (int j = 0; j < right[0].size(); ++j) {
            res[i][j] = 0;
            for (int k = 0; k < right.size(); ++k) {
                res[i][j] += left[i][k]*right[k][j];
            }
        }
    }
    return res;
}

void LU(std::vector<std::vector<double>> &A){
    if (A.empty()){
        return;
    }
    if (A[0].empty()){
        return;
    }
    for (int i = 0; i < std::min(A.size()-1,A[0].size()); ++i) {
        for (int j = i+1; j < A.size(); ++j) {
            A[j][i] = A[j][i]/A[i][i];
        }
        if (i<A[0].size()){
            for (int j = i+1; j < A.size(); ++j) {
                for (int k = i+1; k < A[0].size(); ++k) {
                    A[j][k]=A[j][k]-A[j][i]*A[i][k];
                }
            }
        }
    }
}

void matrix_out(std::vector<std::vector<double>> const& A){
    if (A.empty()){
        return;
    }
    if (A[0].empty()){
        return;
    }
    for (const auto & i : A) {
        for (double j : i) {
            std::cout << j << "  ";
        }
        std::cout << std::endl;
    }
}

void LU_Blocks(std::vector<std::vector<double>> &A, int b){
    for (int i = 0; i < A[0].size()-1; i=i+b) {
        std::vector<std::vector<double>> subA(A[0].size()-i);
        for (int j = i; j < A[0].size(); ++j) {
            subA[j-i] = std::vector<double>(b);
            for (int k = i; k < i+b; ++k) {
                subA[j-i][k-i] = A[j][k];
            }
        }
        LU(subA);
        for (int j = i; j < A[0].size(); ++j) {
            for (int k = i; k < i+b; ++k) {
                A[j][k] = subA[j-i][k-i];
            }
        }
        if ((int) A[0].size()-i-b > 0) {
            std::vector<std::vector<double>> subL(b);
            for (int j = i; j < i+b; ++j) {
                subL[j-i] = std::vector<double>(b);
                subL[j-i][j-i] = 1;
                for (int k = j-i; k < b; ++k) {
                    subL[j-i][k] = 0;
                }
                for (int k = i; k < j-i; ++k) {
                    subL[j-i][k-i] = A[j][k];
                }
            }
            subL = inverse_L(subL);
            subA = std::vector<std::vector<double>>(b);
            for (int j = i; j < i+b; ++j) {
                subA[j - i] = std::vector<double>(A[0].size() - i - b);
                for (int k = i+b; k < A[0].size(); ++k) {
                    subA[j-i][k-i-b] = A[j][k];
                }
            }
            subA = prod(subL, subA);
            for (int j = i; j < i + b; ++j) {
                for (int k = i + b; k < A[0].size(); ++k) {
                    A[j][k] = subA[j - i][k - i - b];
                }
            }
            std::vector<std::vector<double>> subA1(A[0].size() - i - b);
            for (int j = i + b; j < A[0].size(); ++j) {
                subA1[j - i - b] = std::vector<double>(b);
                for (int k = i; k < i + b; ++k) {
                    subA1[j - i - b][k - i] = A[j][k];
                }
            }
            std::vector<std::vector<double>> subA2(b);
            for (int j = i; j < i + b; ++j) {
                subA2[j - i] = std::vector<double>(A[0].size() - i - b);
                for (int k = i + b; k < A[0].size(); ++k) {
                    subA2[j - i][k - i - b] = A[j][k];
                }
            }
            subA = prod(subA1, subA2);
            for (int j = i + b; j < A[0].size(); ++j) {
                for (int k = i + b; k < A[0].size(); ++k) {
                    A[j][k] = A[j][k] - subA[j - i - b][k - i - b];
                }
            }
        }
    }
}

int main() {
    srand(time(0));
    int n;
    int m;
    std::cin >> n >> m;
    std::vector<std::vector<double>> A(n);
    for (auto & i : A) {
        i = std::vector<double>(m);
        for (double & j : i) {
            j = (double)rand() /RAND_MAX;
        }
    }
    std::cout << "Исходная матрица:" << std::endl;
    matrix_out(A);
    std::vector<std::vector<double>> B = A;
    LU(B);
    std::cout << "После не блочного разложения" << std::endl;
    matrix_out(B);
    LU_Blocks(A,4);
    std::cout << "После блочного разложения" << std::endl;
    matrix_out(A);
    for (int i = 0; i < A.size(); ++i) {
        for (int j = 0; j < A[i].size(); ++j) {
            A[i][j] = A[i][j] - B[i][j];
        }
    }
    std::cout << "Разность" << std::endl;
    matrix_out(A);
    return 0;
}
