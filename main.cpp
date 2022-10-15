#include <iostream>
#include <cmath>
#include "omp.h"
#include "vector"
#include "ctime"

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

std::vector<std::vector<double>> inverse_L_parallel(std::vector<std::vector<double>> const& L){
    std::vector<std::vector<double>> res(L.size());
    for (int i = 0; i < L.size(); ++i) {
        res[i] = std::vector<double>(L.size());
        for (int j = 0; j < L.size(); ++j) {
            res[i][j] = 0;
        }
        res[i][i] = 1;
    }
    for (int i = 0; i < L.size()-1; ++i) {
#pragma omp parallel for default(none) shared(res,i,L)
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
                res[i][j] = res[i][j] + left[i][k]*right[k][j];
            }
        }
    }
    return res;
}

std::vector<std::vector<double>> prod_parallel(std::vector<std::vector<double>> const& left,
                                      std::vector<std::vector<double>> const& right){
    std::vector<std::vector<double>> res(left.size());
    for (int i = 0; i < left.size(); ++i) {
        res[i] = std::vector<double>(right[0].size());
#pragma omp parallel for default(none) shared(res,i,left,right)
        for (int j = 0; j < right[0].size(); ++j) {
            res[i][j] = 0;
            for (int k = 0; k < right.size(); ++k) {
                res[i][j] = res[i][j] + left[i][k]*right[k][j];
            }
        }
    }
    return res;
}

void LU_parallel(std::vector<std::vector<double>> &A){
    if (A.empty()){
        return;
    }
    if (A[0].empty()){
        return;
    }
    for (int i = 0; i < std::min(A.size()-1,A[0].size()); ++i) {
#pragma omp parallel for default(none) shared(A,i)
        for (int j = i+1; j < A.size(); ++j) {
            A[j][i] = A[j][i]/A[i][i];
        }
        if (i<A[0].size()){
#pragma omp parallel for default(none) shared(A,i)
            for (int j = i+1; j < A.size(); ++j) {
                for (int k = i+1; k < A[0].size(); ++k) {
                    A[j][k]=A[j][k]-A[j][i]*A[i][k];
                }
            }
        }
    }
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

std::vector<std::vector<double>> difference(std::vector<std::vector<double>> const& right,
                                            std::vector<std::vector<double>> const& left){
    std::vector<std::vector<double>> result(right.size());
    for (int i = 0; i < right.size(); ++i) {
        result[i] = std::vector<double>(right[0].size());
        for (int j = 0; j < right[0].size(); ++j) {
            result[i][j] = right[i][j]-left[i][j];
        }
    }
    return result;
}

std::vector<std::vector<double>> difference_parallel(std::vector<std::vector<double>> const& right,
                                            std::vector<std::vector<double>> const& left){
    std::vector<std::vector<double>> result(right.size());
#pragma omp parallel for default(none) shared(result,right,left)
    for (int i = 0; i < right.size(); ++i) {
        result[i] = std::vector<double>(right[0].size());
        for (int j = 0; j < right[0].size(); ++j) {
            result[i][j] = right[i][j]-left[i][j];
        }
    }
    return result;
}

void LU_Blocks(std::vector<std::vector<double>> &A, int b){
        std::vector<std::vector<double>> subA(A[0].size());
        for (int j = 0; j < A[0].size(); ++j) {
            subA[j] = std::vector<double>(b);
            for (int k = 0; k < b; ++k) {
                subA[j][k] = A[j][k];
            }
        }
        LU(subA);
        for (int j = 0; j < A[0].size(); ++j) {
            for (int k = 0; k < b; ++k) {
                A[j][k] = subA[j][k];
            }
        }
        if ((int) A[0].size()-b > 0) {
            std::vector<std::vector<double>> subL(b);
            for (int j = 0; j < b; ++j) {
                subL[j] = std::vector<double>(b);
                subL[j][j] = 1;
                for (int k = j; k < b; ++k) {
                    subL[j][k] = 0;
                }
                for (int k = 0; k < j; ++k) {
                    subL[j][k] = A[j][k];
                }
            }
            subL = inverse_L(subL);
            subA = std::vector<std::vector<double>>(b);
            for (int j = 0; j < b; ++j) {
                subA[j] = std::vector<double>(A[0].size() - b);
                for (int k = b; k < A[0].size(); ++k) {
                    subA[j][k-b] = A[j][k];
                }
            }
            subA = prod(subL, subA);
            for (int j = 0; j < b; ++j) {
                for (int k = b; k < A[0].size(); ++k) {
                    A[j][k] = subA[j][k - b];
                }
            }
            std::vector<std::vector<double>> subA1(A[0].size() - b);
            for (int j = b; j < A[0].size(); ++j) {
                subA1[j - b] = std::vector<double>(b);
                for (int k = 0; k < b; ++k) {
                    subA1[j - b][k] = A[j][k];
                }
            }
            std::vector<std::vector<double>> subA2(b);
            for (int j = 0; j < b; ++j) {
                subA2[j] = std::vector<double>(A[0].size() - b);
                for (int k = b; k < A[0].size(); ++k) {
                    subA2[j][k - b] = A[j][k];
                }
            }
            subA = prod(subA1,subA2);
            for (int j = b; j < A[0].size(); ++j) {
                for (int k = b; k < A[0].size(); ++k) {
                    A[j][k] = A[j][k] - subA[j - b][k - b];
                }
            }
            subA = std::vector<std::vector<double>>(A.size()-b);
            for (int i = 0; i < A.size()-b; ++i) {
                subA[i] = std::vector<double>(A[0].size()-b);
                for (int j = 0; j < A[0].size()-b; ++j) {
                    subA[i][j] = A[i+b][j+b];
                }
            }
            LU_Blocks(subA,b);
            for (int i = 0; i < A.size()-b; ++i) {
                for (int j = 0; j < A[0].size()-b; ++j) {
                    A[i+b][j+b] = subA[i][j];
                }
            }
        }
}

void LU_Blocks_parallel(std::vector<std::vector<double>> &A, int b){
    std::vector<std::vector<double>> subA(A[0].size());
    for (int j = 0; j < A[0].size(); ++j) {
        subA[j] = std::vector<double>(b);
        for (int k = 0; k < b; ++k) {
            subA[j][k] = A[j][k];
        }
    }
    LU_parallel(subA);
    for (int j = 0; j < A[0].size(); ++j) {
        for (int k = 0; k < b; ++k) {
            A[j][k] = subA[j][k];
        }
    }
    if ((int) A[0].size()-b > 0) {
        std::vector<std::vector<double>> subL(b);
        for (int j = 0; j < b; ++j) {
            subL[j] = std::vector<double>(b);
            subL[j][j] = 1;
            for (int k = j; k < b; ++k) {
                subL[j][k] = 0;
            }
            for (int k = 0; k < j; ++k) {
                subL[j][k] = A[j][k];
            }
        }
        subL = inverse_L_parallel(subL);
        subA = std::vector<std::vector<double>>(b);
        for (int j = 0; j < b; ++j) {
            subA[j] = std::vector<double>(A[0].size() - b);
            for (int k = b; k < A[0].size(); ++k) {
                subA[j][k-b] = A[j][k];
            }
        }
        subA = prod_parallel(subL, subA);
        for (int j = 0; j < b; ++j) {
            for (int k = b; k < A[0].size(); ++k) {
                A[j][k] = subA[j][k - b];
            }
        }
        std::vector<std::vector<double>> subA1(A[0].size() - b);
        for (int j = b; j < A[0].size(); ++j) {
            subA1[j - b] = std::vector<double>(b);
            for (int k = 0; k < b; ++k) {
                subA1[j - b][k] = A[j][k];
            }
        }
        std::vector<std::vector<double>> subA2(b);
        for (int j = 0; j < b; ++j) {
            subA2[j] = std::vector<double>(A[0].size() - b);
            for (int k = b; k < A[0].size(); ++k) {
                subA2[j][k - b] = A[j][k];
            }
        }
        subA = prod_parallel(subA1,subA2);
        for (int j = b; j < A[0].size(); ++j) {
            for (int k = b; k < A[0].size(); ++k) {
                A[j][k] = A[j][k] - subA[j - b][k - b];
            }
        }
        subA = std::vector<std::vector<double>>(A.size()-b);
#pragma omp parallel for default(none) shared(A,b,subA)
        for (int i = 0; i < A.size()-b; ++i) {
            subA[i] = std::vector<double>(A[0].size()-b);
            for (int j = 0; j < A[0].size()-b; ++j) {
                subA[i][j] = A[i+b][j+b];
            }
        }
        LU_Blocks_parallel(subA,b);
        for (int i = 0; i < A.size()-b; ++i) {
            for (int j = 0; j < A[0].size()-b; ++j) {
                A[i+b][j+b] = subA[i][j];
            }
        }
    }
}

int main() {
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    int a = 5;
    srand(time(0));
    int n;
    int m;
    std::cin >> n;
    m = n;
    std::vector<std::vector<double>> A(n);
    for (auto & i : A) {
        i = std::vector<double>(m);
        for (double & j : i) {
            j = (double)rand() /RAND_MAX;
        }
    }
    std::vector<std::vector<double>> B(A);
    long int t1 = clock();
    LU(B);
    long int t2 = clock();
    double time1 = (double) (t2-t1)/CLOCKS_PER_SEC;
    B = A;
    t1 = clock();
    LU_parallel(B);
    t2 = clock();
    double time2 = (double) (t2-t1)/CLOCKS_PER_SEC;
    B = A;
    t1 = clock();
    LU_Blocks(B,32);
    t2 = clock();
    double time3 = (double) (t2-t1)/CLOCKS_PER_SEC;
    B = A;
    t1 = clock();
    LU_Blocks_parallel(B,32);
    t2 = clock();
    double time4 = (double) (t2-t1)/CLOCKS_PER_SEC;
    std::cout << time1 << "   " << time2 << "  " << time3 << "   " << time4 << std::endl;
    return 0;
}
