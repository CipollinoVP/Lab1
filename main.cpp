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
#pragma omp parallel for default(none) shared(A,b,subA)
    for (int j = 0; j < A[0].size(); ++j) {
        subA[j] = std::vector<double>(b);
        for (int k = 0; k < b; ++k) {
            subA[j][k] = A[j][k];
        }
    }
    LU_parallel(subA);
#pragma omp parallel for default(none) shared(A,b,subA)
    for (int j = 0; j < A[0].size(); ++j) {
        for (int k = 0; k < b; ++k) {
            A[j][k] = subA[j][k];
        }
    }
    if ((int) A[0].size()-b > 0) {
        std::vector<std::vector<double>> subL(b);
#pragma omp parallel for default(none) shared(A,b,subL)
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
#pragma omp parallel for default(none) shared(A,b,subA)
        for (int j = 0; j < b; ++j) {
            subA[j] = std::vector<double>(A[0].size() - b);
            for (int k = b; k < A[0].size(); ++k) {
                subA[j][k-b] = A[j][k];
            }
        }
        subA = prod_parallel(subL, subA);
#pragma omp parallel for default(none) shared(A,b,subA)
        for (int j = 0; j < b; ++j) {
            for (int k = b; k < A[0].size(); ++k) {
                A[j][k] = subA[j][k - b];
            }
        }
        std::vector<std::vector<double>> subA1(A[0].size() - b);
#pragma omp parallel for default(none) shared(A,b,subA1)
        for (int j = b; j < A[0].size(); ++j) {
            subA1[j - b] = std::vector<double>(b);
            for (int k = 0; k < b; ++k) {
                subA1[j - b][k] = A[j][k];
            }
        }
        std::vector<std::vector<double>> subA2(b);
#pragma omp parallel for default(none) shared(A,b,subA2)
        for (int j = 0; j < b; ++j) {
            subA2[j] = std::vector<double>(A[0].size() - b);
            for (int k = b; k < A[0].size(); ++k) {
                subA2[j][k - b] = A[j][k];
            }
        }
        subA = prod_parallel(subA1,subA2);
#pragma omp parallel for default(none) shared(A,b,subA)
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
#pragma omp parallel for default(none) shared(A,b,subA)
        for (int i = 0; i < A.size()-b; ++i) {
            for (int j = 0; j < A[0].size()-b; ++j) {
                A[i+b][j+b] = subA[i][j];
            }
        }
    }
}

int main() {
    omp_set_dynamic(0);
    omp_set_num_threads(5);
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
    std::vector<std::vector<double>> B1(A);
    long int t1 = clock();
    LU(B1);
    long int t2 = clock();
    double time1 = (double) (t2-t1)/CLOCKS_PER_SEC;
    std::vector<std::vector<double>> B2 = A;
    t1 = clock();
    LU_parallel(B2);
    t2 = clock();
    double err1 = 0;
    for (int i = 0; i < B2.size(); ++i) {
        for (int j = 0; j < B2[0].size(); ++j) {
            if (err1 < std::abs(B1[i][j]-B2[i][j])) {
                err1 = std::abs(B1[i][j]-B2[i][j]);
            }
        }
    }
    double time2 = (double) (t2-t1)/CLOCKS_PER_SEC;
    std::vector<std::vector<double>> B3 = A;
    t1 = clock();
    LU_Blocks(B3,16);
    t2 = clock();
    double err2 = 0;
    for (int i = 0; i < B3.size(); ++i) {
        for (int j = 0; j < B3[0].size(); ++j) {
            if (err2 < std::abs(B1[i][j]-B3[i][j])) {
                err2 = std::abs(B1[i][j]-B3[i][j]);
            }
        }
    }
    double time3 = (double) (t2-t1)/CLOCKS_PER_SEC;
    std::vector<std::vector<double>> B4 = A;
    t1 = clock();
    LU_Blocks_parallel(B4,16);
    t2 = clock();
    double err3 = 0;
    for (int i = 0; i < B4.size(); ++i) {
        for (int j = 0; j < B4[0].size(); ++j) {
            if (err3 < std::abs(B1[i][j]-B4[i][j])) {
                err3 = std::abs(B1[i][j]-B4[i][j]);
            }
        }
    }
    double time4 = (double) (t2-t1)/CLOCKS_PER_SEC;
    std::cout << "Неблочное LU-разложение без распараллеливания" << std::endl << "Время: " <<
    time1 << std::endl <<"Неблочное LU-разложение с распараллеливанием" << std::endl << "Время " << time2 <<
    "  Ошибка в сравнении с первыи разложением: " << err1 << std::endl
    << "Ускорение " << time1/time2 << std::endl
    << "Блочное LU-разложение без распараллеливания"<< std::endl << "Время: " << time3 << "  Ошибка в сравнении с первыи разложением: "
    << err2 << std::endl
    << "Блочное LU-разложение с распараллеливанием" << std::endl << "Время: " << time4
    << "  Ошибка в сравнении с первыи разложением: " << err3 << std::endl << "Ускорение " << time3/time4 << std::endl;
    return 0;
}
