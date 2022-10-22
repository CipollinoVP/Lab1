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

template<typename matrix>
matrix prod_parallel(matrix const& left,
                     matrix const& right){
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

template<typename matrix>
void LU_parallel(matrix &A){
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
#pragma omp parallel for default(none) shared(A,i) collapse(2)
            for (int j = i+1; j < A.size(); ++j) {
                for (int k = i+1; k < A[0].size(); ++k) {
                    A[j][k]=A[j][k]-A[j][i]*A[i][k];
                }
            }
        }
    }
}

template<typename matrix>
void LU(matrix &A){
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

template<typename matrix>
void matrix_out(matrix const& A){
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

template<typename matrix>
std::vector<std::vector<double>> difference(matrix const& right,
                                            matrix const& left){
    std::vector<std::vector<double>> result(right.size());
    for (int i = 0; i < right.size(); ++i) {
        result[i] = std::vector<double>(right[0].size());
        for (int j = 0; j < right[0].size(); ++j) {
            result[i][j] = right[i][j]-left[i][j];
        }
    }
    return result;
}

template<typename matrix>
std::vector<std::vector<double>> difference_parallel(matrix const& right,
                                                     matrix const& left){
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

template<typename matrix>
void LU_Blocks(matrix &A, int b){
    for (int i = 0; i<A.size()-1; i+=b)
    {
        std::vector<std::vector<double>> subA(A[0].size()-i);
        for (int j = 0; j < A[0].size()-i; ++j) {
            subA[j] = std::vector<double>(b);
            for (int k = 0; k < b; ++k) {
                subA[j][k] = A[j+i][k+i];
            }
        }
        LU(subA);
        for (int j = 0; j < A[0].size()-i; ++j) {
            for (int k = 0; k < b; ++k) {
                A[j+i][k+i] = subA[j][k];
            }
        }
        if ((int) A[0].size()-i - b > 0) {
            std::vector<std::vector<double>> subL(b);
            for (int j = 0; j < b; ++j) {
                subL[j] = std::vector<double>(b);
                subL[j][j] = 1;
                for (int k = j; k < b; ++k) {
                    subL[j][k] = 0;
                }
                for (int k = 0; k < j; ++k) {
                    subL[j][k] = A[j+i][k+i];
                }
            }
            subL = inverse_L(subL);
            subA = std::vector<std::vector<double>>(b);
            for (int j = 0; j < b; ++j) {
                subA[j] = std::vector<double>(A[0].size()-i - b);
                for (int k = b; k < A[0].size()-i; ++k) {
                    subA[j][k - b] = A[j+i][k+i];
                }
            }
            subA = prod(subL, subA);
            for (int j = 0; j < b; ++j) {
                for (int k = b; k < A[0].size()-i; ++k) {
                    A[j+i][k+i] = subA[j][k - b];
                }
            }
            std::vector<std::vector<double>> subA1(A[0].size()-i - b);
            for (int j = b; j < A[0].size()-i; ++j) {
                subA1[j - b] = std::vector<double>(b);
                for (int k = 0; k < b; ++k) {
                    subA1[j - b][k] = A[j+i][k+i];
                }
            }
            std::vector<std::vector<double>> subA2(b);
            for (int j = 0; j < b; ++j) {
                subA2[j] = std::vector<double>(A[0].size()-i - b);
                for (int k = b; k < A[0].size()-i; ++k) {
                    subA2[j][k - b] = A[j+i][k+i];
                }
            }
            subA = prod(subA1, subA2);
            for (int j = b; j < A[0].size()-i; ++j) {
                for (int k = b; k < A[0].size()-i; ++k) {
                    A[j+i][k+i] = A[j+i][i+k] - subA[j - b][k - b];
                }
            }
        }
    }
}

template<typename matrix>
void LU_Blocks_parallel(matrix &A, int b) {
    for (int i = 0; i < A.size() - 1; i += b) {
        std::vector<std::vector<double>> subA(A[0].size() - i);
        for (int j = 0; j < A[0].size() - i; ++j) {
            subA[j] = std::vector<double>(b);
            for (int k = 0; k < b; ++k) {
                subA[j][k] = A[j+i][k+i];
            }
        }
        LU_parallel(subA);
        for (int j = 0; j < A[0].size() - i; ++j) {
            for (int k = 0; k < b; ++k) {
                A[j+i][k+i] = subA[j][k];
            }
        }
        if ((int) A[0].size() - i - b > 0) {
            std::vector<std::vector<double>> subL(b);
            for (int j = 0; j < b; ++j) {
                subL[j] = std::vector<double>(b);
                subL[j][j] = 1;
                for (int k = j; k < b; ++k) {
                    subL[j][k] = 0;
                }
                for (int k = 0; k < j; ++k) {
                    subL[j][k] = A[j+i][k+i];
                }
            }
            subL = inverse_L_parallel(subL);
            subA = std::vector<std::vector<double>>(b);
            for (int j = 0; j < b; ++j) {
                subA[j] = std::vector<double>(A[0].size() - i - b);
                for (int k = b; k < A[0].size() - i; ++k) {
                    subA[j][k - b] = A[j+i][k+i];
                }
            }
            subA = prod_parallel(subL, subA);
            for (int j = 0; j < b; ++j) {
                for (int k = b; k < A[0].size() - i; ++k) {
                    A[j+i][k+i] = subA[j][k - b];
                }
            }
            std::vector<std::vector<double>> subA1(A[0].size() - i - b);
            for (int j = b; j < A[0].size() - i; ++j) {
                subA1[j - b] = std::vector<double>(b);
                for (int k = 0; k < b; ++k) {
                    subA1[j - b][k] = A[j+i][k+i];
                }
            }
            std::vector<std::vector<double>> subA2(b);
            for (int j = 0; j < b; ++j) {
                subA2[j] = std::vector<double>(A[0].size() - i - b);
                for (int k = b; k < A[0].size() - i; ++k) {
                    subA2[j][k - b] = A[j+i][k+i];
                }
            }
            subA = prod_parallel(subA1, subA2);
            for (int j = b; j < A[0].size() - i; ++j) {
                for (int k = b; k < A[0].size() - i; ++k) {
                    A[j+i][k+i] = A[j+i][k+i] - subA[j - b][k - b];
                }
            }
        }
    }
}

int main() {
    omp_set_dynamic(0);
    omp_set_num_threads(8);
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
    LU_Blocks(B3,64);
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
    LU_Blocks_parallel(B4,64);
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