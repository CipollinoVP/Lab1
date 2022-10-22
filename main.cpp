#include <iostream>
#include <cmath>
#include "omp.h"
#include "vector"
#include "ctime"

class sub_vector{
private:
    std::vector<double> *data;
    unsigned long n0;
    unsigned long nf;
public:
    sub_vector(std::vector<double> &A, unsigned long n){
        data = &A;
        n0 = n;
        nf = A.size()-1;
    }
    sub_vector(sub_vector &A, unsigned long n){
        data = A.data;
        n0 = A.n0 + n;
        nf = A.nf;
    }
    sub_vector(std::vector<double> &A, unsigned long n, unsigned long f){
        data = &A;
        n0 = n;
        nf = f;
    }
    sub_vector(sub_vector &A, unsigned long n, unsigned long f){
        data = A.data;
        n0 = A.n0 + n;
        nf = A.n0 + f;
    }
    [[nodiscard]] unsigned long size() const{
        return nf-n0+1;
    }
    double& operator[](unsigned long i){
        return data->operator[](i+n0);
    }
    [[nodiscard]] bool empty() const{
        if ((int) nf - n0 < 0){
            return true;
        } else {
            return false;
        }
    }
};

class sub_matrix{
private:
    std::vector<std::vector<double>> *data;
    unsigned long n0;
    unsigned long m0;
    unsigned long nf;
    unsigned long mf;
public:
    auto operator[](unsigned int i){
        return sub_vector(data->operator[](n0+i),m0,mf);
    }
    sub_matrix(std::vector<std::vector<double>> &A, unsigned long n, unsigned long m){
        data = &A;
        n0 = n;
        m0 = m;
        nf = A.size()-1;
        mf = A[0].size() - 1;
    }
    sub_matrix(sub_matrix &A, unsigned long n, unsigned long m){
        data = A.data;
        n0 = A.n0 + n;
        m0 = A.m0 + m;
        nf = A.nf;
        mf = A.mf;
    }
    sub_matrix(std::vector<std::vector<double>> &A, unsigned long n, unsigned long m, unsigned long fn, unsigned long fm){
        data = &A;
        n0 = n;
        m0 = m;
        nf = fn;
        mf = fm;
    }
    sub_matrix(sub_matrix &A, unsigned long n, unsigned long m, unsigned long fn, unsigned long fm){
        data = A.data;
        n0 = A.n0 + n;
        m0 = A.m0 + m;
        nf = A.n0 + fn;
        mf = A.m0 + fm;
    }
    [[nodiscard]] unsigned long size() const{
        return nf-n0+1;
    }
    [[nodiscard]] bool empty() const{
        if ((int) nf - n0 < 0){
            return true;
        } else {
            return false;
        }
    }
};

template<typename matrix>
std::vector<std::vector<double>> inverse_L(matrix& L){
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

template<typename matrix>
std::vector<std::vector<double>> inverse_L_parallel(matrix& L){
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

template<typename matrix1,typename matrix2>
std::vector<std::vector<double>> prod(matrix1& left,
                                      matrix2& right){
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

template<typename matrix1,typename matrix2>
std::vector<std::vector<double>> prod_parallel(matrix1& left,
                     matrix2& right){
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
#pragma omp parallel for default(none) shared(A,i)
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
        sub_matrix subA(A,0,0,A[0].size()-1,b-1);
        LU(subA);
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
            subA = sub_matrix(A,0,b,b-1,A[0].size()-1);
            std::vector<std::vector<double>> subAA = prod(subL, subA);
            for (int j = 0; j < b; ++j) {
                for (int k = b; k < A[0].size(); ++k) {
                    A[j][k] = subAA[j][k - b];
                }
            }

            sub_matrix subA1(A,b,0,A[0].size()-1,b-1);
            sub_matrix subA2(A,0,b,b-1,A[0].size()-1);
            subAA = prod(subA1,subA2);
            for (int j = b; j < A[0].size(); ++j) {
                for (int k = b; k < A[0].size(); ++k) {
                    A[j][k] = A[j][k] - subAA[j - b][k - b];
                }
            }
            sub_matrix nS(A,b,b);
            LU_Blocks(nS,b);
        }
}

template<typename matrix>
void LU_Blocks_parallel(matrix &A, int b){
    sub_matrix subA(A,0,0,A[0].size()-1,b-1);
    LU_parallel(subA);
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
        subA = sub_matrix(A,0,b,b-1,A[0].size()-1);
        std::vector<std::vector<double>> subAA = prod_parallel(subL, subA);
#pragma omp parallel for default(none) shared(A,b,subAA)
        for (int j = 0; j < b; ++j) {
            for (int k = b; k < A[0].size(); ++k) {
                A[j][k] = subAA[j][k - b];
            }
        }
        sub_matrix subA1(A,b,0,A[0].size()-1,b-1);
        sub_matrix subA2(A,0,b,b-1,A[0].size()-1);
        subAA = prod_parallel(subA1,subA2);
#pragma omp parallel for default(none) shared(A,b,subAA)
        for (int j = b; j < A[0].size(); ++j) {
            for (int k = b; k < A[0].size(); ++k) {
                A[j][k] = A[j][k] - subAA[j - b][k - b];
            }
        }
        sub_matrix nS(A,b,b);
        LU_Blocks_parallel(nS,b);
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
    LU_Blocks(B3,32);
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
    LU_Blocks_parallel(B4,32);
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
