//   ______  _______   _______   ______  __
//  /      ||       \ |   ____| /      ||  |
// |  ,----'|  .--.  ||  |__   |  ,----'|  |
// |  |     |  |  |  ||   __|  |  |     |  |
// |  `----.|  '--'  ||  |     |  `----.|  |
//  \______||_______/ |__|      \______||__|
//
// Coordinate Descent Full Configuration Interaction (CDFCI) package in C++14
// https://github.com/quan-tum/CDFCI
//
// Copyright (c) 2019, Zhe Wang, Yingzhou Li and Jianfeng Lu
// All rights reserved.
//
// This source code is licensed under the BSD 3-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef CDFCI_SOLVER_H
#define CDFCI_SOLVER_H 1

#include <quadmath.h>

#include <chrono>
#include <iomanip>
#include <string>

#include "hamiltonian.h"
#include "lib/robin_hood/robin_hood.h"
#include "wavefunction.h"

// DSYEVR: computes selected eigenvalues and eigenvectors of a real symmetric
// matrix
// DGEEV: computes for an N-by-N real nonsymmetric matrix A, the
// eigenvalues and, optionally, the left and/or right eigenvectors.
// DPOTRF: Cholesky decomposition
// DGETRF: LU decomposition with pivoting
extern "C" {
extern double ddot_(int *N, double *X, int *INCX, double *Y, int *INCY);
extern double dnrm2_(int *N, double *X, int *INCX);
extern double dcopy_(int *N, double *X, int *INCX, double *Y, int *INCY);
extern int dsymv_(char *UPLO, int *N, double *ALPHA, double *A, int *LDA,
                  double *X, int *INCX, double *BETA, double *Y, int *INCY);
extern int dsyevr_(char *JOBZ, char *RANGE, char *UPLO, int *N, double *A,
                   int *LDA, double *VL, double *VU, int *IL, int *IU,
                   double *ABSTOL, int *M, double *W, double *Z, int *LDZ,
                   int *ISUPPZ, double *WORK, int *LWORK, int *IWORK,
                   int *LIWORK, int *INFO);
extern int dgeev_(char *, char *, int *, double *, int *, double *, double *,
                  double *, int *, double *, int *, double *, int *, int *);
extern int dpotrf_(char *UPLO, int *N, double *A, int *LDA, int *INFO);
extern int dgetrf_(int *M, int *N, double *A, int *LDA, int *IPIV, int *INFO);
}

void print_min_max(double *evec, int n, std::string s) {
  double max_evec = 0.0;
  int max_i;
  double min_evec = 1.0;
  int min_i;
  for (int i = 0; i < n; i++) {
    if (max_evec < fabs(evec[i])) {
      max_evec = fabs(evec[i]);
      max_i = i;
    }
    if (min_evec > fabs(evec[i])) {
      min_evec = fabs(evec[i]);
      min_i = i;
    }
  }
  std::cout << s;
  printf(": v[0] = %e, max v[%d] = %e, min v[%d] = %e\n", evec[0], max_i,
         max_evec, min_i, min_evec);
}

double alert_max(double *evec, int n, std::string s, double threshold) {
  double max_evec = 0.0;
  int max_i;
  for (int i = 0; i < n; i++) {
    if (max_evec < fabs(evec[i])) {
      max_evec = fabs(evec[i]);
      max_i = i;
    }
  }
  if (max_evec > threshold) {
    std::cout << s;
    printf(": max v[%d] = %e\n", max_i, max_evec);
  }
  return max_evec;
}

/**
 * @brief Coordinate Pick Functor
 *
 * @tparam N
 */
template <int N = 1>
class CoordinatePick {
 public:
  virtual WaveFunctionVector<N> operator()(WaveFunctionVector<N> &det_list,
                                           int k) const = 0;

  virtual ~CoordinatePick(){};
};

/* CoordinatePickGcdGrad: Pick Coordinate by Gradients */

template <int N = 1>
class CoordinatePickGcdGrad : public CoordinatePick<N> {
 public:
  ~CoordinatePickGcdGrad(){};

  /**
   * @brief Pick k coordinates according to their gradients.
   *
   * @param det_list The candidate list from last iteration.
   * @param k Number of coordinates to pick.
   * @return WaveFunctionVector<N> The picked out coordinates, along with
   *                               their corresponding x and z.
   */
  WaveFunctionVector<N> operator()(WaveFunctionVector<N> &det_list,
                                   int k) const {
    // LOG_WAVEFUNCTIONVECTOR("det_list", det_list.data);
    WaveFunctionVector<N> sub_det_list;
    sub_det_list.data.resize(k);

    QUAD_PRECISION s = det_list.xx() * det_list.scale() * det_list.scale();

    // listSortAndPop(det_list, sub_det_list, k, s);
    priorityQueuePop(det_list, sub_det_list, k, s);  // more efficient

    /*
        // sort the picked x, to facilitate extract subH
        const auto detComp = [](auto &a, auto &b) {
          int i;
          for (i = 0; i < N - 1 && a.first.repr[i] == b.first.repr[i]; i++)
            ;
          return (a.first.repr[i] < b.first.repr[i]);
        };
        std::sort(sub_det_list.data.begin(), sub_det_list.data.end(), detComp);
    */

    // sub_det_list.printXZ();
    LOG_WAVEFUNCTIONVECTOR("sub_det_list", sub_det_list.data);
    return sub_det_list;
  }

  /**
   * @brief Pick k det's with largest gradients from det_list (contains
   *        duplicates) and store them in sub_det_list.
   *
   * This implementation first sorts the list, then adds the determinants
   * one by one, if the current adding one is different from previous one.
   *
   * @param det_list from list
   * @param sub_det_list to list
   * @param k number of det's
   * @param s grad = fabs(x * s + z)
   */
  void listSortAndPop(WaveFunctionVector<N> &det_list,
                      WaveFunctionVector<N> &sub_det_list, int k,
                      double s) const {
    struct valComp {
      double s;
      explicit valComp(double scale) : s(scale) {}
      bool operator()(typename WaveFunctionVector<N>::value_type &a,
                      typename WaveFunctionVector<N>::value_type &b) {
        if (fabs(a.second[0] * s + a.second[1]) ==
            fabs(b.second[0] * s + b.second[1]))
          return detComp_global(a.first, b.first) > 0;
        return (fabs(a.second[0] * s + a.second[1]) >
                fabs(b.second[0] * s + b.second[1]));
      }
    };
    std::sort(det_list.data.begin(), det_list.data.end(), valComp(s));

    sub_det_list.data[0] = det_list.data[0];
    DeterminantEqual<N> isEqual;
    for (int i = 1, j = 1; j < k; i++)
      if (!isEqual(det_list.data[i].first, sub_det_list.data[j - 1].first))
        sub_det_list.data[j++] = det_list.data[i];
  }

  /**
   * @brief Pick k det's with largest gradients from det_list (contains
   *        duplicates) and store them in sub_det_list.
   *
   * This implementation first pushes everything into priority queue,
   * then pops out the determinants one by one. Add it into sub_det_list,
   * if the current adding one is different from previous one.
   *
   * priorityQueuePop achieves up to 3x speedup than listSortAndPop.
   *
   * @param det_list from list
   * @param sub_det_list to list
   * @param k number of det's
   * @param s grad = fabs(x * s + z)
   */
  void priorityQueuePop(WaveFunctionVector<N> &det_list,
                        WaveFunctionVector<N> &sub_det_list, int k,
                        QUAD_PRECISION s) const {
    using vt = typename WaveFunctionVector<N>::value_type;
    using T1 = std::pair<vt *, double>;
    auto cmpT1 = [](T1 a, T1 b) {
      if (a.second == b.second)
        return detComp_global((a.first)->first, (b.first)->first) < 0;
      return a.second < b.second;
    };  // the largest on the top
    std::priority_queue<T1, std::vector<T1>, decltype(cmpT1)> pq(cmpT1);

    // push everything into priority queue
    for (int i = 0; i < det_list.data.size(); i++) {
      auto &a = det_list.data[i];
      pq.push({&a, fabs(a.second[0] * s + a.second[1])});
    }

    // pop out k different det's
    sub_det_list.data[0] = *(pq.top().first);
    pq.pop();
    DeterminantEqual<N> isEqual;
    for (int i = 1, j = 1; j < k; i++) {
      auto elem = pq.top().first;
      pq.pop();
      if (!isEqual(elem->first, sub_det_list.data[j - 1].first))
        sub_det_list.data[j++] = *elem;
    }
  }
};

/* ----------------------------------------------------*/

/* Coordinate Update functors*/
template <int N = 1>
class CoordinateUpdate {
 public:
  virtual std::vector<double> operator()(
      double *H_sub, WaveFunction<N> &xz,
      WaveFunctionVector<N> &sub_det_list) = 0;

  virtual ~CoordinateUpdate(){};

  /** some log informations */
  int num_iterations = 0;
  double max_residual = 0.0;
  double curr_max_abs_evec = 0.0;
  int sum_iterations = 0;
  int sum_inv_iter = 0;
  int curr_inv_iter = 0;
  double sum_diff_inv_iter = 0.0;
  double max_diff_inv_iter = 0.0;
  double curr_diff_inv_iter = 0.0;
  Timer eig_timer = Timer(5);
};

/* CoordinateUpdateLS: Update coordinate by Line Search */

template <int N = 1>
class CoordinateUpdateLS : public CoordinateUpdate<N> {
 public:
  ~CoordinateUpdateLS(){};

  std::vector<double> operator()(double *H_sub, WaveFunction<N> &xz,
                                 WaveFunctionVector<N> &sub_det_list) {
    double dx = 0;
    auto det_picked = sub_det_list.data[0];
    auto det = det_picked.first;
    auto x = det_picked.second[0];
    auto z = det_picked.second[1];
    auto xx = xz.xx();

    // Diagonal dA
    double dA = -H_sub[0];

    // Line Search
    auto p1 = xx - x * x - dA;
    auto q = z + dA * x;  //
    auto p3 = p1 / 3;
    auto q2 = q / 2;
    auto d = p3 * p3 * p3 + q2 * q2;
    double rt = 0;

    const double pi = atan(1.0) * 4;

    if (d >= 0) {
      auto qrtd = sqrt(d);
      rt = cbrt(-q2 + qrtd) + cbrt(-q2 - qrtd);
    } else {
      auto qrtd = sqrt(-d);
      if (q2 >= 0) {
        rt = 2 * sqrt(-p3) * cos((atan2(-qrtd, -q2) - 2 * pi) / 3.0);
      } else {
        rt = 2 * sqrt(-p3) * cos(atan2(qrtd, -q2) / 3.0);
      }
    }

    dx = rt - x;

    // Newton iteration to improve accuracy
    auto dxn = dx - (dx * (dx * (dx + 3 * x) + (3 * x * x + p1)) + p1 * x + q +
                     x * x * x) /
                        (dx * (3 * dx + 6 * x) + 3 * x * x + p1);

    const double depsilon = 1e-12;
    const int max_iter = 10;
    int iter = 0;
    while (fabs((dxn - dx) / dx) > depsilon && iter < max_iter) {
      dx = dxn;
      dxn = dx - (dx * (dx * (dx + 3 * x) + (3 * x * x + p1)) + p1 * x + q +
                  x * x * x) /
                     (dx * (3 * dx + 6 * x) + 3 * x * x + p1);
      ++iter;
    }

    std::vector<double> res = {dx};
    return res;
  }
};

/* CoordinateUpdateEig: Update coordinate by solving eigenvalue problem */

template <int N = 1>
class CoordinateUpdateEig : public CoordinateUpdate<N> {
 public:
  ~CoordinateUpdateEig(){};

  /**
   * @brief Update coordinates by solving eigenvalue problem.
   *
   * @param H_sub submatrix of Hamiltonian
   * @param xz
   * @param sub_det_list
   * @return std::vector<double> the change dx of each picked coordinates.
   *         Its size = k.
   *         The ordering is the same with sub_det_list.
   */
  std::vector<double> operator()(double *H_sub, WaveFunction<N> &xz,
                                 WaveFunctionVector<N> &sub_det_list) {
    double dx = 0.0;
    QUAD_PRECISION scale = 1.0, new_xx, new_xz;
    char Vchar = 'V', Ichar = 'I', Lchar = 'L', Nchar = 'N';
    double zero = 0.0, one_d = 1.0;
    int index = 1;
    int one = 1, null = 0;

    // k is the dimension of H_sub and n is the dimension of submatrix
    int k = sub_det_list.data.size(), n;

    // Calculate x'*x, x'*z, H_sub*x, x'*H_sub*x
    double *x = new double[k], *z = new double[k], *Hx = new double[k];
    for (int i = 0; i < k; i++) {
      auto &det_picked = sub_det_list.data[i];
      x[i] = det_picked.second[0];
      z[i] = det_picked.second[1];
    }

    double xTx = ddot_(&k, x, &one, x, &one);
    double xTz = ddot_(&k, x, &one, z, &one);
    dsymv_(&Lchar, &k, &one_d, H_sub, &k, x, &one, &zero, Hx, &one);
    double xTHx = ddot_(&k, x, &one, Hx, &one);

    double yy = xz.xx() - xTx;
    double y1 = sqrt(yy);
    LOG_NUMBER("y1", y1);

    std::vector<double> res(k);

    // Construct the eigenvalue problem
    if (fabs(yy) > FP_EPSILON) {
      this->eig_timer.start();
      n = k + 1;
      int nn = n * n;
      double *data = new double[n * n];
      double *data_db = new double[n * n]; // back up because lapack change the value of data

      /**
       * @brief Compute largest eigenvalue and the corresponding eigenvector
       * of [alpha, b'; b, H_sub], where
       *     alpha = (xz - 2*x[I]'*z[I] + x[I]'*H_sub*x[I])/yy;
       *     b = (z[I] - H_sub*x[I])/y1;
       */
      data[0] = (xz.xz() - 2 * xTz + xTHx) / yy;
      for (int i = 0; i < k; i++)
        data[i + 1] = (z[i] - Hx[i]) / y1;
      for (int i = 0; i < k; i++)
        for (int j = i; j < k; j++)
          data[(i + 1) * n + j + 1] = H_sub[i * k + j];
      dcopy_(&nn, data, &one, data_db, &one);

      LOG_MATRIX("Subproblem", data, n);

      double *eval = new double[n];
      double *evec = new double[n];

      int lwork = -1;
      double wkopt;
      int liwork = -1;
      int iwkopt;
      int info;

      this->eig_timer.lap(0);
      // calculate eigenvalues using the DSYEVR subroutine
      // query the size of work and iwork
      dsyevr_(&Vchar, &Ichar, &Lchar, &n, data, &n, &zero, &zero, &index,
              &index, &zero, &one, eval, evec, &n, &null, &wkopt, &lwork,
              &iwkopt, &liwork, &info);

      lwork = (int)wkopt;
      double *work = new double[lwork];
      liwork = (int)iwkopt;
      int *iwork = new int[liwork];

      dsyevr_(&Vchar, &Ichar, &Lchar, &n, data, &n, &zero, &zero, &index,
              &index, &zero, &one, eval, evec, &n, &null, work, &lwork, iwork,
              &liwork, &info);

      LOG_NUMBER("Eigenvalue", eval[0]);
      LOG_VECTOR("Eigenvector", evec, n);
      this->eig_timer.lap(1);
      this->num_iterations++;

      // compute the residual = 2-norm of Ax-λx
      double *x_d = new double[n];
      double beta_d = -eval[0];
      dcopy_(&n, evec, &one, x_d, &one);
      dcopy_(&nn, data_db, &one, data, &one);
      dsymv_(&Lchar, &n, &one_d, data, &n, evec, &one, &beta_d, x_d, &one);
      this->max_residual = std::max(this->max_residual, dnrm2_(&n, x_d, &one));

      /* Inverse iterations */
      this->eig_timer.start();
      double max_evec = 0.0;
      for (int i = 1; i < n; i++)
        if (max_evec < fabs(evec[i])) max_evec = fabs(evec[i]);
      this->curr_max_abs_evec = max_evec;

      int iter = 0, max_iter = MAX_ITER;
      // control the number of inverse iterations
      if (max_evec > INV_ITER_TOL) max_iter = 0;

      // A = A - λI, then do LU factorization
      for (int i = 0; i < n; i++) {
        data[i + i * n] -= (eval[0] - FP_EPSILON);
        for (int j = i + 1; j < n; j++)
          data[i + j * n] = data[j + i * n];
      }
      int *ipiv = new int[n];
      if (max_iter > 0)
        dgetrf_(&n, &n, data, &n, ipiv, &info);
      // test if factorization succeeds
      if (info != 0) {
        printf("Factorization fails! \n");
        exit(-1);
      }

      this->eig_timer.lap(2);

      // store evec in a higher precision array
      std::vector<QUAD_PRECISION> quad_x(n, 0.0);
      std::vector<QUAD_PRECISION> quad_y(n, 0.0);
      for (int i = 0; i < n; i++) quad_x[i] = (QUAD_PRECISION)evec[i];
      // do inverse iteration
      for (iter = 0; iter < max_iter; iter++) {
        // (reference: LAPACK dgetrs)
        // y = x
        for (int i = 0; i < n; i++) quad_y[i] = quad_x[i];
        // y = Py
        for (int i = 0; i < n; i++) {
          QUAD_PRECISION temp = quad_y[i];
          quad_y[i] = quad_y[ipiv[i]-1];
          quad_y[ipiv[i]-1] = temp;
        }
        // Ly = y
        for (int i = 0; i < n; i++)
          for (int j = 0; j < i; j++)
            quad_y[i] = quad_y[i] - data[i + j * n] * quad_y[j];
        // Uy = y
        for (int i = n - 1; i >= 0; i--) {
          for (int j = i + 1; j < n; j++)
            quad_y[i] = quad_y[i] - data[i + j * n] * quad_y[j];
          quad_y[i] /= data[i + i * n];
        }
        // x = normalized y
        // (reference: LAPACK dnrm2)
        QUAD_PRECISION y_norm = 0.0;
        QUAD_PRECISION scale = 0.0, ssq = 1.0;
        for (int i = 0; i < n; i++) {
          if (quad_y[i] != 0) {
            QUAD_PRECISION absyi = fabsq(quad_y[i]);
            if (scale < absyi) {
              ssq = 1.0 + ssq * (scale / absyi) * (scale / absyi);
              scale = absyi;
            } else ssq += (absyi / scale) * (absyi / scale);
          }
        }
        y_norm = scale * sqrtq(ssq);

        bool flag = false;
        // printf("Iteration %d\n", iter);
        for (int i = 0; i < n; i++) {
          quad_y[i] /= y_norm;
          if (fabsq(quad_x[i]) != fabsq(quad_y[i])) flag = true;
          // printf("%d   %e\n", i, fabs((double) quad_x[i]) - fabs((double) quad_y[i]));
          quad_x[i] = quad_y[i];
        }
        if (!flag) break;
      }

      if (iter > 0) this->sum_iterations++;
      this->sum_inv_iter += iter;
      this->curr_inv_iter = iter;
      this->eig_timer.lap(3);

       // Use a double array to store delta_x = quad_x - x;
      int sign = ((quad_x[0] * evec[0]) > 0) ? 1 : -1;
      double *delta_x = new double[n];
      for (int i = 0; i < n; i++) {
        delta_x[i] = sign * quad_x[i] - evec[i];
      }
      // (x+Δx)'H(x+Δx) ~ x'Hx + 2*Δx'Hx
      dsymv_(&Lchar, &n, &one_d, data_db, &n, evec, &one, &zero, Hx, &one);
      QUAD_PRECISION rq = ddot_(&n, evec, &one, Hx, &one);
      rq += 2 * ddot_(&n, delta_x, &one, Hx, &one);

      QUAD_PRECISION lambda_q = (max_iter > 0) ? -rq : -eval[0];
      QUAD_PRECISION gamma_q = quad_x[0] / y1;
      // update scale (old scale is cancelled out)
      QUAD_PRECISION scale_q = gamma_q * sqrtq(lambda_q);
      // dx = x_new - x_old = a / scale - x;
      // the result is scaled
      for (int i = 0; i < k; i++)
        res[i] = (double) (quad_x[i + 1] / gamma_q - x[i]);

      // See how much inverse iteration improves
      scale = evec[0] / y1 * sqrt(-eval[0]);
      QUAD_PRECISION diff_inv_iter = fabsq(scale_q) - fabs(scale);
      this->sum_diff_inv_iter += fabsq(diff_inv_iter);
      this->max_diff_inv_iter = std::max(this->max_diff_inv_iter, fabs(diff_inv_iter));
      this->curr_diff_inv_iter = fabsq(diff_inv_iter);

      scale = scale_q;
      new_xx = lambda_q / scale_q / scale_q;
      // new_xz = -new_xx * lambda_q;

      // delete[] data, data_db, eval, evec, work, iwork, x_d, ipiv, delta_x;
      delete[] data;
      delete[] data_db;
      delete[] eval;
      delete[] evec;
      delete[] work;
      delete[] iwork;
      delete[] x_d;
      delete[] ipiv;
      delete[] delta_x;
      this->eig_timer.lap(4);
    } else {
      // Only in the first iteration this happens
      n = k;

      // Compute largest eigenvalue and the corresponding eigenvector of H_sub
      double *eval = new double[n];
      double *evec = new double[n];

      int lwork = -1;
      double wkopt;
      int liwork = -1;
      int iwkopt;
      int info;

      LOG_MATRIX("Subproblem", H_sub, k);

      // calculate eigenvalues using the DSYEVR subroutine
      // query the size of work and iwork
      dsyevr_(&Vchar, &Ichar, &Lchar, &n, H_sub, &n, &zero, &zero, &index,
              &index, &zero, &one, eval, evec, &n, &null, &wkopt, &lwork,
              &iwkopt, &liwork, &info);

      lwork = (int)wkopt;
      double *work = new double[lwork];
      liwork = (int)iwkopt;
      int *iwork = new int[liwork];

      // calculate eigenvalues using the DSYEVR subroutine
      dsyevr_(&Vchar, &Ichar, &Lchar, &n, H_sub, &n, &zero, &zero, &index,
              &index, &zero, &one, eval, evec, &n, &null, work, &lwork, iwork,
              &liwork, &info);

      LOG_NUMBER("Eigenvalue", eval[0]);
      LOG_VECTOR("Eigenvector", evec, k);

      new_xx = -eval[0];
      new_xz = -eval[0] * eval[0];
      // dx = x_new - x_old = a - x;
      for (int i = 0; i < k; i++)
        res[i] = evec[i] * sqrt(-eval[0]) - sub_det_list.data[i].second[0];

      // delete[] eval, evec, work, iwork;
      delete[] eval;
      delete[] evec;
      delete[] work;
      delete[] iwork;
    }

    xz.set_scale(scale);
    xz.set_xx(new_xx);
    // xz.set_xz(new_xz);

    delete[] x;
    delete[] z;
    delete[] Hx;

    return res;
  }
};

/* CDFCI solver */
class Solver {
 public:
  // Store the energy solved.
  double result_energy = 0;

  virtual ~Solver(){};
};

template <int N = 1>
class CDFCI : public Solver {
 public:
  using Column = typename Hamiltonian<N>::Column;
  Option option = {
      {"num_iterations", 10},           {"report_interval", 1},
      {"coordinate_pick", "gcd_grad"},  {"coordinate_update", "ls"},
      {"ref_det_occ", Option::array()}, {"z_threshold", 0.0},
      {"z_threshold_search", false},    {"max_wavefunction_size", 1000000}};

  CDFCI(){};

  CDFCI(Option &opt) { option.merge_patch(opt); }

  ~CDFCI(){};

  void get_hamiltonian_submatrix(Hamiltonian<N> &h,
                                 WaveFunctionVector<N> &det_picked,
                                 double *H_sub) {
    int k = det_picked.data.size();
    // initialize H_sub
    for (int i = 0; i < k * k; i++) H_sub[i] = 0.0;

#ifdef _OPENMP
#pragma omp parallel for num_threads(k)
#endif
    // fill in off-diagonals of submatrix
    for (int i = 0; i < k; i++) {
      Determinant<N> det1 = det_picked.data[i].first;
      DeterminantDecoded<N> det1_decoded(det1);
      for (int j = 0; j < k; j++) {
        Determinant<N> det2 = det_picked.data[j].first;
        DeterminantDecoded<N> det2_decoded(det2);
        double value = h.get_off_diagonal(det1_decoded, det2_decoded);
        H_sub[i * k + j] = value;
      }
    }

#ifdef _OPENMP
#pragma omp parallel for num_threads(k)
#endif
    // fill in diagonals of submatrix
    for (int i = 0; i < k; i++) {
      Determinant<N> curr_det = det_picked.data[i].first;
      DeterminantDecoded<N> det_decoded(curr_det);
      H_sub[i * k + i] = h.get_diagonal(det_decoded);
    }

    LOG_MATRIX("New method matrix", H_sub, k);
  }

  int solve(Hamiltonian<N> &h, CoordinatePick<N> &coord_pick,
            CoordinateUpdate<N> &coord_update, int k, WaveFunction<N> &xz,
            WaveFunctionVector<N> &sub_xz) {
    double energy = 0;

    int num_iter = option["num_iterations"];
    int report_interval = option["report_interval"];
    double z_threshold = option["z_threshold"];
    bool z_threshold_search = option["z_threshold_search"];
    size_t last_xz_size = 0;

    Timer global_timer, step_timer = Timer(6);
    global_timer.start();

    std::cout << std::setw(13) << std::left << "Iteration";
    std::cout << std::setw(18) << std::right << "Energy";
    std::cout << std::setw(18) << "|dx|_max";
    std::cout << std::setw(12) << "|x|_0";
    std::cout << std::setw(12) << "|z|_0";
    std::cout << std::setw(10) << "|H[I]|_0";
    std::cout << std::setw(10) << "upds";
    std::cout << std::setw(10) << "Time";
    std::cout << std::setw(10) << "scale";
    std::cout << std::endl;

    Column *cols = new Column[k];
    double *H_sub = new double[k * k];
    std::vector<double> dx;

    for (auto i = 0; i < num_iter / report_interval; ++i) {
      for (auto j = 0; j < report_interval; ++j) {
        LOG_NUMBER("Iteration", i * report_interval + j);
        step_timer.start();

        // Coordinate Pick
        auto det_picked = coord_pick(sub_xz, k);
        step_timer.lap(0);

        // Extract relevant columns of H and recalculate z [parallel]
        get_hamiltonian_submatrix(h, det_picked, H_sub);
        // extract_hamiltonian_and_calculate_z(h, det_picked, cols, H_sub, xz,
        //                                     z_threshold);
        step_timer.lap(1);

        // Get H(:, det_picked) and recalculate z if z_threshold
#ifdef _OPENMP
#pragma omp parallel for num_threads(k)
#endif
        for (int ii = 0; ii < k; ii++) {
          Determinant<N> curr_det = det_picked.data[ii].first;
          DeterminantDecoded<N> det_decoded(curr_det);
          cols[ii] = h.get_column(det_decoded);

/*
          // z_threshold will make some z in hashtable not correct
          if (z_threshold > 0 && fabs(det_picked.data[ii].second[0]) != 0) {
            double new_z = 0.0;
            for (int jj = 0; jj < cols[ii].size(); jj++) {
              auto det = cols[ii][jj].first;
              auto h = cols[ii][jj].second;
              double x, z;
              auto _ = xz.fetch_val(det, 0, x, z);
              new_z += h * x;
            }
            det_picked.data[ii].second[1] = new_z;
          }
*/
        }
        step_timer.lap(2);

        // Coordinate Update
        // This version updates xx and xz here
        // So in this version LS is not available
        dx = coord_update(H_sub, xz, det_picked);
        step_timer.lap(3);

        // Update x
        xz.update_x(det_picked, dx);
        step_timer.lap(4);

        // Update z [parallel]
        xz.update_z_and_get_sub(cols, det_picked, dx, sub_xz, z_threshold);
        step_timer.lap(5);

        LOG_NUMBER("current xx", (double) xz.xx());
        LOG_NUMBER("current xz", (double) xz.xz());
        LOG_NUMBER("current scale", (double) xz.scale());
      }
      // xz.refresh();
      energy = xz.get_variational_energy();
      auto xz_size = xz.size();
      double dx_max = 0.0;
      for (int index = 0; index < k; index++)
        dx_max = std::max(fabs(dx[index]), dx_max);
      size_t cols_size = 0;
      for (int index = 0; index < k; index++) cols_size += cols[index].size();

      global_timer.lap();

      // Output
      std::cout << std::setw(13) << std::left << (i + 1) * report_interval;
      std::cout << std::setw(18) << std::right << std::fixed
                << std::setprecision(10) << energy;
      std::cout << std::setw(18) << std::scientific << std::setprecision(4)
                << dx_max;
      std::cout << std::setw(12) << xz.x_size();
      std::cout << std::setw(12) << xz_size;
      std::cout << std::setw(10) << cols_size;
      std::cout << std::setw(10) << xz.updates_size();
      std::cout << std::setw(10) << std::fixed << std::setprecision(2)
                << global_timer.data[0];
      std::cout << std::setw(10) << std::fixed << std::setprecision(6)
                << (double) xz.scale();
      std::cout << std::endl;

      printf(
          "[LOG] max residual: %e, max eigenvector entry: %e, number of "
          "iterations provoking inv iters: %d, avg number of inv iters: %.1f, "
          "curr number of inv iters: %d, avg diff inv iter makes: %e, curr "
          "diff inv iter makes: %e, max diff inv iter makes: %e.\n",
          coord_update.max_residual, coord_update.curr_max_abs_evec,
          coord_update.sum_iterations,
          ((double)coord_update.sum_inv_iter) / coord_update.sum_iterations,
          coord_update.curr_inv_iter,
          coord_update.sum_diff_inv_iter / coord_update.sum_iterations,
          coord_update.curr_diff_inv_iter,
          coord_update.max_diff_inv_iter);

      // xz.debug_print_xx_xz();

      // Auto adjust z_threshold
      if (z_threshold_search) {
        auto inc_ratio =
            1000.0 * (xz_size - last_xz_size) / sub_xz.size() / report_interval;
        if ((inc_ratio >= 1) && (xz_size > 0.7 * xz.max_size)) {
          return -1;
        }
      }
      last_xz_size = xz.size();

      // Check overflow. May be unnecessary but cheap.
      if (xz_size > 0.79 * xz.max_size) {
        throw std::overflow_error(
            "The hash table is full. Please increase max_memory or "
            "z_threshold.");
      }
    }
    delete[] cols;
    delete[] H_sub;

    step_timer.log("step timer");
    coord_update.eig_timer.log("eig timer");
    xz.wf_timer.log("wf timer");

    // Store the final computed energy in the Solver class.
    result_energy = energy;
    return 0;
  }

  int solve(Hamiltonian<N> &h, CoordinatePick<N> &coord_pick,
            CoordinateUpdate<N> &coord_update, int k, int m) {
    bool z_threshold_search = option["z_threshold_search"];
    double z_threshold = option["z_threshold"];

    // Print information
    std::cout << "CDFCI calculation" << std::endl;
    std::cout << "-----------------" << std::endl;
    // Find the initial value.
    auto hf = get_ref_det(h);
    DeterminantDecoded<N> hf_decoded(hf);
    auto hf_energy = h.get_diagonal(hf_decoded);
    std::cout << "Hartree Fock determinant occupied spin-orbitals:"
              << std::endl;
    for (auto &orb : hf_decoded.occupied_orbitals) {
      std::cout << orb << "  ";
    }
    std::cout << std::endl;
    std::cout << "Hartree Fock energy: " << std::fixed << std::setprecision(10)
              << hf_energy << std::endl
              << std::endl;

    while (true) {
      // Initialize xz
#ifdef _OPENMP
      WaveFunctionCuckoo<N> xz(option["max_wavefunction_size"]);
#else
      WaveFunctionStd<N> xz(option["max_wavefunction_size"]);
#endif
      xz.MM = m;

      // Initialize sub_xz, x = e_HF and z = H[:,HF], xx = 1
      WaveFunctionVector<N> sub_xz;
      Column *cols = new Column[1];
      cols[0] = h.get_column(hf_decoded);
      WaveFunctionVector<N> det_picked;
      det_picked.data.push_back({hf, {0.0, 0.0}});
      std::vector<double> dx = {1.0};
      xz.set_scale(1.0);
      xz.set_xx(1.0);
      xz.set_xz(0.0); // calculate xz manually
      xz.update_x(det_picked, dx);
      xz.update_z_and_get_sub(cols, det_picked, dx, sub_xz, z_threshold);

      delete[] cols;

      LOG_NUMBER("Init xx", (double) xz.xx());
      LOG_NUMBER("Init xz", (double) xz.xz());
      LOG_NUMBER("Init scale", (double) xz.scale());

      auto ierr = solve(h, coord_pick, coord_update, k, xz, sub_xz);
      if (z_threshold_search && (ierr == -1)) {
        z_threshold = z_threshold * 2;
        option["z_threshold"] = z_threshold;
        std::cout << "z_threshold_search: z_threshold is too small. Increase "
                     "z_threhold to ";
        std::cout << std::scientific << std::setprecision(2)
                  << option["z_threshold"] << " and restart." << std::endl;
      } else {
        return ierr;
      }
    }
  }

  Determinant<N> get_ref_det(Hamiltonian<N> &h) const {
    std::cout << "N = " << N << std::endl;
    std::vector<Orbital> occ_list = option["ref_det_occ"];
    Determinant<N> hf;
    if (occ_list.empty()) {
      // std::cout << "Reference determinant is not provided. Use Hartree Fock
      // state from FCIDUMP." << std::endl;
      return h.get_hartree_fock();
    } else {
      for (auto orb : occ_list) {
        if ((orb >= 0) && (orb < h.norb)) {
          hf.set_orbital(orb);
        }
      }
      if (hf.get_occupied_orbitals().size() != h.nelec) {
        throw std::invalid_argument(
            "Reference determinant is invalid. The number of valid orbitals "
            "does not match "
            "the number of electrons. Please provide " +
            std::to_string(h.nelec) + " orbitals from 0 to " +
            std::to_string(h.norb - 1) + " or remove the argument.");
      }
    }
    return hf;
  }
};

#endif
