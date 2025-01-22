//   ______  _______   _______   ______  __
//  /      ||       \ |   ____| /      ||  |
// |  ,----'|  .--.  ||  |__   |  ,----'|  |
// |  |     |  |  |  ||   __|  |  |     |  |
// |  `----.|  '--'  ||  |     |  `----.|  |
//  \______||_______/ |__|      t\______||__|
//
// Coordinate Descent Full Configuration Interaction (CDFCI) package in C++14
// https://github.com/quan-tum/CDFCI
//
// Copyright (c) 2019, Zhe Wang, Yingzhou Li and Jianfeng Lu
// All rights reserved.
//
// This source code is licensed under the BSD 3-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef CDFCI_WAVEFUNCTION_H
#define CDFCI_WAVEFUNCTION_H 1

#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_map>

#include "determinant.h"
#include "hamiltonian.h"
#include "lib/libcuckoo/cuckoohash_map.hh"
#include "lib/robin_hood/robin_hood.h"

template <int N = 1>
int detComp_global(Determinant<N>& a, Determinant<N>& b) {
  int i;
  for (i = 0; i < N - 1 && a.repr[i] == b.repr[i]; i++);
  long diff = a.repr[i] - b.repr[i];
  return (diff > 0) - (diff < 0);
};

template <int N>
struct DeterminantEqual {
  bool operator()(const Determinant<N>& det1,
                  const Determinant<N>& det2) const {
    bool result = true;
    for (auto i = 0; i < N; ++i) {
      result = result && (det1.repr[i] == det2.repr[i]);
    }
    return result;
  }
};

template <int N = 1>
class WaveFunctionVector {
 public:
  using value_type = std::pair<Determinant<N>, std::array<double, 2>>;
  std::vector<value_type> data;  // TODO: use iterator, and move to private.

  std::array<QUAD_PRECISION, 2> norm_square_ = {0.0, 1.0};
  QUAD_PRECISION dot_product_ = 0.0;
  // Temporary data when updating xz.
  size_t n_new_element = 0;  // Number of elements not in xz. Used to update
                             // xz.size() for Cuckoo hash table.
  double new_z = 0.0;          // Store the recalculated z_i.

 public:
  WaveFunctionVector(long capacity = 0) { data.reserve(capacity); }

  ~WaveFunctionVector(){};

  QUAD_PRECISION xx() const { return norm_square_[0]; }

  void set_xx(QUAD_PRECISION xx) { norm_square_[0] = xx; }

  QUAD_PRECISION scale() const { return norm_square_[1]; }

  void set_scale(QUAD_PRECISION scale) { norm_square_[1] = scale; }

  QUAD_PRECISION xz() const { return dot_product_; }

  void set_xz(QUAD_PRECISION xz) { dot_product_ = xz; }

  size_t size() const { return data.size(); }

  void clear() {
    data.clear();
    norm_square_ = {0.0, 1.0};
    dot_product_ = 0.0;
    n_new_element = 0L;
    new_z = 0.0;
  }

  void append_WFVector(WaveFunctionVector<N>& sub_xz) {
    data.insert(data.end(), sub_xz.data.begin(), sub_xz.data.end());
  }

  void append_vector(std::vector<value_type>& small_data) {
    data.insert(data.end(), small_data.begin(), small_data.end());
  }

  /**
   * @brief  this function assumes:
   * 1) no duplicates in small_data
   * 2) data is sorted
   *
   * @param small_data
   */
  void append_vector_unique(std::vector<value_type>& small_data) {
    const auto detComp = [](auto& a, auto& b) {
      int i;
      for (i = 0; i < N - 1 && a.first.repr[i] == b.first.repr[i]; i++)
        ;
      return (a.first.repr[i] < b.first.repr[i]);
    };
    DeterminantEqual<N> isEqual;

    if (data.size() == 0) {
      std::sort(small_data.begin(), small_data.end(), detComp);
      data.insert(data.end(), small_data.begin(), small_data.end());
    } else {
      for (auto& item : small_data) {
        auto lower = std::lower_bound(data.begin(), data.end(), item, detComp);
        if (!isEqual(lower->first, item.first)) data.insert(lower, item);
      }
    }
  }

  /**
   * @brief If the element is already in the vector,
   *        do update, not insert.
   *
   * @param element
   */
  void append_element_unique(value_type element) {
    const auto detComp = [](auto& a, auto& b) {
      int i;
      for (i = 0; i < N - 1 && a.first.repr[i] == b.first.repr[i]; i++)
        ;
      return (a.first.repr[i] < b.first.repr[i]);
    };
    DeterminantEqual<N> isEqual;

    if (data.size() == 0) {
      data.push_back(element);
    } else {
      auto lower = std::lower_bound(data.begin(), data.end(), element, detComp);
      if (!isEqual(lower->first, element.first))
        data.insert(lower, element);
      else
        lower->second = element.second;
    }
  }

  void printXZ() {
    std::cout << "####################################" << std::endl;
    for (int i = 0; i < data.size(); i++) {
      printf("%d: %lu, %f, %f\n", i, data[i].first.repr[0], data[i].second[0],
             data[i].second[1]);
      if (i > 0 && (data[i].first.repr[0] == data[i - 1].first.repr[0]))
        printf("DUPLICATE!!!\n");
    }
    std::cout << "####################################" << std::endl;
  }
};

template <int N = 1>
class WaveFunction {
 protected:
  // norm_square_[0] = xx, norm_square[1] = scale
  // dot_product = xz
  std::array<QUAD_PRECISION, 2> norm_square_ = {0, 1};
  QUAD_PRECISION dot_product_ = 0;

 public:
  using key_type = Determinant<N>;
  using value_type = std::array<double, 2>;

  size_t max_size = 0;
  // A parameter that controls the trade-off between algorithm efficiency
  // and parallel efficiency
  int MM = 100;
  Timer wf_timer = Timer(2);

  QUAD_PRECISION xx() const { return norm_square_[0]; }

  void set_xx(QUAD_PRECISION xx) { norm_square_[0] = xx; }

  QUAD_PRECISION scale() const { return norm_square_[1]; }

  void set_scale(QUAD_PRECISION scale) { norm_square_[1] = scale; }

  QUAD_PRECISION xz() const { return dot_product_; }

  void set_xz(QUAD_PRECISION xz) { dot_product_ = xz; }

  void update_xz(QUAD_PRECISION inc) { dot_product_ += inc; }

  double get_variational_energy() const {
    // xz/xx
    return (double) dot_product_ / norm_square_[0];
  }

  virtual void update_x(WaveFunctionVector<N>& det_picked,
                        std::vector<double>& dx_vec){};
  virtual void update_z_and_get_sub(
      std::vector<std::pair<key_type, double>>* cols,
      WaveFunctionVector<N>& det_picked, std::vector<double>& dx_vec,
      WaveFunctionVector<N>& sub_xz, double z_threshold){};
  virtual double fetch_val(Determinant<N>& det, double s, double& x,
                           double& z) = 0;
  virtual size_t x_size() = 0;
  virtual size_t size() = 0;
  virtual size_t update_size(size_t n_new_element){ return 0; };
  virtual size_t updates_size(){ return 0; };
  virtual void reinsert_z(key_type& key, double new_z){};

  // double fetch_val(Determinant<N>& det, double s, double& x, double& z) {
  //   return 0.0;
  // }
  void debug_calculate_xx_xz(double& xx, double& xz) { return; }
  void debug_print_xx_xz() { return; }

  virtual ~WaveFunction(){};
};

/* Hash function mapping determinants to size_t */
// Warning: The following hash functions are naive and just okay to use. They
// should be replaced with better hash functions. The determinants are
// represented as an array of size_t integers.
template <int N>
struct DeterminantHash {
  size_t operator()(const Determinant<N>& det) const {
    throw std::invalid_argument(
        "The DeterminantHash does not support N > 4. Please modify the "
        "\"DeterminantHash\" class in \"wavefunction.h\" and provide a "
        "DeterminantHash.");
    return 0;
  }
};

template <>
size_t DeterminantHash<1>::operator()(const Determinant<1>& det) const {
  return det.repr[0];
}

template <>
size_t DeterminantHash<2>::operator()(const Determinant<2>& det) const {
  return det.repr[0] * 2038076783 + det.repr[1] * 179426549;
}

template <>
size_t DeterminantHash<3>::operator()(const Determinant<3>& det) const {
  return det.repr[0] * 2038076783 + det.repr[1] * 179426549 +
         det.repr[2] * 500002577;
}

template <>
size_t DeterminantHash<4>::operator()(const Determinant<4>& det) const {
  /*
  return det.repr[0] * 2038076783 + det.repr[1] * 179426549 +
         det.repr[2] * 500002577 + det.repr[3] * 255477023;
  */
    // A simple hash function combining the four values
    size_t hashValue = 0;

    for (size_t i = 0; i < 4; ++i) {
        // Mix the bits using bitwise XOR and left shift
        hashValue ^= (det.repr[i] + 0x9e3779b9 + (hashValue << 6) + (hashValue >> 2));
    }

    return hashValue;
}

template <int N>
struct DeterminantHashRobinhood {
  size_t operator()(const Determinant<N>& det) const {
    DeterminantHash<N> hash1;
    return (robin_hood::hash_int(hash1(det)));
  }
};

template <int N = 1>
class WaveFunctionStd : public WaveFunction<N> {
  using typename WaveFunction<N>::key_type;
  using typename WaveFunction<N>::value_type;
  using Column = std::vector<std::pair<Determinant<N>, double>>;

  using WaveFunction<N>::norm_square_;
  using WaveFunction<N>::dot_product_;
  using WaveFunction<N>::max_size;

 private:
  robin_hood::unordered_flat_map<key_type, value_type, DeterminantHash<N>,
                                 DeterminantEqual<N>>
      data_;
  size_t x_size_ = 0;

 public:
  WaveFunctionStd(size_t capacity) {
    data_.reserve(capacity);
    max_size = capacity;
  }

  ~WaveFunctionStd(){};

  void update_x(WaveFunctionVector<N>& det_picked,
                std::vector<double>& dx_vec) {
    for (int i = 0; i < dx_vec.size(); i++) {
      auto key = det_picked.data[i].first;
      auto dx = dx_vec[i];

      auto iter = data_.find(key);
      if (iter != data_.end()) {
        iter->second[0] += dx;
      } else {
        value_type val{dx, 0};
        data_.insert({key, val});
        x_size_ ++;
      }
    }
    return;
  }

  double fetch_val(Determinant<N>& det, double s, double& x, double& z) {
    return 0.0;
  }

  /**
   * @brief Updating z in single thread, one coordinate by one.
   *
   * Note that one coordinate might be updated multiple times.
   * When inserting into sub_xz, overwrite the existing element.
   *
   * @param cols
   * @param det_picked
   * @param dx_vec
   * @param sub_xz
   * @param z_threshold
   */
  void update_z_and_get_sub(std::vector<std::pair<key_type, double>>* cols,
                            WaveFunctionVector<N>& det_picked,
                            std::vector<double>& dx_vec,
                            WaveFunctionVector<N>& sub_xz,
                            double z_threshold = 0.0) {
    int k = dx_vec.size();
    sub_xz.clear();

    for (int columnIdx = 0; columnIdx < k; columnIdx ++)
    {
      double dx = dx_vec[columnIdx];
      // Recalculate z -> new_z
      double new_z = 0.0;

      for (std::pair<key_type, double>& entry : cols[columnIdx]) {
        key_type& det = entry.first;
        double dz = entry.second * dx;
        double x = 0.0;

        // Update z part
        auto iter = data_.find(det);
        value_type val;
        if (iter != data_.end()) {
          x = iter->second[0];
          // Update z
          iter->second[1] += dz;
          val = {x, iter->second[1]};
          sub_xz.append_element_unique({det, val});
        } else {
          // Insert only if above threshold
          if (fabs(dz * this->scale()) > z_threshold) {
            val = {0, dz};
            data_.insert({det, val});
            // Update sub_xz
            sub_xz.append_element_unique({det, val});
          }
        }
        // z_i = H(i,:) * x(:)
        new_z += entry.second * val[0];
      }

      auto key = det_picked.data[columnIdx].first;
      reinsert_z(key, new_z);
    }

    // Update sub_xz
    sub_xz.set_xx(norm_square_[0]);
    sub_xz.set_scale(norm_square_[1]);
    sub_xz.set_xz(dot_product_);
    return;
  }

  size_t x_size() { return x_size_; }
  size_t size() { return data_.size(); }

  void reinsert_z(key_type& key, double new_z) {
    value_type val{0, new_z};
    data_.insert({key, val});
  };
};

template <int N = 1>
class WaveFunctionCuckoo : public WaveFunction<N> {
  using typename WaveFunction<N>::key_type;
  using typename WaveFunction<N>::value_type;
  using Column = std::vector<std::pair<Determinant<N>, double>>;

  using WaveFunction<N>::norm_square_;
  using WaveFunction<N>::dot_product_;
  using WaveFunction<N>::max_size;

 private:
  libcuckoo::cuckoohash_map<key_type, value_type, DeterminantHashRobinhood<N>,
                 DeterminantEqual<N>,
                 std::allocator<std::pair<const key_type, value_type>>, 8>
      data_;
  size_t x_size_ = 0;
  size_t z_size_ = 0;
  size_t updates_ = 0;

 public:
  WaveFunctionCuckoo(size_t capacity) {
    data_.reserve(capacity); // allocated at the very beginning
    max_size = capacity;
  }

  ~WaveFunctionCuckoo(){};

  void update_x(WaveFunctionVector<N>& det_picked,
                std::vector<double>& dx_vec) {
    int n_new_element = 0;
    for (int i = 0; i < dx_vec.size(); i++) {
      auto& key = det_picked.data[i].first;
      auto dx = dx_vec[i];

      value_type xz{0, 0};
      value_type val_new{dx, 0};
      auto update_fn = [dx, &xz, &n_new_element](value_type& val) {
        // Substract new element if val[0] != 0
        if (fabs(val[0]) != 0)
          n_new_element--;
        // Update x
        val[0] += dx;
        // Get the value after update
        xz[0] = val[0];
        xz[1] = val[1];
        return false;  // always return false
      };
      n_new_element++;
      data_.upsert(key, update_fn, val_new);
      dot_product_ += xz[1] * dx;
    }
    // Update elements
    update_x_size(n_new_element);
  }

  /**
   * @brief Updating z in parallel. Each thread responsible for one coordinate.
   *
   * Stage 1: N pick M. Each thread updates z and picks out the largest M
   *          from priority queue.
   *          z is also recalculated for picked coordinates.
   *
   * Stage 2: Refetch correct z from hashtable.
   *
   * @param cols
   * @param det_picked
   * @param dx_vec
   * @param sub_xz
   * @param z_threshold
   */
  void update_z_and_get_sub(std::vector<std::pair<key_type, double>>* cols,
                            WaveFunctionVector<N>& det_picked,
                            std::vector<double>& dx_vec,
                            WaveFunctionVector<N>& sub_xz,
                            double z_threshold = 0.0) {
    int k = dx_vec.size();
    sub_xz.clear();
    QUAD_PRECISION xss = this->xx() * this->scale() * this->scale();
    this->wf_timer.start();
    updates_ = 0;

#ifdef _OPENMP
#pragma omp parallel num_threads(k)
    {
      int threadIdx = omp_get_thread_num();
      double dot_product_thread = 0.0;
      double dx = dx_vec[threadIdx];
      int n_new_element = 0;     // new inserted elements
      int n_update_element = 0;  // updated elements (insert + update)
      // Recalculate z -> new_z
      double new_z = 0.0;
      // T1: the datatype used in heap largestM.
      // T2: the datatype used after refetch from hashtable.
      typedef std::pair<key_type*, double> T1;
      typedef std::pair<key_type, value_type> T2;

      // Stage 1: N pick M
      auto cmpT1 = [](T1 left, T1 right) {
        return left.second > right.second;
      };  // the smallest on the top
      std::priority_queue<T1, std::vector<T1>, decltype(cmpT1)> largestM(cmpT1);
      for (std::pair<key_type, double>& entry : cols[threadIdx]) {
        key_type& det = entry.first;
        double dz = entry.second * dx;
        double x = 0.0;

        // Update z part
        value_type val_new = {0.0, 0.0};
        auto update_functor = [dz, &val_new, &n_new_element](value_type& val) {
          // Update z
          val[1] += dz;
          // Get the value after update
          val_new[0] = val[0];
          val_new[1] = val[1];
          // Substract new element because update does not incur new element [Algo 1]
          n_new_element--;
          return false;  // always return false
        };

        /* Algo 1: Upsert if above threshold, update otherwise */
        if (fabs(dz * this->scale()) > z_threshold) {
          val_new = {0.0, dz};
          n_new_element++;
          n_update_element++;
          data_.upsert(det, update_functor, val_new);
        } else {
          // Call update_functor if det is found
          int result = data_.update_fn(det, update_functor);
          n_new_element += result;
          n_update_element += result;
        }

        /* Algo 2: Call update_functor if det is found
        bool flag_found = data_.update_fn(det, update_functor);
        if ((!flag_found) && (fabs(dz * this->scale()) > z_threshold)) // Only update if above threold
        {
          val_new = {0.0, dz};
          data_.insert(det, val_new);
          n_new_element++;
        }
        */

        // z_i = H(i,:) * x(:)
        new_z += entry.second * val_new[0];
        dot_product_thread += dz * val_new[0];

        // Choose the M largest part
        double val = fabs(val_new[0] * xss + val_new[1]);
        if (largestM.size() < this->MM)
          largestM.push({&det, val});
        else if (largestM.top().second < val) {
          largestM.pop();
          largestM.push({&det, val});
        }
      }

#pragma omp barrier
      if (threadIdx == 0) this->wf_timer.lap(0);

      // Reinsert z
      // (the barrier is set so that no more threads can modify the new_z)
      auto key = det_picked.data[threadIdx].first;
      value_type val_new{0, new_z};
      auto update_fn = [new_z](value_type& val) {
        val[1] = new_z;
        return false;
      };
      data_.upsert(key, update_fn, val_new);

      // Stage 2: Refetch the correct gradients from the hash table
      std::vector<T2> correctM;
      while (!largestM.empty()) {
        key_type det = *(largestM.top().first);
        largestM.pop();
        value_type val_new{0, 0};
        bool flag_found = data_.find(det, val_new);
        if (flag_found) correctM.push_back({det, val_new});
      }

// Aggregate into one subvector
#pragma omp critical
      {
        // Update elements
        this->update_size(n_new_element);
        updates_ += n_update_element;
        dot_product_ += dot_product_thread;
        // Aggregate sub_xz
        // if (z_threshold > 0.0)
        //   sub_xz.append_vector_unique(correctM);
        // else
        sub_xz.append_vector(correctM);
      }
    }  // end of parallel section
#endif
    this->wf_timer.lap(1);

    // Check overflow
    if (size() > 0.79 * max_size) {
      throw std::overflow_error(
          "The hash table is full. Please increase max_wavefunction_size or "
          "z_threshold.");
    }

    sub_xz.set_xx(norm_square_[0]);
    sub_xz.set_scale(norm_square_[1]);
    sub_xz.set_xz(dot_product_);
    return;
  }

  size_t x_size() { return x_size_; }

  size_t update_x_size(size_t n_new_element) {
    x_size_ += n_new_element;
    return x_size_;
  }

  size_t size() { return z_size_; }
  size_t updates_size() { return updates_; }

  size_t update_size(size_t n_new_element) {
    z_size_ += n_new_element;
    return z_size_;
  }

  void reinsert_z(key_type& key, double new_z) {
    value_type val_new{0, new_z};
    auto update_fn = [new_z](value_type& val) {
      val[1] = new_z;
      return false;
    };
    data_.upsert(key, update_fn, val_new);
  }

  double fetch_val(Determinant<N>& det, double s, double& x, double& z) {
    std::array<double, 2> val_new{0, 0};
    bool flag_found = data_.find(det, val_new);
    x = val_new[0];
    z = val_new[1];
    double val2 = fabs(x * s + z);
    return val2;
  }

  void debug_calculate_xx_xz(double& xx, double& xz) {
    // very useful debug function :-(
    auto lt = data_.lock_table();
    xx = 0.0;
    xz = 0.0;
    for (const auto& it : lt) {
      // if (it.second[0] != 0)
      //     printf("det = %d, z = %f\n", it.first.repr[0], it.second[1]);
      xx += it.second[0] * it.second[0];
      xz += it.second[0] * it.second[1];
    }
  }

  void debug_print_xx_xz() {
    // DEBUG: check if f monotonically decrease.
    double real_xx, real_xz;
    debug_calculate_xx_xz(real_xx, real_xz);
    double scale = this->scale();
    double real_f = real_xx * real_xx * scale * scale * scale * scale +
                    2 * real_xz * scale * scale;
    std::cout << "[RE] xx = " << std::setprecision(10) << real_xx
              << " xz = " << real_xz << " f = " << real_f << std::endl;
    double f = this->xx() * this->xx() * scale * scale * scale * scale +
               2 * this->xz() * scale * scale;
    std::cout << "[CD] xx = " << std::setprecision(10) << this->xx()
              << " xz = " << this->xz() << " f = " << f << std::endl;
    if (fabs(real_xx - this->xx()) > FP_EPSILON) exit(-1);
  }
};

/* Extract chosen columns and submatrix of Hamiltonian, columns sorted */
template <int N = 1>
void extract_hamiltonian_and_calculate_z(Hamiltonian<N>& h,
                                         WaveFunctionVector<N>& det_picked,
                                         typename Hamiltonian<N>::Column* cols,
                                         double* H_sub, WaveFunction<N>& xz,
                                         double z_threshold) {
  int k = det_picked.data.size();
  // initialize H_sub
  for (int i = 0; i < k * k; i++) H_sub[i] = 0.0;

    // extract columns and fill in submatrix
#ifdef _OPENMP
#pragma omp parallel for num_threads(k)
#endif
  for (int i = 0; i < k; i++) {
    Determinant<N> curr_det = det_picked.data[i].first;
    DeterminantDecoded<N> det_decoded(curr_det);
    typename Hamiltonian<N>::Column& col = cols[i];
    col.clear();
    h.get_sorted_column(det_decoded, col);

    // z_threshold will make some z in hashtable not correct
    if (z_threshold > 0 && fabs(det_picked.data[i].second[0]) < FP_EPSILON) {
      double new_z = 0.0;
      for (int j = 0; j < col.size(); j++) {
        auto det = col[j].first;
        auto h = col[j].second;
        double x, z;
        auto _ = xz.fetch_val(det, 0, x, z);
        new_z += h * x;
      }
      det_picked.data[i].second[1] = new_z;
    }

    // traverse through sorted col
    int colidx = 0, detidx = 0;
    double* subcol = H_sub + i * k;
    while (colidx < col.size() && detidx < k) {
      int res =
          detComp_global(col[colidx].first, det_picked.data[detidx].first);
      if (res == 0)
        subcol[detidx++] = col[colidx++].second;
      else if (res > 0)
        detidx++;
      else
        colidx++;
    }

    subcol[i] = h.get_diagonal(det_decoded);
  }
}

#endif
