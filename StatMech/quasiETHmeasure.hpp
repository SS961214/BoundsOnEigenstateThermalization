#pragma once

#include <Eigen/Core>
#pragma omp declare reduction(+ : Eigen::VectorXd : omp_out = omp_out + omp_in) \
    initializer(omp_priv = omp_orig)

#pragma omp declare reduction(+ : Eigen::MatrixXd : omp_out = omp_out + omp_in) \
    initializer(omp_priv = omp_orig)

#pragma omp declare reduction(+ : Eigen::MatrixXcd : omp_out = omp_out + omp_in) \
    initializer(omp_priv = omp_orig)

#pragma omp declare reduction(+ : Eigen::ArrayXd : omp_out = omp_out + omp_in) \
    initializer(omp_priv = omp_orig)

#pragma omp declare reduction(+ : Eigen::ArrayXcd : omp_out = omp_out + omp_in) \
    initializer(omp_priv = omp_orig)

#include "quasiETHmeasure_Spin.hpp"
#include "quasiETHmeasure_BosonFermion.hpp"