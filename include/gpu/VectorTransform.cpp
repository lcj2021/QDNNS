/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <VectorTransform.h>

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>

#include <FaissAssert.h>
#include <distances.h>
#include <random.h>
#include <utils.h>

using namespace faiss;

extern "C" {

// this is to keep the clang syntax checker happy
#ifndef FINTEGER
#define FINTEGER int
#endif

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

int sgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* b,
        FINTEGER* ldb,
        float* beta,
        float* c,
        FINTEGER* ldc);

int dgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const double* alpha,
        const double* a,
        FINTEGER* lda,
        const double* b,
        FINTEGER* ldb,
        double* beta,
        double* c,
        FINTEGER* ldc);

int ssyrk_(
        const char* uplo,
        const char* trans,
        FINTEGER* n,
        FINTEGER* k,
        float* alpha,
        float* a,
        FINTEGER* lda,
        float* beta,
        float* c,
        FINTEGER* ldc);

/* Lapack functions from http://www.netlib.org/clapack/old/single/ */

int ssyev_(
        const char* jobz,
        const char* uplo,
        FINTEGER* n,
        float* a,
        FINTEGER* lda,
        float* w,
        float* work,
        FINTEGER* lwork,
        FINTEGER* info);

int dsyev_(
        const char* jobz,
        const char* uplo,
        FINTEGER* n,
        double* a,
        FINTEGER* lda,
        double* w,
        double* work,
        FINTEGER* lwork,
        FINTEGER* info);

int sgesvd_(
        const char* jobu,
        const char* jobvt,
        FINTEGER* m,
        FINTEGER* n,
        float* a,
        FINTEGER* lda,
        float* s,
        float* u,
        FINTEGER* ldu,
        float* vt,
        FINTEGER* ldvt,
        float* work,
        FINTEGER* lwork,
        FINTEGER* info);

int dgesvd_(
        const char* jobu,
        const char* jobvt,
        FINTEGER* m,
        FINTEGER* n,
        double* a,
        FINTEGER* lda,
        double* s,
        double* u,
        FINTEGER* ldu,
        double* vt,
        FINTEGER* ldvt,
        double* work,
        FINTEGER* lwork,
        FINTEGER* info);
}

/*********************************************
 * VectorTransform
 *********************************************/

float* VectorTransform::apply(idx_t n, const float* x) const {
    float* xt = new float[n * d_out];
    apply_noalloc(n, x, xt);
    return xt;
}

void VectorTransform::train(idx_t, const float*) {
    // does nothing by default
}

void VectorTransform::reverse_transform(idx_t, const float*, float*) const {
    FAISS_THROW_MSG("reverse transform not implemented");
}

void VectorTransform::check_identical(const VectorTransform& other) const {
    FAISS_THROW_IF_NOT(other.d_in == d_in && other.d_in == d_in);
}

/*********************************************
 * LinearTransform
 *********************************************/

/// both d_in > d_out and d_out < d_in are supported
LinearTransform::LinearTransform(int d_in, int d_out, bool have_bias)
        : VectorTransform(d_in, d_out),
          have_bias(have_bias),
          is_orthonormal(false),
          verbose(false) {
    is_trained = false; // will be trained when A and b are initialized
}

void LinearTransform::apply_noalloc(idx_t n, const float* x, float* xt) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "Transformation not trained yet");

    float c_factor;
    if (have_bias) {
        FAISS_THROW_IF_NOT_MSG(b.size() == d_out, "Bias not initialized");
        float* xi = xt;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d_out; j++)
                *xi++ = b[j];
        c_factor = 1.0;
    } else {
        c_factor = 0.0;
    }

    FAISS_THROW_IF_NOT_MSG(
            A.size() == d_out * d_in, "Transformation matrix not initialized");

    float one = 1;
    FINTEGER nbiti = d_out, ni = n, di = d_in;
    sgemm_("Transposed",
           "Not transposed",
           &nbiti,
           &ni,
           &di,
           &one,
           A.data(),
           &di,
           x,
           &di,
           &c_factor,
           xt,
           &nbiti);
}

void LinearTransform::transform_transpose(idx_t n, const float* y, float* x)
        const {
    if (have_bias) { // allocate buffer to store bias-corrected data
        float* y_new = new float[n * d_out];
        const float* yr = y;
        float* yw = y_new;
        for (idx_t i = 0; i < n; i++) {
            for (int j = 0; j < d_out; j++) {
                *yw++ = *yr++ - b[j];
            }
        }
        y = y_new;
    }

    {
        FINTEGER dii = d_in, doi = d_out, ni = n;
        float one = 1.0, zero = 0.0;
        sgemm_("Not",
               "Not",
               &dii,
               &ni,
               &doi,
               &one,
               A.data(),
               &dii,
               y,
               &doi,
               &zero,
               x,
               &dii);
    }

    if (have_bias)
        delete[] y;
}

void LinearTransform::set_is_orthonormal() {
    if (d_out > d_in) {
        // not clear what we should do in this case
        is_orthonormal = false;
        return;
    }
    if (d_out == 0) { // borderline case, unnormalized matrix
        is_orthonormal = true;
        return;
    }

    double eps = 4e-5;
    FAISS_ASSERT(A.size() >= d_out * d_in);
    {
        std::vector<float> ATA(d_out * d_out);
        FINTEGER dii = d_in, doi = d_out;
        float one = 1.0, zero = 0.0;

        sgemm_("Transposed",
               "Not",
               &doi,
               &doi,
               &dii,
               &one,
               A.data(),
               &dii,
               A.data(),
               &dii,
               &zero,
               ATA.data(),
               &doi);

        is_orthonormal = true;
        for (long i = 0; i < d_out; i++) {
            for (long j = 0; j < d_out; j++) {
                float v = ATA[i + j * d_out];
                if (i == j)
                    v -= 1;
                if (fabs(v) > eps) {
                    is_orthonormal = false;
                }
            }
        }
    }
}

void LinearTransform::reverse_transform(idx_t n, const float* xt, float* x)
        const {
    if (is_orthonormal) {
        transform_transpose(n, xt, x);
    } else {
        FAISS_THROW_MSG(
                "reverse transform not implemented for non-orthonormal matrices");
    }
}

void LinearTransform::print_if_verbose(
        const char* name,
        const std::vector<double>& mat,
        int n,
        int d) const {
    if (!verbose)
        return;
    printf("matrix %s: %d*%d [\n", name, n, d);
    FAISS_THROW_IF_NOT(mat.size() >= n * d);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            printf("%10.5g ", mat[i * d + j]);
        }
        printf("\n");
    }
    printf("]\n");
}

void LinearTransform::check_identical(const VectorTransform& other_in) const {
    VectorTransform::check_identical(other_in);
    auto other = dynamic_cast<const LinearTransform*>(&other_in);
    FAISS_THROW_IF_NOT(other);
    FAISS_THROW_IF_NOT(other->A == A && other->b == b);
}

/*********************************************
 * RandomRotationMatrix
 *********************************************/

void RandomRotationMatrix::init(int seed) {
    if (d_out <= d_in) {
        A.resize(d_out * d_in);
        float* q = A.data();
        float_randn(q, d_out * d_in, seed);
        matrix_qr(d_in, d_out, q);
    } else {
        // use tight-frame transformation
        A.resize(d_out * d_out);
        float* q = A.data();
        float_randn(q, d_out * d_out, seed);
        matrix_qr(d_out, d_out, q);
        // remove columns
        int i, j;
        for (i = 0; i < d_out; i++) {
            for (j = 0; j < d_in; j++) {
                q[i * d_in + j] = q[i * d_out + j];
            }
        }
        A.resize(d_in * d_out);
    }
    is_orthonormal = true;
    is_trained = true;
}

void RandomRotationMatrix::train(idx_t /*n*/, const float* /*x*/) {
    // initialize with some arbitrary seed
    init(12345);
}

/*********************************************
 * PCAMatrix
 *********************************************/

PCAMatrix::PCAMatrix(
        int d_in,
        int d_out,
        float eigen_power,
        bool random_rotation)
        : LinearTransform(d_in, d_out, true),
          eigen_power(eigen_power),
          random_rotation(random_rotation) {
    is_trained = false;
    max_points_per_d = 1000;
    balanced_bins = 0;
    epsilon = 0;
}

namespace {

/// Compute the eigenvalue decomposition of symmetric matrix cov,
/// dimensions d_in-by-d_in. Output eigenvectors in cov.

void eig(size_t d_in, double* cov, double* eigenvalues, int verbose) {
    { // compute eigenvalues and vectors
        FINTEGER info = 0, lwork = -1, di = d_in;
        double workq;

        dsyev_("Vectors as well",
               "Upper",
               &di,
               cov,
               &di,
               eigenvalues,
               &workq,
               &lwork,
               &info);
        lwork = FINTEGER(workq);
        double* work = new double[lwork];

        dsyev_("Vectors as well",
               "Upper",
               &di,
               cov,
               &di,
               eigenvalues,
               work,
               &lwork,
               &info);

        delete[] work;

        if (info != 0) {
            fprintf(stderr,
                    "WARN ssyev info returns %d, "
                    "a very bad PCA matrix is learnt\n",
                    int(info));
            // do not throw exception, as the matrix could still be useful
        }

        if (verbose && d_in <= 10) {
            printf("info=%ld new eigvals=[", long(info));
            for (int j = 0; j < d_in; j++)
                printf("%g ", eigenvalues[j]);
            printf("]\n");

            double* ci = cov;
            printf("eigenvecs=\n");
            for (int i = 0; i < d_in; i++) {
                for (int j = 0; j < d_in; j++)
                    printf("%10.4g ", *ci++);
                printf("\n");
            }
        }
    }

    // revert order of eigenvectors & values

    for (int i = 0; i < d_in / 2; i++) {
        std::swap(eigenvalues[i], eigenvalues[d_in - 1 - i]);
        double* v1 = cov + i * d_in;
        double* v2 = cov + (d_in - 1 - i) * d_in;
        for (int j = 0; j < d_in; j++)
            std::swap(v1[j], v2[j]);
    }
}

} // namespace

void PCAMatrix::train(idx_t n, const float* x_in) {
    const float* x = fvecs_maybe_subsample(
            d_in, (size_t*)&n, max_points_per_d * d_in, x_in, verbose);
    TransformedVectors tv(x_in, x);

    // compute mean
    mean.clear();
    mean.resize(d_in, 0.0);
    if (have_bias) { // we may want to skip the bias
        const float* xi = x;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d_in; j++)
                mean[j] += *xi++;
        }
        for (int j = 0; j < d_in; j++)
            mean[j] /= n;
    }
    if (verbose) {
        printf("mean=[");
        for (int j = 0; j < d_in; j++)
            printf("%g ", mean[j]);
        printf("]\n");
    }

    if (n >= d_in) {
        // compute covariance matrix, store it in PCA matrix
        PCAMat.resize(d_in * d_in);
        float* cov = PCAMat.data();
        { // initialize with  mean * mean^T term
            float* ci = cov;
            for (int i = 0; i < d_in; i++) {
                for (int j = 0; j < d_in; j++)
                    *ci++ = -n * mean[i] * mean[j];
            }
        }
        {
            FINTEGER di = d_in, ni = n;
            float one = 1.0;
            ssyrk_("Up",
                   "Non transposed",
                   &di,
                   &ni,
                   &one,
                   (float*)x,
                   &di,
                   &one,
                   cov,
                   &di);
        }
        if (verbose && d_in <= 10) {
            float* ci = cov;
            printf("cov=\n");
            for (int i = 0; i < d_in; i++) {
                for (int j = 0; j < d_in; j++)
                    printf("%10g ", *ci++);
                printf("\n");
            }
        }

        std::vector<double> covd(d_in * d_in);
        for (size_t i = 0; i < d_in * d_in; i++)
            covd[i] = cov[i];

        std::vector<double> eigenvaluesd(d_in);

        eig(d_in, covd.data(), eigenvaluesd.data(), verbose);

        for (size_t i = 0; i < d_in * d_in; i++)
            PCAMat[i] = covd[i];
        eigenvalues.resize(d_in);

        for (size_t i = 0; i < d_in; i++)
            eigenvalues[i] = eigenvaluesd[i];

    } else {
        std::vector<float> xc(n * d_in);

        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < d_in; j++)
                xc[i * d_in + j] = x[i * d_in + j] - mean[j];

        // compute Gram matrix
        std::vector<float> gram(n * n);
        {
            FINTEGER di = d_in, ni = n;
            float one = 1.0, zero = 0.0;
            ssyrk_("Up",
                   "Transposed",
                   &ni,
                   &di,
                   &one,
                   xc.data(),
                   &di,
                   &zero,
                   gram.data(),
                   &ni);
        }

        if (verbose && d_in <= 10) {
            float* ci = gram.data();
            printf("gram=\n");
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++)
                    printf("%10g ", *ci++);
                printf("\n");
            }
        }

        std::vector<double> gramd(n * n);
        for (size_t i = 0; i < n * n; i++)
            gramd[i] = gram[i];

        std::vector<double> eigenvaluesd(n);

        // eig will fill in only the n first eigenvals

        eig(n, gramd.data(), eigenvaluesd.data(), verbose);

        PCAMat.resize(d_in * n);

        for (size_t i = 0; i < n * n; i++)
            gram[i] = gramd[i];

        eigenvalues.resize(d_in);
        // fill in only the n first ones
        for (size_t i = 0; i < n; i++)
            eigenvalues[i] = eigenvaluesd[i];

        { // compute PCAMat = x' * v
            FINTEGER di = d_in, ni = n;
            float one = 1.0;

            sgemm_("Non",
                   "Non Trans",
                   &di,
                   &ni,
                   &ni,
                   &one,
                   xc.data(),
                   &di,
                   gram.data(),
                   &ni,
                   &one,
                   PCAMat.data(),
                   &di);
        }

        if (verbose && d_in <= 10) {
            float* ci = PCAMat.data();
            printf("PCAMat=\n");
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < d_in; j++)
                    printf("%10g ", *ci++);
                printf("\n");
            }
        }
        fvec_renorm_L2(d_in, n, PCAMat.data());
    }

    prepare_Ab();
    is_trained = true;
}

void PCAMatrix::copy_from(const PCAMatrix& other) {
    FAISS_THROW_IF_NOT(other.is_trained);
    mean = other.mean;
    eigenvalues = other.eigenvalues;
    PCAMat = other.PCAMat;
    prepare_Ab();
    is_trained = true;
}

void PCAMatrix::prepare_Ab() {
    FAISS_THROW_IF_NOT_FMT(
            d_out * d_in <= PCAMat.size(),
            "PCA matrix cannot output %d dimensions from %d ",
            d_out,
            d_in);

    if (!random_rotation) {
        A = PCAMat;
        A.resize(d_out * d_in); // strip off useless dimensions

        // first scale the components
        if (eigen_power != 0) {
            float* ai = A.data();
            for (int i = 0; i < d_out; i++) {
                float factor = pow(eigenvalues[i] + epsilon, eigen_power);
                for (int j = 0; j < d_in; j++)
                    *ai++ *= factor;
            }
        }

        if (balanced_bins != 0) {
            FAISS_THROW_IF_NOT(d_out % balanced_bins == 0);
            int dsub = d_out / balanced_bins;
            std::vector<float> Ain;
            std::swap(A, Ain);
            A.resize(d_out * d_in);

            std::vector<float> accu(balanced_bins);
            std::vector<int> counter(balanced_bins);

            // greedy assignment
            for (int i = 0; i < d_out; i++) {
                // find best bin
                int best_j = -1;
                float min_w = 1e30;
                for (int j = 0; j < balanced_bins; j++) {
                    if (counter[j] < dsub && accu[j] < min_w) {
                        min_w = accu[j];
                        best_j = j;
                    }
                }
                int row_dst = best_j * dsub + counter[best_j];
                accu[best_j] += eigenvalues[i];
                counter[best_j]++;
                memcpy(&A[row_dst * d_in], &Ain[i * d_in], d_in * sizeof(A[0]));
            }

            if (verbose) {
                printf("  bin accu=[");
                for (int i = 0; i < balanced_bins; i++)
                    printf("%g ", accu[i]);
                printf("]\n");
            }
        }

    } else {
        FAISS_THROW_IF_NOT_MSG(
                balanced_bins == 0,
                "both balancing bins and applying a random rotation "
                "does not make sense");
        RandomRotationMatrix rr(d_out, d_out);

        rr.init(5);

        // apply scaling on the rotation matrix (right multiplication)
        if (eigen_power != 0) {
            for (int i = 0; i < d_out; i++) {
                float factor = pow(eigenvalues[i], eigen_power);
                for (int j = 0; j < d_out; j++)
                    rr.A[j * d_out + i] *= factor;
            }
        }

        A.resize(d_in * d_out);
        {
            FINTEGER dii = d_in, doo = d_out;
            float one = 1.0, zero = 0.0;

            sgemm_("Not",
                   "Not",
                   &dii,
                   &doo,
                   &doo,
                   &one,
                   PCAMat.data(),
                   &dii,
                   rr.A.data(),
                   &doo,
                   &zero,
                   A.data(),
                   &dii);
        }
    }

    b.clear();
    b.resize(d_out);

    for (int i = 0; i < d_out; i++) {
        float accu = 0;
        for (int j = 0; j < d_in; j++)
            accu -= mean[j] * A[j + i * d_in];
        b[i] = accu;
    }

    is_orthonormal = eigen_power == 0;
}

/*********************************************
 * ITQMatrix
 *********************************************/

ITQMatrix::ITQMatrix(int d)
        : LinearTransform(d, d, false), max_iter(50), seed(123) {}

/** translated from fbcode/deeplearning/catalyzer/catalyzer/quantizers.py */
void ITQMatrix::train(idx_t n, const float* xf) {
    size_t d = d_in;
    std::vector<double> rotation(d * d);

    if (init_rotation.size() == d * d) {
        memcpy(rotation.data(),
               init_rotation.data(),
               d * d * sizeof(rotation[0]));
    } else {
        RandomRotationMatrix rrot(d, d);
        rrot.init(seed);
        for (size_t i = 0; i < d * d; i++) {
            rotation[i] = rrot.A[i];
        }
    }

    std::vector<double> x(n * d);

    for (size_t i = 0; i < n * d; i++) {
        x[i] = xf[i];
    }

    std::vector<double> rotated_x(n * d), cov_mat(d * d);
    std::vector<double> u(d * d), vt(d * d), singvals(d);

    for (int i = 0; i < max_iter; i++) {
        print_if_verbose("rotation", rotation, d, d);
        { // rotated_data = np.dot(training_data, rotation)
            FINTEGER di = d, ni = n;
            double one = 1, zero = 0;
            dgemm_("N",
                   "N",
                   &di,
                   &ni,
                   &di,
                   &one,
                   rotation.data(),
                   &di,
                   x.data(),
                   &di,
                   &zero,
                   rotated_x.data(),
                   &di);
        }
        print_if_verbose("rotated_x", rotated_x, n, d);
        // binarize
        for (size_t j = 0; j < n * d; j++) {
            rotated_x[j] = rotated_x[j] < 0 ? -1 : 1;
        }
        // covariance matrix
        { // rotated_data = np.dot(training_data, rotation)
            FINTEGER di = d, ni = n;
            double one = 1, zero = 0;
            dgemm_("N",
                   "T",
                   &di,
                   &di,
                   &ni,
                   &one,
                   rotated_x.data(),
                   &di,
                   x.data(),
                   &di,
                   &zero,
                   cov_mat.data(),
                   &di);
        }
        print_if_verbose("cov_mat", cov_mat, d, d);
        // SVD
        {
            FINTEGER di = d;
            FINTEGER lwork = -1, info;
            double lwork1;

            // workspace query
            dgesvd_("A",
                    "A",
                    &di,
                    &di,
                    cov_mat.data(),
                    &di,
                    singvals.data(),
                    u.data(),
                    &di,
                    vt.data(),
                    &di,
                    &lwork1,
                    &lwork,
                    &info);

            FAISS_THROW_IF_NOT(info == 0);
            lwork = size_t(lwork1);
            std::vector<double> work(lwork);
            dgesvd_("A",
                    "A",
                    &di,
                    &di,
                    cov_mat.data(),
                    &di,
                    singvals.data(),
                    u.data(),
                    &di,
                    vt.data(),
                    &di,
                    work.data(),
                    &lwork,
                    &info);
            FAISS_THROW_IF_NOT_FMT(info == 0, "sgesvd returned info=%d", info);
        }
        print_if_verbose("u", u, d, d);
        print_if_verbose("vt", vt, d, d);
        // update rotation
        {
            FINTEGER di = d;
            double one = 1, zero = 0;
            dgemm_("N",
                   "T",
                   &di,
                   &di,
                   &di,
                   &one,
                   u.data(),
                   &di,
                   vt.data(),
                   &di,
                   &zero,
                   rotation.data(),
                   &di);
        }
        print_if_verbose("final rot", rotation, d, d);
    }
    A.resize(d * d);
    for (size_t i = 0; i < d; i++) {
        for (size_t j = 0; j < d; j++) {
            A[i + d * j] = rotation[j + d * i];
        }
    }
    is_trained = true;
}

ITQTransform::ITQTransform(int d_in, int d_out, bool do_pca)
        : VectorTransform(d_in, d_out),
          do_pca(do_pca),
          itq(d_out),
          pca_then_itq(d_in, d_out, false) {
    if (!do_pca) {
        FAISS_THROW_IF_NOT(d_in == d_out);
    }
    max_train_per_dim = 10;
    is_trained = false;
}

void ITQTransform::train(idx_t n, const float* x_in) {
    FAISS_THROW_IF_NOT(!is_trained);

    size_t max_train_points = std::max(d_in * max_train_per_dim, 32768);
    const float* x =
            fvecs_maybe_subsample(d_in, (size_t*)&n, max_train_points, x_in);
    TransformedVectors tv(x_in, x);

    std::unique_ptr<float[]> x_norm(new float[n * d_in]);
    { // normalize
        int d = d_in;

        mean.resize(d, 0);
        for (idx_t i = 0; i < n; i++) {
            for (idx_t j = 0; j < d; j++) {
                mean[j] += x[i * d + j];
            }
        }
        for (idx_t j = 0; j < d; j++) {
            mean[j] /= n;
        }
        for (idx_t i = 0; i < n; i++) {
            for (idx_t j = 0; j < d; j++) {
                x_norm[i * d + j] = x[i * d + j] - mean[j];
            }
        }
        fvec_renorm_L2(d_in, n, x_norm.get());
    }

    // train PCA

    PCAMatrix pca(d_in, d_out);
    float* x_pca;
    std::unique_ptr<float[]> x_pca_del;
    if (do_pca) {
        pca.have_bias = false; // for consistency with reference implem
        pca.train(n, x_norm.get());
        x_pca = pca.apply(n, x_norm.get());
        x_pca_del.reset(x_pca);
    } else {
        x_pca = x_norm.get();
    }

    // train ITQ
    itq.train(n, x_pca);

    // merge PCA and ITQ
    if (do_pca) {
        FINTEGER di = d_out, dini = d_in;
        float one = 1, zero = 0;
        pca_then_itq.A.resize(d_in * d_out);
        sgemm_("N",
               "N",
               &dini,
               &di,
               &di,
               &one,
               pca.A.data(),
               &dini,
               itq.A.data(),
               &di,
               &zero,
               pca_then_itq.A.data(),
               &dini);
    } else {
        pca_then_itq.A = itq.A;
    }
    pca_then_itq.is_trained = true;
    is_trained = true;
}

void ITQTransform::apply_noalloc(idx_t n, const float* x, float* xt) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "Transformation not trained yet");

    std::unique_ptr<float[]> x_norm(new float[n * d_in]);
    { // normalize
        int d = d_in;
        for (idx_t i = 0; i < n; i++) {
            for (idx_t j = 0; j < d; j++) {
                x_norm[i * d + j] = x[i * d + j] - mean[j];
            }
        }
        // this is not really useful if we are going to binarize right
        // afterwards but OK
        fvec_renorm_L2(d_in, n, x_norm.get());
    }

    pca_then_itq.apply_noalloc(n, x_norm.get(), xt);
}

void ITQTransform::check_identical(const VectorTransform& other_in) const {
    VectorTransform::check_identical(other_in);
    auto other = dynamic_cast<const ITQTransform*>(&other_in);
    FAISS_THROW_IF_NOT(other);
    pca_then_itq.check_identical(other->pca_then_itq);
    FAISS_THROW_IF_NOT(other->mean == mean);
}

/*********************************************
 * OPQMatrix
 *********************************************/

OPQMatrix::OPQMatrix(int d, int M, int d2)
        : LinearTransform(d, d2 == -1 ? d : d2, false), M(M) {
    is_trained = false;
    // OPQ is quite expensive to train, so set this right.
    max_train_points = 256 * 256;
}

/*********************************************
 * NormalizationTransform
 *********************************************/

NormalizationTransform::NormalizationTransform(int d, float norm)
        : VectorTransform(d, d), norm(norm) {}

NormalizationTransform::NormalizationTransform()
        : VectorTransform(-1, -1), norm(-1) {}

void NormalizationTransform::apply_noalloc(idx_t n, const float* x, float* xt)
        const {
    if (norm == 2.0) {
        memcpy(xt, x, sizeof(x[0]) * n * d_in);
        fvec_renorm_L2(d_in, n, xt);
    } else {
        FAISS_THROW_MSG("not implemented");
    }
}

void NormalizationTransform::reverse_transform(
        idx_t n,
        const float* xt,
        float* x) const {
    memcpy(x, xt, sizeof(xt[0]) * n * d_in);
}

void NormalizationTransform::check_identical(
        const VectorTransform& other_in) const {
    VectorTransform::check_identical(other_in);
    auto other = dynamic_cast<const NormalizationTransform*>(&other_in);
    FAISS_THROW_IF_NOT(other);
    FAISS_THROW_IF_NOT(other->norm == norm);
}

/*********************************************
 * CenteringTransform
 *********************************************/

CenteringTransform::CenteringTransform(int d) : VectorTransform(d, d) {
    is_trained = false;
}

void CenteringTransform::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(n > 0, "need at least one training vector");
    mean.resize(d_in, 0);
    for (idx_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d_in; j++) {
            mean[j] += *x++;
        }
    }

    for (size_t j = 0; j < d_in; j++) {
        mean[j] /= n;
    }
    is_trained = true;
}

void CenteringTransform::apply_noalloc(idx_t n, const float* x, float* xt)
        const {
    FAISS_THROW_IF_NOT(is_trained);

    for (idx_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d_in; j++) {
            *xt++ = *x++ - mean[j];
        }
    }
}

void CenteringTransform::reverse_transform(idx_t n, const float* xt, float* x)
        const {
    FAISS_THROW_IF_NOT(is_trained);

    for (idx_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d_in; j++) {
            *x++ = *xt++ + mean[j];
        }
    }
}

void CenteringTransform::check_identical(
        const VectorTransform& other_in) const {
    VectorTransform::check_identical(other_in);
    auto other = dynamic_cast<const CenteringTransform*>(&other_in);
    FAISS_THROW_IF_NOT(other);
    FAISS_THROW_IF_NOT(other->mean == mean);
}

/*********************************************
 * RemapDimensionsTransform
 *********************************************/

RemapDimensionsTransform::RemapDimensionsTransform(
        int d_in,
        int d_out,
        const int* map_in)
        : VectorTransform(d_in, d_out) {
    map.resize(d_out);
    for (int i = 0; i < d_out; i++) {
        map[i] = map_in[i];
        FAISS_THROW_IF_NOT(map[i] == -1 || (map[i] >= 0 && map[i] < d_in));
    }
}

RemapDimensionsTransform::RemapDimensionsTransform(
        int d_in,
        int d_out,
        bool uniform)
        : VectorTransform(d_in, d_out) {
    map.resize(d_out, -1);

    if (uniform) {
        if (d_in < d_out) {
            for (int i = 0; i < d_in; i++) {
                map[i * d_out / d_in] = i;
            }
        } else {
            for (int i = 0; i < d_out; i++) {
                map[i] = i * d_in / d_out;
            }
        }
    } else {
        for (int i = 0; i < d_in && i < d_out; i++)
            map[i] = i;
    }
}

void RemapDimensionsTransform::apply_noalloc(idx_t n, const float* x, float* xt)
        const {
    for (idx_t i = 0; i < n; i++) {
        for (int j = 0; j < d_out; j++) {
            xt[j] = map[j] < 0 ? 0 : x[map[j]];
        }
        x += d_in;
        xt += d_out;
    }
}

void RemapDimensionsTransform::reverse_transform(
        idx_t n,
        const float* xt,
        float* x) const {
    memset(x, 0, sizeof(*x) * n * d_in);
    for (idx_t i = 0; i < n; i++) {
        for (int j = 0; j < d_out; j++) {
            if (map[j] >= 0)
                x[map[j]] = xt[j];
        }
        x += d_in;
        xt += d_out;
    }
}

void RemapDimensionsTransform::check_identical(
        const VectorTransform& other_in) const {
    VectorTransform::check_identical(other_in);
    auto other = dynamic_cast<const RemapDimensionsTransform*>(&other_in);
    FAISS_THROW_IF_NOT(other);
    FAISS_THROW_IF_NOT(other->map == map);
}
