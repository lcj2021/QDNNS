#include <algorithm>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include "knncuda.h"
#include "distance.h"
#include <omp.h>

/**
 * Initializes randomly the reference and query points.
 *
 * @param ref        refence points
 * @param ref_nb     number of reference points
 * @param query      query points
 * @param query_nb   number of query points
 * @param dim        dimension of points
 */
void initialize_data(float * ref,
                     int     ref_nb,
                     float * query,
                     int     query_nb,
                     int     dim) {

    // Initialize random number generator
    srand(time(NULL));

    // Generate random reference points
    for (int i=0; i<ref_nb*dim; ++i) {
        ref[i] = 1e6 * (float)(rand() / (double)RAND_MAX);
    }

    // Generate random query points
    for (int i=0; i<query_nb*dim; ++i) {
        query[i] = 1e6 * (float)(rand() / (double)RAND_MAX);
    }
}


/**
 * Computes the Euclidean distance between a reference point and a query point.
 *
 * @param ref          refence points
 * @param ref_nb       number of reference points
 * @param query        query points
 * @param query_nb     number of query points
 * @param dim          dimension of points
 * @param ref_index    index to the reference point to consider
 * @param query_index  index to the query point to consider
 * @return computed distance
 */
template<float (*metric)(float, float)>
float compute_distance(const float * ref,
                       int           ref_nb,
                       const float * query,
                       int           query_nb,
                       int           dim,
                       int           ref_index,
                       int           query_index) {
    double sum = 0.f;
    for (int d=0; d<dim; ++d) {
        sum += metric(ref[d * ref_nb + ref_index], query[d * query_nb + query_index]);
    }
    return sum;
}

/*
 * For each input query point, locates the k-NN (indexes and distances) among the reference points.
 *
 * @param ref        refence points
 * @param ref_nb     number of reference points
 * @param query      query points
 * @param query_nb   number of query points
 * @param dim        dimension of points
 * @param k          number of neighbors to consider
 * @param knn_dist   output array containing the query_nb x k distances
 * @param knn_index  output array containing the query_nb x k indexes
 */
template<float (*metric)(float, float)>
bool knn_c(const float * ref,
           size_t           ref_nb,
           const float * query,
           size_t           query_nb,
           int           dim,
           int           k,
           float *       knn_dist,
           int *         knn_index) {

    // Process one query point at the time
#pragma omp parallel for num_threads(24)
    for (int i=0; i<query_nb; ++i) {
        std::vector<std::pair<float, int>> pair(ref_nb);
        for (int j=0; j<ref_nb; ++j) {
            pair[j].first  = compute_distance<metric>(ref, ref_nb, query, query_nb, dim, j, i);
            pair[j].second = j;
        }

        // Sort distances / indexes
        // modified_insertion_sort<lt>(dist.data(), index.data(), ref_nb, k);
        std::partial_sort(pair.begin(), pair.begin() + k, pair.end(), [] (auto &a, auto &b) { 
        // std::sort(pair.begin(), pair.end(), [] (auto &a, auto &b) { 
            if (a.first !=  b.first) return a.first < b.first;
            return (a).second < (b).second;
        });

        for (int j=0; j<k; ++j) {
            knn_dist[j * query_nb + i]  = pair[j].first;
            knn_index[j * query_nb + i] = pair[j].second;
        }

        if (rand() % 100 < 1) {
            std::cerr << i << std::endl;
        }
    }

    return true;

}


/**
 * Test an input k-NN function implementation by verifying that its output
 * results (distances and corresponding indexes) are similar to the expected
 * results (ground truth).
 *
 * Since the k-NN computation might end-up in slightly different results
 * compared to the expected one depending on the considered implementation,
 * the verification consists in making sure that the accuracy is high enough.
 *
 * The tested function is ran several times in order to have a better estimate
 * of the processing time.
 *
 * @param ref            reference points
 * @param ref_nb         number of reference points
 * @param query          query points
 * @param query_nb       number of query points
 * @param dim            dimension of reference and query points
 * @param k              number of neighbors to consider
 * @param gt_knn_dist    ground truth distances
 * @param gt_knn_index   ground truth indexes
 * @param knn            function to test
 * @param name           name of the function to test (for display purpose)
 * @param nb_iterations  number of iterations
 * return false in case of problem, true otherwise
 */
bool test(const float * ref,
          int           ref_nb,
          const float * query,
          int           query_nb,
          int           dim,
          int           k,
          float *       gt_knn_dist,
          int *         gt_knn_index,
          bool (*knn)(const float *, int, const float *, int, int, int, float *, int *),
          const char *  name,
          int           nb_iterations) {

    // Parameters
    const float precision    = 1e-6; // distance error max
    const float min_accuracy = 1-1e-6; // percentage of correct values required

    // Display k-NN function name
    printf("- %-17s : \n", name);

    // Allocate memory for computed k-NN neighbors
    std::vector<float> test_knn_dist(query_nb * k);
    std::vector<int> test_knn_index(query_nb * k);

    // Start timer
    struct timeval tic;
    gettimeofday(&tic, NULL);

    // Compute k-NN several times
    for (int i=0; i<nb_iterations; ++i) {
        if (!knn(ref, ref_nb, query, query_nb, dim, k, test_knn_dist.data(), test_knn_index.data())) {
            return false;
        }
    }

    // Stop timer
    struct timeval toc;
    gettimeofday(&toc, NULL);

    // Elapsed time in ms
    double elapsed_time = toc.tv_sec - tic.tv_sec;
    elapsed_time += (toc.tv_usec - tic.tv_usec) / 1000000.;

    // Verify both precisions and indexes of the k-NN values
    int nb_correct_precisions = 0;
    int nb_correct_indexes    = 0;
    for (int i=0; i<query_nb*k; ++i) {
        if (fabs(test_knn_dist[i] - gt_knn_dist[i]) <= precision) {
            nb_correct_precisions++;
        }
        if (test_knn_index[i] == gt_knn_index[i]) {
            nb_correct_indexes++;
        }
    }

    // Compute accuracy
    float precision_accuracy = nb_correct_precisions / ((float) query_nb * k);
    float index_accuracy     = nb_correct_indexes    / ((float) query_nb * k);
    std::cout << "precision_accuracy: " << precision_accuracy << std::endl;
    std::cout << "index_accuracy: " << index_accuracy << std::endl;

    // Display report
    if (precision_accuracy >= min_accuracy && index_accuracy >= min_accuracy ) {
        printf("SUCCESS\n");
    }
    else {
        printf("FAILED\n");
    }
    printf("Elapsed in %8.5f seconds (averaged over %3d iterations)\n", elapsed_time / nb_iterations, nb_iterations);

    return true;
}


/**
 * 1. Create the synthetic data (reference and query points).
 * 2. Compute the ground truth.
 * 3. Test the different implementation of the k-NN algorithm.
 */
int main(void) {

    // Parameters
    const size_t ref_nb   = 1'000'000;
    const size_t query_nb = 32;
    const int dim      = 1024;
    const int k        = 100;

    // Display
    printf("PARAMETERS\n");
    printf("- Number reference points : %ld\n",   ref_nb);
    printf("- Number query points     : %ld\n",   query_nb);
    printf("- Dimension of points     : %d\n",   dim);
    printf("- Number of neighbors     : %d\n\n", k);

    // Sanity check
    if (ref_nb<k) {
        printf("Error: k value is larger that the number of reference points\n");
        return EXIT_FAILURE;
    }

    // Allocate input points and output k-NN distances / indexes
    float * ref        = (float*) malloc(ref_nb   * dim * sizeof(float));
    float * query      = (float*) malloc(query_nb * dim * sizeof(float));
    float * knn_dist   = (float*) malloc(query_nb * k   * sizeof(float));
    int   * knn_index  = (int*)   malloc(query_nb * k   * sizeof(int));
    size_t total_memory = (size_t)ref_nb * dim * sizeof(float) + query_nb * dim * sizeof(float) + query_nb * k * sizeof(float) + query_nb * k * sizeof(int);
    std::cout << "Total memory allocated: " << (total_memory) / 1024.0 / 1024.0 / 1024.0 << " GB" << std::endl;

    // Allocation checks
    if (!ref || !query || !knn_dist || !knn_index) {
        printf("Error: Memory allocation error\n"); 
        free(ref);
	    free(query);
	    free(knn_dist);
	    free(knn_index);
        return EXIT_FAILURE;
    }

    // Initialize reference and query points with random values
    initialize_data(ref, ref_nb, query, query_nb, dim);

    // Compute the ground truth k-NN distances and indexes for each query point
    printf("Ground truth computation in progress...\n\n");
    if (!knn_c<L2>(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index)) {
        free(ref);
	    free(query);
	    free(knn_dist);
	    free(knn_index);
        return EXIT_FAILURE;
    }

    // Test all k-NN functions
    printf("TESTS\n");
    // test(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index, &knn_c,            "knn_c",              2);
    test(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index, &knn_cuda_global,  "knn_cuda_global",  1); 
    test(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index, &knn_cublas,       "knn_cublas",       100); 

    // Deallocate memory 
    free(ref);
    free(query);
    free(knn_dist);
    free(knn_index);

    return EXIT_SUCCESS;
}
