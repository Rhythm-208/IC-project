#ifndef MAT_H_
#define MAT_H_

#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#ifndef NN_MALLOC
#define NN_MALLOC malloc
#endif

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif

#define ARRAY_LEN(xs) sizeof((xs))/sizeof((xs)[0])

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Mat;

#define MAT_AT(m, i, j) (m).es[(i) * (m).stride + (j)]

float rand_float(void) {
    return (float)rand() / (float)RAND_MAX;
}

Mat mat_alloc(size_t rows, size_t cols) {
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = (float*)NN_MALLOC(sizeof(float) * rows * cols);
    NN_ASSERT(m.es != NULL);
    return m;
}

void mat_fill(Mat m, float val) {
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = val;
        }
    }
}

void mat_rand(Mat m, float low, float high) {
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = rand_float() * (high - low) + low;
        }
    }
}

void mat_dot(Mat dst, Mat a, Mat b) {
    NN_ASSERT(a.cols == b.rows);
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == b.cols);

    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < a.cols; ++k) {
                sum += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
            MAT_AT(dst, i, j) = sum;
        }
    }
}

void mat_add(Mat dst, Mat a) {
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == a.cols);
    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}

float sigmoidf(float x) {
    return 1.f / (1.f + expf(-x));
}

void mat_sig(Mat m) {
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}

#endif // MAT_H_
