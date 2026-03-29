#ifndef NN_H_
#define NN_H_

#include "mat.h"

typedef struct {
    size_t count;
    Mat *ws;
    Mat *bs;
    Mat *as;    
} NN;

NN nn_alloc(size_t *arch, size_t arch_count) {
    NN_ASSERT(arch_count > 0);
    
    NN nn;
    nn.count = arch_count - 1;
    
    nn.ws = (Mat*)NN_MALLOC(sizeof(Mat) * nn.count);
    NN_ASSERT(nn.ws != NULL);
    nn.bs = (Mat*)NN_MALLOC(sizeof(Mat) * nn.count);
    NN_ASSERT(nn.bs != NULL);
    nn.as = (Mat*)NN_MALLOC(sizeof(Mat) * (nn.count + 1));
    NN_ASSERT(nn.as != NULL);
    
    nn.as[0] = mat_alloc(1, arch[0]);
    
    for(size_t i = 1; i < arch_count; ++i) {
        nn.ws[i-1] = mat_alloc(nn.as[i-1].cols, arch[i]);
        nn.bs[i-1] = mat_alloc(1, arch[i]);
        nn.as[i]   = mat_alloc(1, arch[i]);
    }
    
    return nn;
}

void nn_rand(NN nn, float low, float high) {
    for (size_t i = 0; i < nn.count; ++i) {
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}

void nn_forward(NN nn) {
    for (size_t i = 0; i < nn.count; ++i) {
        mat_dot(nn.as[i+1], nn.as[i], nn.ws[i]);
        mat_add(nn.as[i+1], nn.bs[i]);
        mat_sig(nn.as[i+1]);
    }
}

float nn_cost(NN nn, Mat ti, Mat to) {
    NN_ASSERT(ti.rows == to.rows);
    NN_ASSERT(to.cols == nn.as[nn.count].cols);
    size_t n = ti.rows;
    
    float cost = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        Mat x = {
            .rows = 1,
            .cols = ti.cols,
            .stride = ti.stride,
            .es = &MAT_AT(ti, i, 0)
        };
        
        Mat y = {
            .rows = 1,
            .cols = to.cols,
            .stride = to.stride,
            .es = &MAT_AT(to, i, 0)
        };
        
        for (size_t j = 0; j < nn.as[0].cols; ++j) {
            MAT_AT(nn.as[0], 0, j) = MAT_AT(x, 0, j);
        }
        
        nn_forward(nn);
        
        for (size_t j = 0; j < to.cols; ++j) {
            float d = MAT_AT(nn.as[nn.count], 0, j) - MAT_AT(y, 0, j);
            cost += d*d;
        }
    }
    
    return cost / n; // Return mean
}

void nn_backprop(NN nn, NN g, Mat ti, Mat to) {
    NN_ASSERT(ti.rows == to.rows);
    NN_ASSERT(to.cols == nn.as[nn.count].cols);
    size_t n = ti.rows;
    
    for (size_t i = 0; i < g.count; ++i) {
        mat_fill(g.ws[i], 0.0f);
        mat_fill(g.bs[i], 0.0f);
        mat_fill(g.as[i], 0.0f);
    }
    mat_fill(g.as[g.count], 0.0f);
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < nn.as[0].cols; ++j) {
            MAT_AT(nn.as[0], 0, j) = MAT_AT(ti, i, j);
        }
        nn_forward(nn);
        
        for (size_t j = 0; j <= nn.count; ++j) {
            mat_fill(g.as[j], 0.f);
        }

        for (size_t j = 0; j < to.cols; ++j) {
            MAT_AT(g.as[nn.count], 0, j) = MAT_AT(nn.as[nn.count], 0, j) - MAT_AT(to, i, j);
            MAT_AT(g.as[nn.count], 0, j) *= 2.0f; 
        }
        
        for (size_t l = nn.count; l > 0; --l) {
            for (size_t j = 0; j < nn.as[l].cols; ++j) {
                float a = MAT_AT(nn.as[l], 0, j);
                float da = MAT_AT(g.as[l], 0, j);
                
                float qa = da * a * (1 - a); 
                
                MAT_AT(g.bs[l-1], 0, j) += qa;
                
                for (size_t k = 0; k < nn.as[l-1].cols; ++k) {
                    float pa = MAT_AT(nn.as[l-1], 0, k); 
                    float w = MAT_AT(nn.ws[l-1], k, j);
                    
                    MAT_AT(g.ws[l-1], k, j) += qa * pa;
                    MAT_AT(g.as[l-1], 0, k) += qa * w;
                }
            }
        }
    }
    
    for (size_t i = 0; i < g.count; ++i) {
        for (size_t j = 0; j < g.ws[i].rows; ++j) {
            for (size_t k = 0; k < g.ws[i].cols; ++k) {
                MAT_AT(g.ws[i], j, k) /= n;
            }
        }
        for (size_t j = 0; j < g.bs[i].rows; ++j) {
            for (size_t k = 0; k < g.bs[i].cols; ++k) {
                MAT_AT(g.bs[i], j, k) /= n;
            }
        }
    }
}

void nn_learn(NN nn, NN g, float rate) {
    for (size_t i = 0; i < nn.count; ++i) {
        for (size_t j = 0; j < nn.ws[i].rows; ++j) {
            for (size_t k = 0; k < nn.ws[i].cols; ++k) {
                MAT_AT(nn.ws[i], j, k) -= rate * MAT_AT(g.ws[i], j, k);
            }
        }
        for (size_t j = 0; j < nn.bs[i].rows; ++j) {
            for (size_t k = 0; k < nn.bs[i].cols; ++k) {
                MAT_AT(nn.bs[i], j, k) -= rate * MAT_AT(g.bs[i], j, k);
            }
        }
    }
}

#endif // NN_H_
