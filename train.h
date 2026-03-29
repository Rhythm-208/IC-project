#ifndef TRAIN_H_
#define TRAIN_H_

#include <stdio.h>
#include "nn.h"

void train_model(NN nn, NN g, Mat ti, Mat to, size_t epochs, float rate) {
    printf("Training for %zu epochs...\n", epochs);
    for (size_t i = 0; i < epochs; ++i) {
        nn_backprop(nn, g, ti, to);
        nn_learn(nn, g, rate);
        
        if (i % 1000 == 0) {
            float cost = nn_cost(nn, ti, to);
            printf("Epoch: %8zu \t Cost: %f\n", i, cost);
        }
    }
}

#endif
