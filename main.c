#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "nn.h"

#define STB_IMAGE_IMPLEMENTATION
#include "data.h"

#include "train.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "render.h"

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    
    Mat t, ti, to;
    load_training_data("input.png", &t, &ti, &to);

    size_t arch[] = {2, 7, 4, 1}; 
    NN nn = nn_alloc(arch, ARRAY_LEN(arch)); 
    NN g = nn_alloc(arch, ARRAY_LEN(arch));
    
    nn_rand(nn, -1.0f, 1.0f);
    train_model(nn, g, ti, to, 100000, 1.0f);

    render_upscaled(nn, "upscaled.png", 36, 36);

    free(t.es);
    return 0;
}
