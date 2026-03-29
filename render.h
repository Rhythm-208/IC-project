#ifndef RENDER_H_
#define RENDER_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "nn.h"
#include "stb_image_write.h"

void render_upscaled(NN nn, const char *out_path, int out_width, int out_height) {
    uint8_t *buffer = (uint8_t*)malloc(out_width * out_height);
    
    if(buffer == NULL) {
        fprintf(stderr, "Failed to allocate memory for render buffer\n");
        exit(1);
    }
    
    printf("Rendering to %dx%d image...\n", out_width, out_height);
    
    for (int y = 0; y < out_height; ++y) {
        for (int x = 0; x < out_width; ++x) {
            
            float nx = (float)x / (out_width - 1);
            float ny = (float)y / (out_height - 1);
            
            MAT_AT(nn.as[0], 0, 0) = nx;
            MAT_AT(nn.as[0], 0, 1) = ny;
            
            nn_forward(nn);
            
            float output_activation = MAT_AT(nn.as[nn.count], 0, 0);
            
            uint8_t pixel = (uint8_t)(output_activation * 255.0f);
            
            buffer[y * out_width + x] = pixel;
        }
    }
    
    if (stbi_write_png(out_path, out_width, out_height, 1, buffer, out_width)) {
        printf("Successfully rendered upscaled mapping to '%s'!\n", out_path);
    } else {
        fprintf(stderr, "Failed to write output %s\n", out_path);
    }
    
    free(buffer);
}

#endif