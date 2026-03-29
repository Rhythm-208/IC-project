#ifndef DATA_H_
#define DATA_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "nn.h"
#include "stb_image.h"

void load_training_data(const char *img_path, Mat *t, Mat *ti, Mat *to) {
    int img_x, img_y, img_comp;
    uint8_t *img_data = stbi_load(img_path, &img_x, &img_y, &img_comp, 1);
    
    if (img_data == NULL) {
        fprintf(stderr, "Failed to load %s. Please make sure the file exists and is a valid image!\n", img_path);
        exit(1);
    }
    
    printf("Successfully loaded %s: %dx%d pixels (1 channel)\n", img_path, img_x, img_y);

    *t = mat_alloc(img_x * img_y, 3);
    
    for (int y = 0; y < img_y; ++y) {
        for (int x = 0; x < img_x; ++x) {
            size_t i = y * img_x + x; 
            
            float nx = (float)x / (img_x - 1);
            float ny = (float)y / (img_y - 1);
            
            uint8_t c = img_data[i];
            float nb = (float)c / 255.0f;
            
            MAT_AT(*t, i, 0) = nx;
            MAT_AT(*t, i, 1) = ny;
            MAT_AT(*t, i, 2) = nb;
        }
    }
    
    stbi_image_free(img_data);
    
    *ti = (Mat){
        .rows = t->rows,
        .cols = 2,        
        .stride = t->stride,
        .es = &MAT_AT(*t, 0, 0)
    };
    
    *to = (Mat){
        .rows = t->rows,
        .cols = 1,         
        .stride = t->stride, 
        .es = &MAT_AT(*t, 0, 2) 
    };
}

#endif // DATA_H_
