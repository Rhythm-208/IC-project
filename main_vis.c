/* ==========================================================================
   Neural Network Training Visualizer
   main_vis.c — real-time Win32 GDI visualization of NN learning

   Build (MinGW):
     gcc -o nn_vis.exe main_vis.c -lgdi32 -lm -O2 -mwindows
   Run:
     nn_vis.exe
   ========================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include "nn.h"

#define STB_IMAGE_IMPLEMENTATION
#include "data.h"

#include "visualizer.h"

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    srand((unsigned int)time(NULL));

    /* ── load training data ─────────────────────────────────────────── */
    Mat t, ti, to;
    load_training_data("input.png", &t, &ti, &to);

    /* ── create network ─────────────────────────────────────────────── */
    size_t arch[] = {2, 7, 4, 1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g  = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, -1.0f, 1.0f);

    size_t total_epochs = 100000;
    size_t epoch = 0;

    /* ── fire up the visualizer ─────────────────────────────────────── */
    vis_init(1.0f);

    /* ── main loop: train + render ──────────────────────────────────── */
    while (!vis_should_close()) {
        vis_process_events();

        /* handle reset request */
        if (g_vis.needs_reset) {
            nn_rand(nn, -1.0f, 1.0f);
            epoch = 0;
            g_vis.needs_reset = 0;
        }

        /* train N epochs per frame if not paused */
        if (!g_vis.paused && epoch < total_epochs) {
            for (int i = 0; i < g_vis.epf && epoch < total_epochs; i++, epoch++) {
                nn_backprop(nn, g, ti, to);
                nn_learn(nn, g, g_vis.rate);
            }

            float cost = nn_cost(nn, ti, to);
            g_vis.epoch = epoch;
            vis_record_cost(cost);

            if (epoch >= total_epochs) {
                g_vis.done = 1;
                printf("Training complete! Final cost: %f\n", cost);
            }
        }

        /* render everything */
        vis_render(nn);

        /* throttle to ~60fps */
        Sleep(16);
    }

    vis_close();
    free(t.es);
    return 0;
}
