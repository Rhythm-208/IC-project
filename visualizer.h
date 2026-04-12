#ifndef VISUALIZER_H_
#define VISUALIZER_H_

/*  =====================================================================
    Neural Network Training Visualizer — Pure Win32 GDI (no dependencies)
    =====================================================================
    Panels:
      Top-Left:     Loss / Cost curve (auto-scaling)
      Top-Right:    Live NN output image reconstruction
      Bottom-Left:  Network topology with activation heat-map
      Bottom-Right: Training dashboard + keyboard controls
    =====================================================================  */

#include <windows.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "nn.h"

/* ── configuration ─────────────────────────────────────────────────── */
#define VIS_W           1280
#define VIS_H           720
#define COST_MAX        4000
#define PREVIEW_RES     64
#define EPOCHS_PER_FRAME 100

/* ── colour helpers (0x00BBGGRR for Win32) ─────────────────────────── */
#define RGB_BG          RGB(18,18,36)
#define RGB_PANEL       RGB(28,28,52)
#define RGB_BORDER      RGB(60,60,120)
#define RGB_GRID        RGB(40,40,70)
#define RGB_TEXT        RGB(200,210,230)
#define RGB_DIM         RGB(120,130,160)
#define RGB_ACCENT      RGB(0,255,180)
#define RGB_ACCENT2     RGB(130,80,255)
#define RGB_TITLE       RGB(255,255,255)
#define RGB_GREEN       RGB(0,255,180)
#define RGB_RED         RGB(255,60,80)

/* ── state ─────────────────────────────────────────────────────────── */
typedef struct {
    float   costs[COST_MAX];
    int     cost_n;
    size_t  epoch;
    float   cost;
    float   rate;
    int     paused;
    int     done;
    int     epf;            /* epochs per frame */
    int     needs_reset;

    /* preview DIB */
    uint32_t pix[PREVIEW_RES * PREVIEW_RES];

    /* Win32 */
    HWND    hwnd;
    HDC     memdc;
    HBITMAP membmp;
    HBITMAP oldbmp;
    int     running;

    /* smooth cost for display */
    float   disp_cost;
} Vis;

static Vis g_vis;

/* ── forward declarations ──────────────────────────────────────────── */
static LRESULT CALLBACK vis_wndproc(HWND, UINT, WPARAM, LPARAM);
static void vis_paint(HDC hdc, NN nn);

/* ── colour math ───────────────────────────────────────────────────── */
static COLORREF vis_lerp_rgb(COLORREF a, COLORREF b, float t) {
    if (t < 0) t = 0; if (t > 1) t = 1;
    int r = GetRValue(a) + (int)((GetRValue(b) - GetRValue(a)) * t);
    int g = GetGValue(a) + (int)((GetGValue(b) - GetGValue(a)) * t);
    int bl= GetBValue(a) + (int)((GetBValue(b) - GetBValue(a)) * t);
    return RGB(r, g, bl);
}
static float vis_clamp(float v, float lo, float hi) {
    return v < lo ? lo : v > hi ? hi : v;
}

/* ── API ───────────────────────────────────────────────────────────── */

static void vis_init(float rate) {
    memset(&g_vis, 0, sizeof(g_vis));
    g_vis.rate    = rate;
    g_vis.epf     = EPOCHS_PER_FRAME;
    g_vis.running = 1;

    WNDCLASSA wc  = {0};
    wc.lpfnWndProc   = vis_wndproc;
    wc.hInstance      = GetModuleHandle(NULL);
    wc.lpszClassName  = "NNVis";
    wc.hCursor        = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground  = (HBRUSH)GetStockObject(BLACK_BRUSH);
    RegisterClassA(&wc);

    /* adjust for client area */
    RECT rc = {0, 0, VIS_W, VIS_H};
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);

    g_vis.hwnd = CreateWindowA("NNVis",
        "Neural Network Training Visualizer",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT,
        rc.right - rc.left, rc.bottom - rc.top,
        NULL, NULL, wc.hInstance, NULL);

    /* create double-buffer */
    HDC hdc = GetDC(g_vis.hwnd);
    g_vis.memdc  = CreateCompatibleDC(hdc);
    g_vis.membmp = CreateCompatibleBitmap(hdc, VIS_W, VIS_H);
    g_vis.oldbmp = (HBITMAP)SelectObject(g_vis.memdc, g_vis.membmp);
    ReleaseDC(g_vis.hwnd, hdc);
}

static int vis_should_close(void) { return !g_vis.running; }

static int vis_process_events(void) {
    MSG msg;
    while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) { g_vis.running = 0; return 0; }
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return g_vis.running;
}

static void vis_record_cost(float c) {
    if (g_vis.cost_n < COST_MAX) {
        g_vis.costs[g_vis.cost_n++] = c;
    } else {
        /* decimate: keep every 2nd point */
        int half = COST_MAX / 2;
        for (int i = 0; i < half; i++) g_vis.costs[i] = g_vis.costs[i*2];
        g_vis.cost_n = half;
        g_vis.costs[g_vis.cost_n++] = c;
    }
    g_vis.cost = c;
    g_vis.disp_cost = g_vis.disp_cost * 0.85f + c * 0.15f;
}

/* rebuild the preview pixel buffer */
static void vis_update_preview(NN nn) {
    for (int y = 0; y < PREVIEW_RES; y++) {
        for (int x = 0; x < PREVIEW_RES; x++) {
            MAT_AT(nn.as[0], 0, 0) = (float)x / (PREVIEW_RES - 1);
            MAT_AT(nn.as[0], 0, 1) = (float)y / (PREVIEW_RES - 1);
            nn_forward(nn);
            float v = vis_clamp(MAT_AT(nn.as[nn.count], 0, 0), 0, 1);
            uint8_t c = (uint8_t)(v * 255);
            /* BGRA for Win32 DIB */
            g_vis.pix[y * PREVIEW_RES + x] = (uint32_t)c | ((uint32_t)c << 8) | ((uint32_t)c << 16) | 0xFF000000u;
        }
    }
}

static void vis_render(NN nn) {
    vis_update_preview(nn);
    vis_paint(g_vis.memdc, nn);

    /* blit to screen */
    HDC hdc = GetDC(g_vis.hwnd);
    BitBlt(hdc, 0, 0, VIS_W, VIS_H, g_vis.memdc, 0, 0, SRCCOPY);
    ReleaseDC(g_vis.hwnd, hdc);
}

static void vis_close(void) {
    SelectObject(g_vis.memdc, g_vis.oldbmp);
    DeleteObject(g_vis.membmp);
    DeleteDC(g_vis.memdc);
    DestroyWindow(g_vis.hwnd);
}

/* ── window proc ───────────────────────────────────────────────────── */
static LRESULT CALLBACK vis_wndproc(HWND hw, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_DESTROY: PostQuitMessage(0); g_vis.running = 0; return 0;
    case WM_KEYDOWN:
        switch (wp) {
        case VK_SPACE:  g_vis.paused = !g_vis.paused; break;
        case 'R':       g_vis.needs_reset = 1; g_vis.cost_n = 0;
                        g_vis.epoch = 0; g_vis.done = 0; break;
        case VK_UP:     g_vis.epf *= 2; if (g_vis.epf > 10000) g_vis.epf = 10000; break;
        case VK_DOWN:   g_vis.epf /= 2; if (g_vis.epf < 1) g_vis.epf = 1; break;
        case VK_OEM_PLUS: case VK_ADD:
            g_vis.rate *= 1.5f; if (g_vis.rate > 10) g_vis.rate = 10; break;
        case VK_OEM_MINUS: case VK_SUBTRACT:
            g_vis.rate /= 1.5f; if (g_vis.rate < 0.0001f) g_vis.rate = 0.0001f; break;
        case VK_ESCAPE: PostQuitMessage(0); break;
        }
        return 0;
    case WM_ERASEBKGND: return 1;   /* prevent flicker */
    }
    return DefWindowProc(hw, msg, wp, lp);
}

/* ── GDI drawing helpers ───────────────────────────────────────────── */

static void gdi_fill_rect(HDC hdc, int x, int y, int w, int h, COLORREF col) {
    HBRUSH br = CreateSolidBrush(col);
    RECT rc = {x, y, x+w, y+h};
    FillRect(hdc, &rc, br);
    DeleteObject(br);
}

static void gdi_rect_border(HDC hdc, int x, int y, int w, int h, COLORREF col) {
    HPEN pen = CreatePen(PS_SOLID, 1, col);
    HPEN old = (HPEN)SelectObject(hdc, pen);
    HBRUSH ob = (HBRUSH)SelectObject(hdc, GetStockObject(NULL_BRUSH));
    Rectangle(hdc, x, y, x+w, y+h);
    SelectObject(hdc, ob);
    SelectObject(hdc, old);
    DeleteObject(pen);
}

static void gdi_line(HDC hdc, int x1, int y1, int x2, int y2, COLORREF col, int thick) {
    HPEN pen = CreatePen(PS_SOLID, thick, col);
    HPEN old = (HPEN)SelectObject(hdc, pen);
    MoveToEx(hdc, x1, y1, NULL);
    LineTo(hdc, x2, y2);
    SelectObject(hdc, old);
    DeleteObject(pen);
}

static void gdi_circle(HDC hdc, int cx, int cy, int r, COLORREF fill) {
    HBRUSH br = CreateSolidBrush(fill);
    HBRUSH ob = (HBRUSH)SelectObject(hdc, br);
    HPEN pen = CreatePen(PS_SOLID, 1, fill);
    HPEN op  = (HPEN)SelectObject(hdc, pen);
    Ellipse(hdc, cx-r, cy-r, cx+r, cy+r);
    SelectObject(hdc, op); DeleteObject(pen);
    SelectObject(hdc, ob); DeleteObject(br);
}

static void gdi_text(HDC hdc, int x, int y, const char *s, COLORREF col, int size) {
    HFONT font = CreateFontA(size, 0, 0, 0, FW_NORMAL, 0,0,0,
        DEFAULT_CHARSET, 0,0, ANTIALIASED_QUALITY, FF_MODERN, "Consolas");
    HFONT of = (HFONT)SelectObject(hdc, font);
    SetTextColor(hdc, col);
    SetBkMode(hdc, TRANSPARENT);
    TextOutA(hdc, x, y, s, (int)strlen(s));
    SelectObject(hdc, of);
    DeleteObject(font);
}

static void gdi_text_bold(HDC hdc, int x, int y, const char *s, COLORREF col, int size) {
    HFONT font = CreateFontA(size, 0, 0, 0, FW_BOLD, 0,0,0,
        DEFAULT_CHARSET, 0,0, ANTIALIASED_QUALITY, FF_MODERN, "Consolas");
    HFONT of = (HFONT)SelectObject(hdc, font);
    SetTextColor(hdc, col);
    SetBkMode(hdc, TRANSPARENT);
    TextOutA(hdc, x, y, s, (int)strlen(s));
    SelectObject(hdc, of);
    DeleteObject(font);
}

/* ── panel frame ───────────────────────────────────────────────────── */
static void gdi_panel(HDC hdc, int x, int y, int w, int h, const char *title) {
    gdi_fill_rect(hdc, x, y, w, h, RGB_PANEL);
    gdi_rect_border(hdc, x, y, w, h, RGB_BORDER);
    if (title) {
        gdi_fill_rect(hdc, x+1, y+1, w-2, 26, RGB(40,40,80));
        gdi_line(hdc, x, y+27, x+w, y+27, RGB_BORDER, 1);
        gdi_text_bold(hdc, x+10, y+5, title, RGB_ACCENT, 15);
    }
}

/* ── cost graph ────────────────────────────────────────────────────── */
static void draw_cost_graph(HDC hdc, int px, int py, int pw, int ph) {
    gdi_panel(hdc, px, py, pw, ph, "LOSS CURVE");

    int gx = px + 55, gy = py + 38;
    int gw = pw - 70, gh = ph - 58;

    if (g_vis.cost_n < 2) {
        gdi_text(hdc, gx + gw/2 - 60, gy + gh/2 - 7, "Waiting for data...", RGB_DIM, 14);
        return;
    }

    /* auto-scale */
    float mn = g_vis.costs[0], mx = g_vis.costs[0];
    for (int i = 1; i < g_vis.cost_n; i++) {
        if (g_vis.costs[i] < mn) mn = g_vis.costs[i];
        if (g_vis.costs[i] > mx) mx = g_vis.costs[i];
    }
    if (mx - mn < 1e-4f) mx = mn + 0.001f;

    /* grid lines */
    for (int i = 0; i <= 4; i++) {
        int ly = gy + gh - (int)((float)i / 4.0f * gh);
        gdi_line(hdc, gx, ly, gx+gw, ly, RGB_GRID, 1);
        char buf[32]; snprintf(buf, 32, "%.4f", mn + (mx-mn)*i/4.0f);
        gdi_text(hdc, px+4, ly-6, buf, RGB_DIM, 11);
    }

    /* axis labels */
    { char b[32]; snprintf(b,32,"0"); gdi_text(hdc,gx,gy+gh+3,b,RGB_DIM,11); }
    { char b[32]; snprintf(b,32,"%zu",g_vis.epoch); gdi_text(hdc,gx+gw-40,gy+gh+3,b,RGB_DIM,11); }

    /* draw curve */
    int n = g_vis.cost_n;
    int step = n > gw ? n / gw : 1;

    POINT *pts = (POINT*)malloc(sizeof(POINT) * (n/step + 2));
    int np = 0;

    for (int i = 0; i < n; i += step) {
        float t = (float)i / (n - 1);
        int sx = gx + (int)(t * gw);
        float val = (g_vis.costs[i] - mn) / (mx - mn);
        int sy = gy + gh - (int)(val * gh);
        pts[np].x = sx;
        pts[np].y = sy;
        np++;
    }

    /* filled area (draw vertical lines to create fill effect) */
    for (int i = 0; i < np; i++) {
        COLORREF fc = RGB(0, 60 + (int)(40.0f * (float)i / np), 50);
        gdi_line(hdc, pts[i].x, pts[i].y, pts[i].x, gy+gh, fc, 1);
    }

    /* line on top */
    HPEN pen = CreatePen(PS_SOLID, 2, RGB_GREEN);
    HPEN old = (HPEN)SelectObject(hdc, pen);
    for (int i = 1; i < np; i++) {
        MoveToEx(hdc, pts[i-1].x, pts[i-1].y, NULL);
        LineTo(hdc, pts[i].x, pts[i].y);
    }
    SelectObject(hdc, old); DeleteObject(pen);

    /* end dot */
    if (np > 0) {
        gdi_circle(hdc, pts[np-1].x, pts[np-1].y, 5, RGB_ACCENT);
        gdi_circle(hdc, pts[np-1].x, pts[np-1].y, 2, RGB_TITLE);
    }

    free(pts);
}

/* ── image preview ─────────────────────────────────────────────────── */
static void draw_preview(HDC hdc, int px, int py, int pw, int ph) {
    gdi_panel(hdc, px, py, pw, ph, "NN OUTPUT (Reconstruction)");

    int gy = py + 34;
    int avail = ph - 44;
    int aw = pw - 16;
    int sz = avail < aw ? avail : aw;
    int ix = px + (pw - sz)/2;
    int iy = gy + (avail - sz)/2;

    /* checkerboard bg */
    for (int cy = 0; cy < sz; cy += 8)
        for (int cx = 0; cx < sz; cx += 8) {
            int dark = ((cx/8 + cy/8) & 1) == 0;
            COLORREF cc = dark ? RGB(20,20,40) : RGB(30,30,50);
            int cw = cx+8 > sz ? sz-cx : 8;
            int ch = cy+8 > sz ? sz-cy : 8;
            gdi_fill_rect(hdc, ix+cx, iy+cy, cw, ch, cc);
        }

    /* stretch-blit the preview DIB */
    BITMAPINFO bmi;
    memset(&bmi, 0, sizeof(bmi));
    bmi.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth       = PREVIEW_RES;
    bmi.bmiHeader.biHeight      = -PREVIEW_RES;   /* top-down */
    bmi.bmiHeader.biPlanes      = 1;
    bmi.bmiHeader.biBitCount    = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    SetStretchBltMode(hdc, COLORONCOLOR);
    StretchDIBits(hdc, ix, iy, sz, sz,
                  0, 0, PREVIEW_RES, PREVIEW_RES,
                  g_vis.pix, &bmi, DIB_RGB_COLORS, SRCCOPY);

    /* border */
    gdi_rect_border(hdc, ix, iy, sz, sz, RGB_ACCENT2);

    /* resolution label */
    char buf[32]; snprintf(buf,32,"%dx%d", PREVIEW_RES, PREVIEW_RES);
    gdi_text(hdc, ix + sz - 40, iy + sz + 4, buf, RGB_DIM, 11);
}

/* ── network topology ──────────────────────────────────────────────── */
static void draw_network(HDC hdc, NN nn, int px, int py, int pw, int ph) {
    gdi_panel(hdc, px, py, pw, ph, "NETWORK TOPOLOGY");

    int gx = px + 40;
    int gy = py + 44;
    int gw = pw - 80;
    int gh = ph - 60;

    size_t nL = nn.count + 1;

    float lsp = (float)gw / (float)(nL - 1);

    /* draw connections */
    for (size_t l = 0; l < nn.count; l++) {
        size_t nf = nn.as[l].cols;
        size_t nt = nn.as[l+1].cols;
        float fs = (float)gh / (float)(nf+1);
        float ts = (float)gh / (float)(nt+1);

        for (size_t i = 0; i < nf; i++) {
            int fx = gx + (int)(l * lsp);
            int fy = gy + (int)((i+1) * fs);
            for (size_t j = 0; j < nt; j++) {
                int tx = gx + (int)((l+1) * lsp);
                int ty = gy + (int)((j+1) * ts);
                float w = MAT_AT(nn.ws[l], i, j);
                float aw = fabsf(w); if (aw > 2) aw = 2;
                int thick = 1 + (int)(aw * 1.2f);
                COLORREF wc = w >= 0 ? RGB(0, (int)(100+aw*60), (int)(80+aw*40))
                                     : RGB((int)(100+aw*60), (int)(30+aw*15), (int)(40+aw*20));
                gdi_line(hdc, fx, fy, tx, ty, wc, thick);
            }
        }
    }

    /* draw neurons */
    for (size_t l = 0; l < nL; l++) {
        size_t n = nn.as[l].cols;
        float sp = (float)gh / (float)(n+1);
        int lx = gx + (int)(l * lsp);

        for (size_t i = 0; i < n; i++) {
            int ly = gy + (int)((i+1) * sp);
            float act = vis_clamp(MAT_AT(nn.as[l], 0, i), 0, 1);

            /* outer glow */
            COLORREF glow = vis_lerp_rgb(RGB(20,40,120), RGB(255,120,60), act);
            gdi_circle(hdc, lx, ly, 13 + (int)(act*4), glow);

            /* body */
            COLORREF nc = vis_lerp_rgb(RGB(30,60,160), RGB(255,100,60), act);
            gdi_circle(hdc, lx, ly, 10, nc);

            /* highlight */
            COLORREF hi = vis_lerp_rgb(nc, RGB_TITLE, 0.3f);
            gdi_circle(hdc, lx, ly, 5, hi);

            /* value */
            char buf[16]; snprintf(buf,16,"%.2f", act);
            gdi_text(hdc, lx-14, ly+15, buf, RGB_DIM, 10);
        }

        /* layer label */
        const char *label; char lb[16];
        if (l == 0) label = "INPUT";
        else if (l == nL-1) label = "OUTPUT";
        else { snprintf(lb,16,"H%zu",l); label = lb; }
        gdi_text_bold(hdc, lx - 14, gy - 18, label, RGB_ACCENT2, 12);
    }
}

/* ── stats / controls panel ────────────────────────────────────────── */
static void draw_stats(HDC hdc, int px, int py, int pw, int ph) {
    gdi_panel(hdc, px, py, pw, ph, "TRAINING DASHBOARD");

    int y = py + 40, x = px + 16, vx = px + 148;
    char buf[128];

    /* Epoch */
    gdi_text(hdc, x, y, "EPOCH", RGB_DIM, 14);
    snprintf(buf, 128, "%zu", g_vis.epoch);
    gdi_text_bold(hdc, vx, y, buf, RGB_TITLE, 14);
    y += 26;

    /* Cost */
    gdi_text(hdc, x, y, "COST", RGB_DIM, 14);
    snprintf(buf, 128, "%.6f", g_vis.disp_cost);
    float ct = vis_clamp(g_vis.disp_cost * 4.0f, 0, 1);
    COLORREF cc = vis_lerp_rgb(RGB_GREEN, RGB_RED, ct);
    gdi_text_bold(hdc, vx, y, buf, cc, 14);
    y += 26;

    /* Learning rate */
    gdi_text(hdc, x, y, "LEARN RATE", RGB_DIM, 14);
    snprintf(buf, 128, "%.4f", g_vis.rate);
    gdi_text_bold(hdc, vx, y, buf, RGB_TITLE, 14);
    y += 26;

    /* Speed */
    gdi_text(hdc, x, y, "SPEED", RGB_DIM, 14);
    snprintf(buf, 128, "%d ep/frame", g_vis.epf);
    gdi_text_bold(hdc, vx, y, buf, RGB_TITLE, 14);
    y += 26;

    /* Status */
    gdi_text(hdc, x, y, "STATUS", RGB_DIM, 14);
    if (g_vis.done)       gdi_text_bold(hdc, vx, y, "COMPLETE", RGB_ACCENT, 14);
    else if (g_vis.paused) gdi_text_bold(hdc, vx, y, "PAUSED", RGB(255,200,0), 14);
    else                   gdi_text_bold(hdc, vx, y, "TRAINING...", RGB_ACCENT, 14);
    y += 36;

    /* divider */
    gdi_line(hdc, x, y, px+pw-16, y, RGB_BORDER, 1);
    y += 12;

    /* controls header */
    gdi_text_bold(hdc, x, y, "CONTROLS", RGB_ACCENT2, 14);
    y += 24;

    struct { const char *key; const char *desc; } ctrls[] = {
        {"[SPACE]",  "Pause / Resume"},
        {"[R]",      "Reset weights"},
        {"[UP/DN]",  "Speed +/-"},
        {"[+/-]",    "Learn rate +/-"},
        {"[ESC]",    "Quit"},
    };
    for (int i = 0; i < 5; i++) {
        gdi_text_bold(hdc, x, y, ctrls[i].key, RGB_ACCENT, 12);
        gdi_text(hdc, x+75, y, ctrls[i].desc, RGB_DIM, 12);
        y += 20;
    }

    /* progress bar */
    y += 10;
    gdi_text(hdc, x, y, "PROGRESS", RGB_DIM, 12);
    y += 18;
    int barw = pw - 40;
    float pct = vis_clamp((float)g_vis.epoch / 100000.0f, 0, 1);
    gdi_fill_rect(hdc, x, y, barw, 14, RGB(20,20,45));
    gdi_fill_rect(hdc, x, y, (int)(barw * pct), 14,
                  vis_lerp_rgb(RGB_ACCENT2, RGB_ACCENT, pct));
    gdi_rect_border(hdc, x, y, barw, 14, RGB_BORDER);
    snprintf(buf, 128, "%.1f%%", pct*100);
    gdi_text(hdc, x + barw/2 - 15, y, buf, RGB_TITLE, 12);
}

/* ── main paint ────────────────────────────────────────────────────── */
static void vis_paint(HDC hdc, NN nn) {
    /* clear */
    gdi_fill_rect(hdc, 0, 0, VIS_W, VIS_H, RGB_BG);

    /* accent top bar */
    gdi_fill_rect(hdc, 0, 0, VIS_W, 3, RGB_ACCENT);

    int m  = 8;
    int hw = VIS_W / 2;
    int hh = VIS_H / 2;

    /* top-left:  cost graph */
    draw_cost_graph(hdc, m, m, hw - m*2, hh - m);

    /* top-right: preview */
    draw_preview(hdc, hw + m, m, hw - m*2, hh - m);

    /* bottom-left: network */
    draw_network(hdc, nn, m, hh + m, hw - m*2, hh - m*2);

    /* bottom-right: stats */
    draw_stats(hdc, hw + m, hh + m, hw - m*2, hh - m*2);
}

#endif /* VISUALIZER_H_ */
