#include "logging.h"
#include <stdarg.h>
#include <time.h>

static FILE* g_log_file = NULL;
static int   g_log_level = 0;   /* 0 = INFO, 1 = DEBUG */

void log_init(const char* filename) {
    if (filename) {
        g_log_file = fopen(filename, "w");
        if (!g_log_file) {
            fprintf(stderr, "Cannot open log file: %s\n", filename);
            g_log_file = stderr;
        }
    } else {
        g_log_file = stderr;
    }
}

void log_cleanup(void) {
    if (g_log_file && g_log_file != stderr && g_log_file != stdout)
        fclose(g_log_file);
    g_log_file = NULL;
}

static void log_msg(const char* lvl, const char* fmt, va_list ap) {
    if (!g_log_file) g_log_file = stderr;
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    fprintf(g_log_file, "[%ld.%03ld] %s: ",
            ts.tv_sec % 1000, ts.tv_nsec / 1000000, lvl);
    vfprintf(g_log_file, fmt, ap);
    fprintf(g_log_file, "\n");
    fflush(g_log_file);
}

void log_info(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); log_msg("INFO", fmt, ap); va_end(ap);
}
void log_error(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); log_msg("ERROR", fmt, ap); va_end(ap);
}
void log_debug(const char* fmt, ...) {
    if (g_log_level < 1) return;
    va_list ap; va_start(ap, fmt); log_msg("DEBUG", fmt, ap); va_end(ap);
}
void set_log_level_debug(void) { g_log_level = 1; }
