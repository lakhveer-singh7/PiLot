#ifndef LOGGING_H
#define LOGGING_H

#include <stdio.h>
#include <stdarg.h>
#include <time.h>

void log_init(const char* filename);
void log_cleanup(void);
void log_info(const char* format, ...);
void log_error(const char* format, ...);
void log_debug(const char* format, ...);
void set_log_level_debug(void);

#endif // LOGGING_H
