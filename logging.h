#pragma once

/**
 * log an error message to STDERR
 */
#define LOG_ERROR(msg, ...) do { \
    printf("ERROR: " msg "\n", ##__VA_ARGS__); \
} while (0)

/**
 * log a warning to stderr, unless --quiet was enabled
 */
#define LOG_WARN(params, msg, ...) do { \
    if (!(params)->quiet) printf("WARNING: " msg "\n", ##__VA_ARGS__); \
} while (0)

/**
 * log extra information if --verbose was enabled
 */
#define LOG_VERBOSE(verbose, msg, ...) do { \
    if (verbose) printf(msg "\n", ##__VA_ARGS__); \
} while (0)
