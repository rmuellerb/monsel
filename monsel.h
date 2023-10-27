#pragma once

// enable all the POSIX, GNU, and Linux extensions
// -- must be defined before including any standard library header
#define _GNU_SOURCE

#include <stdint.h>

struct ea_parameters {
    // program
    char* inputfname;
    char* savefname;
    char* genfname;
    uint8_t random;
    uint8_t extended_write;
    uint8_t verbose;
    uint8_t quiet;
    uint8_t irace;
    long tuning;
    long seed;
    long seed_ea;
    float changelevel;
    // ea
    uint8_t dont_reevaluate;
    unsigned long popsize;
    unsigned long tournsize;
    double mutpb;
    unsigned long max_evals;
    unsigned long modelcount;
    // ls
    uint8_t do_localsearch;
    long ls_k;
    uint8_t ls_only;
    // pi
    uint8_t do_pi;
    long pi_width;
    double pi_threshold;
    long pi_size;
};

typedef uint8_t Gene;
typedef long Fitness;

Fitness run_default(struct ea_parameters* params);
int run_random(struct ea_parameters* params);
int run_meta_ea_tuning(struct ea_parameters* params);

/**
 * Print the configuration of the experiment
 */
void print_config(struct ea_parameters *params);
