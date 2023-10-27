#include "monsel.h"
#include "utils.h"
#include "logging.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

typedef struct Tuning_Individual {
    Fitness utility;
    long popsize;
    long tournsize;
    double mutpb;
    long ls_k;
    long pi_width;
    double pi_threshold;
    long pi_size;
} Tuning_Individual;

/*
 * Prints the current Tuning_Individual on one line
 */
void print_tuning_short(Tuning_Individual* ind)
{
    printf("<psize: %ld, tsize: %ld, mutpb: %2.2f, ls_k: %ld, "
            "pi_wi: %ld, pi_thresh: %2.2f, pi_size: %ld>\n",
            ind->popsize, ind->tournsize, ind->mutpb, ind->ls_k,
            ind->pi_width, ind->pi_threshold, ind->pi_size);
}

/*
 * Creates and returns a random Tuning_Individual instance.
 * Change boundaries for parameters here if necessary.
 */
Tuning_Individual create_tuning_ind(struct ea_parameters* params)
{
    Tuning_Individual ind;
    ind.utility = 0;
    ind.popsize = rand() % params->max_evals;
    if(ind.popsize == 0)
        ind.popsize = 1;
    ind.tournsize = rand() % ind.popsize;
    ind.mutpb = frand(1.0);
    ind.ls_k = rand() % ind.popsize;
    ind.pi_width = (rand() % (int)(params->max_evals / params->popsize)) + 1;
    ind.pi_threshold = frand(1.0);
    ind.pi_size = rand() % (int)(params->max_evals / 2);
    return ind;
}

struct ea_parameters* prepare_ea_instance(struct ea_parameters* params, Tuning_Individual* ind)
{
    params->popsize = ind->popsize;
    params->tournsize = ind->tournsize;
    params->mutpb = ind->mutpb;
    params->ls_k = ind->ls_k;
    params->pi_width = ind->pi_width;
    params->pi_threshold = ind->pi_threshold;
    params->pi_size = ind->pi_size;
    return params;
}

static long mutate_i(long value, long delta, long min)
{
    assert(delta > 0);
    assert(min <= value);

    long offset = value - delta;
    if (offset < min)
        offset = min;

    long span = value + delta - offset;

    long result = (rand() % span) + offset;
    assert(min <= result);
    assert(value - delta <= result && result <= value + delta);

    return result;
}

static double mutate_d(double value, double delta, double min, double max)
{
    assert(delta > 0);
    assert(min <= value && value <= max);

    double lo = value - delta;
    if (lo < min)
        lo = min;

    double hi = value + delta;
    if (hi > max)
        hi = max;

    double result = frand(hi - lo) + lo;
    assert(min <= result && result <= max);
    assert(value - delta <= result && result <= value + delta);

    return result;
}

/*
 * Mutates the given tuning individual
 */
void tuning_mutation(Tuning_Individual* ind)
{
    // XXX: Adapt here if more tuning parameters are used!
    // Mutation is done by changing the values by [-50, 50] for integer
    // and [-0.05,0.05] for float values

    assert(ind);
    ind->popsize      = mutate_i(ind->popsize, 50, 1);
    ind->tournsize    = mutate_i(ind->tournsize, 50, 1);
    ind->mutpb        = mutate_d(ind->mutpb, 0.05, 0.0, 1.0);
    ind->ls_k         = mutate_i(ind->ls_k, 50, 1);
    ind->pi_width     = mutate_i(ind->pi_width, 50, 1);
    ind->pi_threshold = mutate_d(ind->pi_threshold, 0.05, 0.0, INFINITY);
    ind->pi_size      = mutate_i(ind->pi_size, 50, 1);
}

#define CROSSOVER(toggle, p1, p2, c1, c2, fieldname) do {                   \
    int _crossover_toggle = (toggle);                                       \
    (c1)->fieldname = _crossover_toggle ? (p1)->fieldname : (p2)->fieldname;\
    (c2)->fieldname = _crossover_toggle ? (p2)->fieldname : (p1)->fieldname;\
} while (0)

/*
 * Creates a uniform crossover between tuning parameters.
 * Works inline on the given individuals
 */
void tuning_crossover(Tuning_Individual* p1, Tuning_Individual* p2, Tuning_Individual* c1, Tuning_Individual* c2)
{
    // XXX adapt here if more tuning parameters are used!
    CROSSOVER(rand() % 2, p1, p2, c1, c2, popsize);
    CROSSOVER(rand() % 2, p1, p2, c1, c2, tournsize);
    CROSSOVER(rand() % 2, p1, p2, c1, c2, mutpb);
    CROSSOVER(rand() % 2, p1, p2, c1, c2, ls_k);
    CROSSOVER(rand() % 2, p1, p2, c1, c2, pi_width);
    CROSSOVER(rand() % 2, p1, p2, c1, c2, pi_threshold);
    CROSSOVER(rand() % 2, p1, p2, c1, c2, pi_size);
}

/*
 * select individuals from the tuning population
 */
int tuning_selection(Tuning_Individual* pop, int size)
{
    const int TUNING_TOURNSIZE = 5;
    int best = rand() % size;
    for(int i=1; i<TUNING_TOURNSIZE; i++)
    {
        int choice = rand() % size;
        if(pop[choice].utility < pop[best].utility)
            best = choice;
    }
    return best;
}


/*
 * Prints the statistics of the given population 'pop' of size 'size' for generation 'gen'
 * for the tuning meta-EA
 */
void print_tuning_stats(Tuning_Individual *pop, int size, int gen)
{
    double values[size];
    for(int i=0; i<size; i++)
        values[i] = (double)pop[i].utility;
    printf("gen %4d", gen);
    printf(" - mean %4.2f", fitness_mean(values, size));
    printf(" - stdev %4.2f", fitness_stdev(values, size));
    printf(" - min %4.2f", fitness_min(values, size));
    printf(" - max %4.2f", fitness_max(values, size));
    printf("\n");
}

/*
 * Runs the parameter tuner
 */
int run_meta_ea_tuning(struct ea_parameters* params)
{
    const int TUNING_POPSIZE = 100;
    int verbose_tuning = params->verbose;
    params->verbose = 0;
    LOG_VERBOSE(verbose_tuning,
            "Running %ld iterations of EA parameter tuning "
            "using %ld evaluations per fitness call",
            params->tuning, params->max_evals);

    Tuning_Individual pop[TUNING_POPSIZE*2];
    Tuning_Individual best;
    LOG_VERBOSE(verbose_tuning, "Initializing population");
    for(int i = 0; i < TUNING_POPSIZE; i++)
    {
        pop[i] = create_tuning_ind(params);
        params = prepare_ea_instance(params, &pop[i]);
        pop[i].utility = run_default(params);
        //printf("Individual <%3d>: %ld\n", (i+1), pop[i].utility);
        if(i)
        {
            if(pop[i].utility < best.utility)
                best = pop[i];
        }
        else
            best = pop[i];
    }
    LOG_VERBOSE(verbose_tuning, "Running evolutionary process");
    for(int i=0; i<params->tuning; i++)
    {
        for(int j = TUNING_POPSIZE; j < TUNING_POPSIZE*2; j = j+2)
        {
            int p1 = tuning_selection(pop, TUNING_POPSIZE);
            int p2 = tuning_selection(pop, TUNING_POPSIZE);
            Tuning_Individual c1 = create_tuning_ind(params);
            Tuning_Individual c2 = create_tuning_ind(params);
            tuning_crossover(&pop[p1], &pop[p2], &c1, &c2);
            tuning_mutation(&c1);
            tuning_mutation(&c2);
            params = prepare_ea_instance(params, &c1);
            params = prepare_ea_instance(params, &c2);
            c1.utility = run_default(params);
            c2.utility = run_default(params);

            pop[j  ] = c1;
            pop[j+1] = c2;
            if(c1.utility < best.utility)
                best = c1;
            if(c2.utility < best.utility)
                best = c2;
        }

        // move the surviving individuals to the front of the population,
        // which simplifies selection.
        for(int j = 0; j < TUNING_POPSIZE; j++)
        {
            int s = j + tuning_selection(pop + j, TUNING_POPSIZE*2 - j);
            SWAP_LVALUE(Tuning_Individual, temp, pop[j], pop[s]);
        }
        if(verbose_tuning)
            print_tuning_stats(pop, TUNING_POPSIZE, i);
    }
    printf("<Tuning Result>\n");
    printf("Best utility: %ld\n", best.utility);
    print_config(prepare_ea_instance(params, &best));
    return 0;
}

