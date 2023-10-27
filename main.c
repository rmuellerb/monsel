#include "monsel.h"
#include "logging.h"

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <stdbool.h>
#include <argp.h>
#include <time.h>

const char *argp_program_version = "0.9.2";
const char *argp_program_bug_address = "<mueller-bady@linux.com>";
static char doc[] =
        "Program to run (LS + PI) EA experiments "
        "and writing fitness value to given outputfile.\n"
        "The inputfiles are given as base file "
        "(format: [E,src,dst,w] and [V,id]) "
        "and parameter '--models/-z' to state the number of models created. "
        "The models are created on-the-fly according to a given change level.\n"
        "If only one file is given and parameter '--models / -z' is set to one, "
        "only the base file is used. "
        "This program can also be used for parameter tuning "
        "using the '--tuning / -u' switch.";
static char args_doc[] = "[network.csv]";

#define DECLARE_OPT(name, key, arg, doc) {name, key, arg, 0, doc, 0}
#define DECLARE_OPT_HEADER(name) {0, 0, 0, 0, name, 0}

static struct argp_option options[] =
{
    DECLARE_OPT_HEADER("EA parameters:"),
    DECLARE_OPT("popsize", 'p', "POPSIZE",
            "Size of the population (default: 100)"),
    DECLARE_OPT("tournsize", 't', "TOURNSIZE",
            "Size of the tournament (default: 5)"),
    DECLARE_OPT("mutpb", 'm', "MUTPB",
            "Mutation probability per gene (default: 0.05)"),
    DECLARE_OPT("nevals", 'n', "MAX_NEVALS",
            "Maximum number of evaluations (default: 100,000)"),
    DECLARE_OPT("models", 'z', "AMOUNT",
            "Amount of models during runtime (default: 1). "
            "Requires a given base model and subsequent numbered models "
            "in the same directory, "
            "e.g., NREN_base.csv, NREN_1.csv, NREN_2.csv, etc."),

    DECLARE_OPT_HEADER("Localsearch parameters:"),
    DECLARE_OPT("ls", 'l', NULL,
            "perform localsearch (default: no)"),
    DECLARE_OPT("ls_k", 'k', "K-VALUE",
            "localsearch k-parameter (default: 50)"),
    DECLARE_OPT("ls-only", 'b', NULL,
            "Skips all parts related to evolutionary optimization "
            "including PI and just performs local search (default: no)."),

    DECLARE_OPT_HEADER("Population injection parameters:"),
    DECLARE_OPT("pi", 'i', NULL,
            "perform population injection (default: no)"),
    DECLARE_OPT("pi-width", 'j', "WIDTH",
            "injection width (default: 3)"),
    DECLARE_OPT("pi-threshold", 'y', "THRESH",
            "injection threshold measured as stdev of population fitness "
            "(default: 0)"),
    DECLARE_OPT("pi-size", 'o', "SIZE",
            "injection size (default: 0)"),

    DECLARE_OPT_HEADER("Program parameters"),
    DECLARE_OPT("verbose", 'v', NULL,
            "Switch on verbose mode (default: no)"),
    DECLARE_OPT("quiet", 'a', NULL,
            "Switch on quiet mode (default: no)"),
    DECLARE_OPT("extended-write", 'e', NULL,
            "Switch on extended fitness mode (default: no). "
            "Ignored in case '--write'/'-w' is not given."),
    DECLARE_OPT("seed", 's', "SEED",
            "Seed of the pseudo random number generator for creating models. "
            "Important: does not affect RNG of the heuristic "
            "(default: current time in ns)"),
    DECLARE_OPT("seed-ea", 'f', "SEED",
            "Seed of the pseudo random number generator for the EA. "
            "Important: does not affect RNG of the instance generator "
            "for dynamic problems (default: current time in ns)"),
    DECLARE_OPT("changelevel", 'c', "LEVEL",
            "Sets the amount of change to apply "
            "for each consecutive network model. "
            "Ignored if '--models'/'-z' is set to 1 (default: 0.05)."),
    DECLARE_OPT("dont-reevaluate", 'd', NULL,
            "After switching a model, the population is not reevaluated. "
            "(default: no). Automatically set if popsize <= (nevals / models) "
            "to avoid using all nevals for reevaluation."),
    DECLARE_OPT("write", 'w', "FILE",
            "Fitness value output file (one value per line)"),
    DECLARE_OPT("gen-write", 'q', "FILE",
            "Extended fitness value output file for the whole generation "
            "(one generation per line using the given FILE)"),
    DECLARE_OPT("full-random", 'r', NULL,
            "Switch on full random mode, "
            "evaluating the defined amount of individuals without using LS/PI/EA. "
            "WARNING: Overwrites all other LS, EA, PI parameters! (default: no)"),

    DECLARE_OPT_HEADER("Tuning parameters:"),
    DECLARE_OPT("tuning", 'u', "AMOUNT",
            "Switch on tuning mode "
            "and use DEPTH evaluations for tuning the meta-EA. "
            "0 switches off tuning (default: 0)"),
    DECLARE_OPT("irace", 'g', NULL,
            "Switching on iRACE mode for external parameter tuning. "
            "Will print some required information to stdout "
            "as the very last line (default: no)."),

    {0}
};

static error_t parse_opt(int key, char *arg, struct argp_state *state)
{
    struct ea_parameters *params = state->input;
    switch(key)
    {
        case 'a': params->quiet = 1; break;
        case 'b': params->ls_only = 1; break;
        case 'c': params->changelevel = arg ? atof(arg) : 0.05; break;
        case 'd': params->dont_reevaluate = 1; break;
        case 'e': params->extended_write = 1; break;
        case 'f': params->seed_ea = arg ? atol(arg) : 42; break;
        case 'g': params->irace = 1; break;
        case 'i': params->do_pi = 1; break;
        case 'j': params->pi_width = arg ? atol(arg) : 0; break;
        case 'k': params->ls_k = arg ? atol(arg) : 0; break;
        case 'l': params->do_localsearch = 1; break;
        case 'm': params->mutpb = arg ? atof(arg) : 0.05; break;
        case 'n': params->max_evals = arg ? atol(arg) : 100000; break;
        case 'o': params->pi_size = arg ? atol(arg) : 0; break;
        case 'p': params->popsize = arg ? atol(arg) : 111; break;
        case 'q': params->genfname = arg; break;
        case 'r': params->random = 1; break;
        case 's': params->seed = arg ? atol(arg) : 42; break;
        case 't': params->tournsize = arg ? atol(arg) : 5; break;
        case 'u': params->tuning = arg ? atol(arg) : 0; break;
        case 'v': params->verbose = 1; break;
        case 'w': params->savefname = arg; break;
        case 'y': params->pi_threshold = arg ? atof(arg) : 0.0; break;
        case 'z': params->modelcount = arg ? atol(arg) : 0; break;
        case ARGP_KEY_END: if(state->arg_num < 1) argp_usage(state); break;
        case ARGP_KEY_ARG: if(state->arg_num > 1) argp_usage(state); else params->inputfname = arg; break;
        default: return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

static struct argp argp = {options, parse_opt, args_doc, doc, NULL, NULL, NULL};

static bool can_open_file(const char* fname, const char* mode) {
    FILE *fp = fopen(fname, mode);
    if (fp == NULL)
        return false;
    fclose(fp);
    return true;
}

/** validate that some property holds
 *
 *  result (bool): is assigned "false" upon failure
 *  expression (bool): the property being checked
 *  msg, ...: a printf format with optional args
 *
 *  expands to: a conditional that can take a block.
 *      The block is executed on success.
 *
 *  Example:
 *      VALIDATE(ok, 0 <= a, "a must be nonnegative");
 *
 *      VALIDATE(ok, a < b, "a must be smaller than b (a=%d, b=%d", a, b) {
 *          do_something(a, b);
 *      }
 */
#define VALIDATE(result, expression, msg, ...) \
    VALIDATE_FORBIDDEN((result), !(expression), msg, ##__VA_ARGS__)

/** validate that some property is FALSE.
 *
 *  Shortcut for VALIDATE(..., !(expression), ...)
 */
#define VALIDATE_FORBIDDEN(result, expression, msg, ...)    \
    if (expression)                                         \
    {                                                       \
        LOG_ERROR(msg, ##__VA_ARGS__);                 \
        (result) = false;                                   \
    } else

#define VALIDATE_RANGE_INC(result, value, lo, hi, msg, ...) \
    VALIDATE((result), lo <= (value) && (value) <= hi, msg, ##__VA_ARGS__)

bool check_parameters(struct ea_parameters* params)
{
    bool retval = true;

    VALIDATE(retval, can_open_file(params->inputfname, "r"),
            "Could not open file '%s'", params->inputfname);

    if(params->savefname)
    {
        VALIDATE(retval, can_open_file(params->savefname, "w"),
                "Could not open file '%s'", params->savefname)
        {
            remove(params->savefname);
        }
    }

    if(params->genfname)
    {
        VALIDATE(retval, can_open_file(params->genfname, "w"),
                "Could not open file '%s'", params->genfname)
        {
            remove(params->genfname);
        }
    }

    if (params->extended_write)
    {
        VALIDATE(retval, params->savefname,
                "Extended write requires a savefilename");
    }

    VALIDATE_FORBIDDEN(retval, params->quiet && params->verbose,
            "Modes '--quiet' and '--verbose' are mutually exclusive!");

    VALIDATE(retval, params->popsize <= params->max_evals,
            "Popsize must be smaller than the maximum number of evaluations");

    VALIDATE_RANGE_INC(retval, params->tournsize, 1, params->max_evals,
            "Tournsize must be between 1 and the maximum number of evaluations");

    VALIDATE_RANGE_INC(retval, params->mutpb, 0, 1,
            "mutpb must be between 0 and 1");

    VALIDATE_RANGE_INC(retval, params->modelcount, 1, params->max_evals,
            "modelcount must be between 1 and maximum number of  evaluations");

    if (params->do_pi)
    {
        VALIDATE(retval, params->pi_width >= 1,
                "PI width must be >= 1\n");

        VALIDATE_RANGE_INC(retval, params->pi_size, 1, (long) params->max_evals,
                "PI size mus be between 1 and the maximum number of evaluations");
    }

    VALIDATE(retval, params->pi_threshold >= 0,
            "PI threshold must be larger than 0\n");

    VALIDATE_RANGE_INC(retval, params->changelevel, 0, 1,
            "Change level must be between 0 and 1");

    if (params->random)
    {
        VALIDATE_FORBIDDEN(retval, params->genfname,
                "Random not compatible with parameter genfname");

        VALIDATE_FORBIDDEN(retval, params->do_pi,
                "Random not compatible with population injection");

        VALIDATE_FORBIDDEN(retval, params->do_localsearch || params->ls_only,
                "Random not compatible with local search");
    }

    if (params->ls_only)
    {
        VALIDATE_FORBIDDEN(retval, params->do_pi,
                "PI not compatible with option '--ls-only' / '-b'");
    }

    VALIDATE(retval, params->tuning >= 0,
            "Tuning parameter must be between 0 and %ld", LONG_MAX);

    if (params->tuning)
    {
        VALIDATE_FORBIDDEN(retval, params->extended_write || params->random,
                "Tuning does not work with extended write or random mode");
    }

    return retval;
}

int main(int argc, char**argv)
{
    struct ea_parameters params;
    params.inputfname = NULL;
    params.savefname = NULL;
    params.genfname = NULL;
    params.extended_write = 0;
    params.popsize = 100;
    params.tournsize = 5;
    params.mutpb = 0.05;
    params.max_evals = 100000;
    params.modelcount = 1;
    params.verbose = 0;
    params.dont_reevaluate = 0;
    params.do_localsearch = 0;
    params.ls_k = 50;
    params.ls_only = 0;
    params.do_pi = 0;
    params.pi_width = 3;
    params.pi_threshold = 0.0;
    params.pi_size = 1;
    params.random = 0;
    params.seed = time(NULL);
    params.seed_ea = time(NULL);
    params.changelevel = 0.05;
    params.tuning = 0;
    params.quiet = 0;
    params.irace = 0;

    argp_parse(&argp, argc, argv, 0, 0, &params);

    if(!check_parameters(&params))
    {
        LOG_ERROR(
                "The provided parameters are invalid, see explanation before. "
                "Exiting.");
        return -1;
    }

    if(!params.random && !params.dont_reevaluate && (params.popsize >= (params.max_evals / params.modelcount)))
    {
        LOG_WARN(&params,
            "Population size is greater than "
            "the amount of evaluations in one model change interval, "
            "setting '--dont-reevaluate' to true!\n"
            "\tIf not desired, decrease popsize or amount of models!");
        params.dont_reevaluate = 1;
        fflush(stdout);
        sleep(2);
    }

    if(params.popsize < params.tournsize)
    {
        LOG_WARN(&params,
            "popsize is smaller than tournsize, "
            "resetting tournsize to popsize %ld\n", params.popsize);
        params.tournsize = params.popsize;
    }

    srand(params.seed_ea);
    srand48(params.seed);

    if(params.irace)
    {
        time_t start_time = time(NULL);
        Fitness retVal = run_default(&params);
        time_t end_time = time(NULL);
        printf("\n%ld %ld\n", retVal, end_time - start_time);
        return 0;
    }
    else if(params.random)
        run_random(&params);
    else if(params.tuning)
        run_meta_ea_tuning(&params);
    else
        run_default(&params);
    return 0;
}
