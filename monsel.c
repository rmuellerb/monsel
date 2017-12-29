#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <argp.h>

// TODO: Externalize model change
// TODO: Only save fitvals (+ allocate memory) if savefname is present
// TODO: Integrate continuous and default version
// TODO: Switch to base + diff files in default mode

// typedefs and structs
typedef unsigned char Gene;
typedef long Fitness;

typedef struct Fitness_ext {
    Fitness fitness;
    long ecount;
    long vcount;
    long edges_weighted;
    long edges_unweighted;
    long n_mons;
} Fitness_ext;

typedef struct Generation_data {
    double mean_edges_weighted;
    double mean_edges_unweighted;
    double mean_n_mons;
    double std_edges_weighted;
    double std_edges_unweighted;
    double std_n_mons;
} Generation_data;

typedef struct Vertex {
    int id;
    int active;
} Vertex;

typedef struct Edge {
    int src;
    int dst;
    int w;
    int active;
} Edge;

typedef struct NetworkModel {
    long id;
    Edge *edges;
    Vertex *vertices;
    long ecount;
    long vcount;
} NetworkModel;

typedef struct Individual {
    Gene *values;
    long size;
    Fitness fitness;
} Individual;

typedef struct Population {
    Individual* ind;
    long size;
    long _memsize;
} Population;

typedef struct Diversity {
    double *values;
    long next;
    long _memsize;
} Diversity;

struct ea_parameters {
    // program
    char* inputfname;
    char* savefname;
    char* genfname;
    int filecheck;
    int extended_write;
    int verbose;
    // ea
    int continuous;
    long popsize;
    long tournsize;
    double mutpb;
    long max_evals;
    long model_changes;
    // ls
    int do_localsearch;
    long ls_k;
    // pi
    int do_pi;
    long pi_width;
    double pi_threshold;
    long pi_size;
};

// HEADER

Fitness fitness(Individual *ind, NetworkModel *model);

const char *argp_program_version = "0.5";
const char *argp_program_bug_address = "<mueller-bady@linux.com>";
static char doc[] = "Program to run (LS + PI) EA experiments and writing fitness value to given outputfile. \nIn case of the default (non-continuous) optimization, the inputfile(s) is/are given as *.csv having the format [src,dst,w].\nIn case of the continuous optimization ('-c' or '--continuous' switch), the inputfiles are given as base file (format: [E,src,dst,w] and [V,id]) and the diff files containing ids (nodes) or src and dst (edges) of elements being inactive.";
static char args_doc[] = "[network.csv]";

static struct argp_option options[] =
{
    {0, 0, 0, 0, "EA parameters"},
    {"popsize", 'p', "POPSIZE", 0, "Size of the population (default: 100)"},
    {"tournsize", 't', "TOURNSIZE", 0, "Size of the tournament (default: 5)"},
    {"mutpb", 'm', "MUTPB", 0, "Mutation probability per gene (default: 0.05)"},
    {"nevals", 'n', "MAX_NEVALS", 0, "Maximum number of evaluations (default: 100,000)"},
    {"model-change", 'z', "AMOUNT", 0, "Amount of model changes during runtime (default: 0). Requires a given base model and subsequent numbered models in the same directory, e.g., NREN_base.csv, NREN_1.csv, NREN_2.csv, etc. Ignored if '-c' or '--continuous' is used."},
    {0, 0, 0, 0, "Localsearch parameters"},
    {"ls", 'l', 0, 0, "perform localsearch (default: no)"},
    {"ls_k", 'k', "K-VALUE", 0, "localsearch k-parameter (default: 0)"},
    {0, 0, 0, 0, "Population injection parameters"},
    {"pi", 'i', 0, 0, "perform population injection (default: no)"},
    {"pi-width", 's', "WIDTH", 0, "injection width (default: 1)"},
    {"pi-threshold", 'y', "THRESHOLD", 0, "injection threshold (default: 0)"},
    {"pi-size", 'o', "SIZE", 0, "injection size (default: 100)"},
    {0, 0, 0, 0, "Program parameters"},
    {"verbose", 'v', 0, 0, "Switch on verbose mode (default: no)"},
    {"extended-write", 'e', 0, 0, "Switch on extended fitness mode (default: no). Ignored in case '--output'/'-w' is not given."},
    {"filecheck", 'f', 0, 0, "Switch on model checking before run (default: no)"},
    {"continuous", 'c', 0, 0, "Runs as continuous optimization, i.e., change model every evaluation. (default: no)"},
    {"output", 'w', "FILE", 0, "Fitness value output file (one value per line)"},
    {"gen-output", 'q', "FILE", 0, "Extended fitness value output file for the whole generation (one generation per line using the given FILE)"},
    {0}
};

static error_t parse_opt(int key, char *arg, struct argp_state *state)
{
    struct ea_parameters *params = state->input;
    switch(key)
    {
        case 'v': params->verbose = 1; break;
        case 'c': params->continuous = 1; break;
        case 'f': params->filecheck = 1; break;
        case 'e': params->extended_write = 1; break;
        case 'p': params->popsize = arg ? atol(arg) : 111; break;
        case 'w': params->savefname = arg; break;
        case 'q': params->genfname = arg; break;
        case 't': params->tournsize = arg ? atol(arg) : 5; break;
        case 'm': params->mutpb = arg ? atof(arg) : 0.05; break;
        case 'n': params->max_evals = arg ? atol(arg) : 100000; break;
        case 'l': params->do_localsearch = 1; break;
        case 'k': params->ls_k = arg ? atol(arg) : 0; break;
        case 'i': params->do_pi = 1; break;
        case 's': params->pi_width = arg ? atol(arg) : 0; break;
        case 'y': params->pi_threshold = arg ? atof(arg) : 0.0; break;
        case 'o': params->pi_size = arg ? atol(arg) : 0; break;
        case 'z': params->model_changes = arg ? atol(arg) : 0; break;
        case ARGP_KEY_END: if(state->arg_num < 1) argp_usage(state); break;
        case ARGP_KEY_ARG: if(state->arg_num > 1) argp_usage(state); else params->inputfname = arg; break;
        default: return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

static struct argp argp = {options, parse_opt, args_doc, doc};

// functions

/*
 * Calculates the standard deviation of an arbitrary long array
 * For performance reasons, the mean should be precalculated and given.
 */
double long_stdev(long *arr, int size, double mean)
{
    double sum = 0;
    for(int i=0; i<size; i++)
        sum += pow(arr[i] - mean, 2);
    return sqrt(sum / (double)size);
}

/*
 * Calculates mean of an arbitrary long array
 */
double long_mean(long *arr, int size)
{
    long sum = 0;
    for(int i=0; i<size; i++)
        sum += arr[i];
    return (sum / (double)size);
}

/*
 * Replace the characters "base" according to a given model sequence number 'netnumber'.
 * Returns a pointer to the new string or NULL in case no base file is given.
 */
char *replace_model(char *base, long netnumber)
{
    char newnumber[12];
    sprintf(newnumber, "%ld", netnumber);
    //TODO: Buffer Overflow or errors if filenames are much longer
    static char buffer[1000];
    char *p;

    if(!(p = strstr(base, "base")))  // Is 'orig' even in 'str'?
    {
        printf("ERROR: Please give file in the format \"network_base.csv\" when using \"-z\" or \"-c\" switch\n");
        return NULL;
    }

    strncpy(buffer, base, p-base); // Copy characters from 'str' start to 'orig' st$
    buffer[p-base] = '\0';
    sprintf(buffer+(p-base), "%s%s", newnumber, p+strlen("base"));
    return buffer;
}
/*
 * Creates an array of n distinct integer values in the range from 0 to size
 */
void distinct_random_values(int *values, long size)
{
    long i;
    for(i=0; i<size; i++)
        values[i] = i;
    for(i=size-1; i>=0; i--)
    {
        int swapto = rand() % size;
        int tmp = values[swapto];
        values[swapto] = values[i];
        values[i] = tmp;
    }
}

/*
 * Checks whether value is contained in array of given size.
 * Returns 1 if value is contained, 0 if not.
 */
int contains(int *array, int size, int value)
{
    if(!size) return 0;
    long i;
    for(i=0; i<size; i++)
        if(array[i] == value) return 1;
    return 0;
}

/*
 * Checks for the best individual in the population and returns the index to it.
 */
long best_individual_idx(Population *pop)
{
    long j, idx = 0;
    for(j=1; j<pop->size; j++)
    {
        if(pop->ind[j].fitness < pop->ind[idx].fitness)
            idx = j;
    }
    return idx;
}

/*
 * Reads the diff file to a previously given model
 * Returns the number of read diffs (vertices + edges)
 */
int read_diff_file(NetworkModel* model, const char* fname)
{
    // Set all vertices/edges to active
    // and increase model ID
    int i;
    for(i=0; i<model->vcount; i++)
        model->vertices[i].active = 1;
    for(i=0; i<model->ecount; i++)
        model->edges[i].active = 1;
    model->id += 1;

    // Read diff from file
    FILE *fp;
    char* line = NULL;
    size_t len = 0;
    ssize_t read;
    int diffs = 0;
    fp = fopen(fname, "r");
    if (fp == NULL)
        return -1;
    while((read = getline(&line, &len, fp)) != -1)
    {
        char* class = strtok(line, ",");
        if((*class) == 'E')
        {
            int src = atoi(strtok(NULL, ","));
            int dst = atoi(strtok(NULL, ","));
            for(i=0; i<model->ecount; i++)
            {
                if(model->edges[i].src == src && model->edges[i].dst == dst)
                {
                    model->edges[i].active = 0;
                    diffs += 1;
                    break;
                }
            }
        }
        else if((*class) == 'V')
        {
            int id = atoi(strtok(NULL, ","));
            model->vertices[id].active = 0;
            diffs += 1;
        }
        else
            printf("WARNING: Read line without prefix 'E' or 'V'\n");
    }
    fclose(fp);
    if(line) free(line);
    //printf("Read network model '%d' diff with %d diffs\n", model->id, diffs);
    return diffs;
}

/*
 * Reads the given file according to its filename and fills the edges into the NetworkModel.
 * The model will be filled with the number of vertices (vcount) the number of edges (ecount),
 * an array of edges containing edge information (src, dst, weight, active) and an array of vertices.
 * Returns the number of lines read or -1 in case of failure
 */
int read_full_file(NetworkModel *model, const char* fname)
{
    FILE *fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    long j=0, i=0, ecount = 0, vcount = 0;
    char* tmp;
    fp = fopen(fname, "r");
    if (fp == NULL)
        return -1;
    while ((read = getline(&line, &len, fp)) != -1) {
        if(line[0] == 'E')
            ecount++;
        else
            vcount++;
    }
    model->id = -1;
    model->ecount = ecount;
    model->vcount = vcount;
    rewind(fp);
    Vertex *vertices;
    vertices = malloc(sizeof(Vertex) * vcount);
    Edge *edges;
    edges = malloc(sizeof(Edge) * ecount);
    while ((read = getline(&line, &len, fp)) != -1) {
        char* class = strtok(line, ",");
        if((*class) == 'E')
        {
            edges[i].active = 1;
            edges[i].src = atoi(strtok(NULL, ","));
            edges[i].dst = atoi(strtok(NULL, ","));
            edges[i].w = atoi(strtok(NULL, ","));
            i++;
        }
        else if((*class) == 'V')
        {
            vertices[j].id = atoi(strtok(NULL, ","));
            if((tmp = strtok(NULL, ",")) != NULL)
            {
                if(strcmp(tmp, "active\n") == 0)
                    vertices[j].active = 1;
                else
                    vertices[j].active = 0;
            }
            else
                vertices[j].active = 1;
            j++;
        }
    }
    model->edges = edges;
    model->vertices = vertices;
    fclose(fp);
    if (line) free(line);
    return ecount + vcount;
}

/*
 * Writes header containing timing and PI information to file.
 * If file exists, appending to existing file.
 * Returns 1 if successful, -1 if not
 */
int write_meta_header(time_t timing, int pi_trigger, char* fname)
{
    FILE *f;
    f = fopen(fname, "a");
    if(f==NULL)
    {
        printf("Unable to open file '%s' for writing.\n", fname);
        return -1;
    }
    fprintf(f, "%ld,%d\n", timing, pi_trigger);
    fclose(f);
    return 1;
}

/*
 * Writes given Fitness_ext values into given textfile, one per line.
 * If file exists, appending to existing file.
 * Returns 1 if successful, -1 if not
 */
int write_fitness_ext(Fitness_ext *values, int size, char* fname)
{
    FILE *f;
    long i;
    f = fopen(fname, "a");
    if(f==NULL)
    {
        printf("Unable to open file '%s' for writing.\n", fname);
        return -1;
    }
    for(i=0; i<size; i++)
        fprintf(f, "%ld,%ld,%ld,%ld,%ld,%ld\n", values[i].fitness, values[i].ecount, values[i].edges_unweighted, values[i].edges_weighted, values[i].vcount, values[i].n_mons);
    fclose(f);
    return 1;
}

/*
 * Writes all Generation_data values file as *.csv.
 * If file exists, appending to existing file.
 * Returns 1 if successful, -1 if not
 */
int write_generation_data(Generation_data* data, int size, char* fname)
{
    FILE *f;
    f = fopen(fname, "a");
    if(f==NULL)
    {
        printf("Unable to open file '%s' for writing.\n", fname);
        return -1;
    }
    fprintf(f, "mean weighted edges,mean unweighted edges,mean nmons,std weighted edges,std unweighted edges,std nmons\n");
    for(int i=0; i<size; i++)
    {
        fprintf(f, "%f,%f,%f,%f,%f,%f\n", data[i].mean_edges_weighted, data[i].mean_edges_unweighted, data[i].mean_n_mons, data[i].std_edges_weighted, data[i].std_edges_unweighted, data[i].std_n_mons);
    }
    fclose(f);
    return 1;
}


/*
 * Writes all fitness values of the given population to a file (in one line) as *.csv.
 * Evaluation is done on demand while writing and does not count to the evaluation count.
 * If file exists, appending to existing file.
 * Returns 1 if successful, -1 if not
 */
int write_generation_fitness(Population* pop, NetworkModel* model, char* fname)
{
    FILE *f;
    long i;
    f = fopen(fname, "a");
    if(f==NULL)
    {
        printf("Unable to open file '%s' for writing.\n", fname);
        return -1;
    }
    for(i=0; i<pop->size; i++)
    {
        if(i == (pop->size - 1))
            fprintf(f, "%ld\n", fitness(&pop->ind[i], model));
        else
            fprintf(f, "%ld,", fitness(&pop->ind[i], model));
    }
    fclose(f);
    return 1;
}

/*
 * Writes given fitness values into given textfile, one per line.
 * If file exists, appending to existing file.
 * Returns 1 if successful, -1 if not
 */
int write_fitness(Fitness *values, int size, char* fname)
{
    FILE *f;
    long i;
    f = fopen(fname, "a");
    if(f==NULL)
    {
        printf("Unable to open file '%s' for writing.\n", fname);
        return -1;
    }
    for(i=0; i<size; i++)
    {
        fprintf(f, "%ld\n", values[i]);
    }
    fclose(f);
    return 1;
}

/*
 * Returns a random float value from 0 to 'to'.
 */
float frand(float to)
{
    return ((float)rand()/(float)(RAND_MAX)) * to;
}

/*
 * Counts and returns the number of active edges
 */
long active_edges(NetworkModel *model)
{
    long edges = 0;
    for(long i=0; i<model->ecount; i++)  
        if(model->edges[i].active)
            edges += 1;
    return edges;
}

/*
 * Counts and returns the number of active nodes
 */
long active_nodes(NetworkModel *model)
{
    long nodes = 0;
    for(long i=0; i<model->vcount; i++)   
        if(model->vertices[i].active)
            nodes += 1;
    return nodes;
}

/*
 * Calculates and returns the number of uncovered edges according to the given edge model and individual.
 */
int uncovered_edges(Individual *ind, NetworkModel *model)
{
    int u_edges = 0;
    for(long i=0; i<model->ecount; i++)
    {
        if(!model->edges[i].active)
            continue;
        if(ind->values[model->edges[i].src] == 0 && ind->values[model->edges[i].dst] == 0)
            u_edges += 1;
    }
    return u_edges;
}

/*
 * Calculates and returns the penalty according to the given edge model and individual.
 */
int penalty(Individual *ind, NetworkModel *model)
{
    int pen = 0;
    int factor = 2;
    for(long i=0; i<model->ecount; i++)
    {
        if(ind->values[model->edges[i].src] == 0 && ind->values[model->edges[i].dst] == 0)
        {
            if(model->edges[i].active)
                pen = pen + (factor * model->edges[i].w);
        }
    }
    return pen;
}

/*
 * Calculates and returns the fitness value struct, containing the fitness value, the 
 * number of uncovered edges and the number of monitors according to the given
 * edge model and individual.
 */
Fitness_ext fitness_ext(Individual *ind, NetworkModel *model)
{
    Fitness_ext retval;
    retval.n_mons = 0;
    for(long i=0; i<ind->size; i++)
        if(model->vertices[i].active)
            retval.n_mons += ind->values[i];
    retval.edges_weighted = penalty(ind, model);
    retval.fitness = retval.n_mons + retval.edges_weighted;
    retval.edges_unweighted = uncovered_edges(ind, model);
    retval.vcount = active_nodes(model);
    retval.ecount = active_edges(model);
    return retval;
}

/*
 * Calculates and returns the fitness values according to the given edge model and individual.
 * The fitness value already includes the penalty value.
 */
Fitness fitness(Individual *ind, NetworkModel *model)
{
    Fitness pen, fit = 0;
    for(long i=0; i<ind->size; i++)
    {
        if(!model->vertices[i].active)
            continue;
        fit = fit + ind->values[i];
    }
    pen = penalty(ind, model);
    return fit + pen;
}

/*
 * Creates an initialized individual with 0 values of'size' size and a fitness of -1 (unitialized).
 */
void create_null_individual(Individual *ind, int size)
{
    ind->values = malloc(sizeof(Gene) * size);
    long i;
    for(i=0; i<size; i++)
    {
        ind->values[i] = 0;
    }
    ind->size = size;
    ind->fitness = -1;
}

/*
 * Creates an initialized individual with random values of size 'size' and a fitness of -1 (unititialized).
 */
void create_random_individual(Individual *ind, int size)
{
    ind->values = malloc(sizeof(Gene) * size);
    for(int i=0; i<size; i++)
    {
        ind->values[i] = rand() % 2;
    }
    ind->size = size;
    ind->fitness = -1;
}

/*
 * Creates a Generation_data struct filled with the information about the given generation and network model
 */
Generation_data create_generation_data(Population* pop, NetworkModel* model)
{
    long edges_weighted[pop->size];
    long edges_unweighted[pop->size];
    long n_mons[pop->size];
    for(int i=0; i<pop->size; i++)
    {
        Fitness_ext tmp = fitness_ext(&pop->ind[i], model);
        edges_weighted[i] = tmp.edges_weighted;
        edges_unweighted[i] = tmp.edges_unweighted;
        n_mons[i] = tmp.n_mons;

    }
    Generation_data data;
    data.mean_edges_weighted = long_mean(edges_weighted, pop->size);
    data.mean_edges_unweighted = long_mean(edges_unweighted, pop->size);
    data.mean_n_mons = long_mean(n_mons, pop->size);
    data.std_edges_weighted = long_stdev(edges_weighted, pop->size, data.mean_edges_weighted);
    data.std_edges_unweighted = long_stdev(edges_unweighted, pop->size, data.mean_edges_unweighted);
    data.std_n_mons = long_stdev(n_mons, pop->size, data.mean_n_mons);
    return data;
}

/*
 * Frees the memory of the given individual
 */
void free_individual(Individual *ind)
{
    free(ind->values);
}

/*
 * Frees the memory for the whole population
 */
void free_population(Population *pop)
{
    long i;
    for(i=0; i<pop->size; i++)
        free_individual(&pop->ind[i]);
    free(pop->ind);
}

/*
 * Frees the memory of the given NetworkModel
 */
void free_model(NetworkModel *model)
{
    free(model->edges);
    free(model->vertices);
}

/*
 * Print the configuration of the experiment
 */
void print_config(struct ea_parameters *params)
{
    printf("<Experiment parameters>\n");
    printf("\tInputfile:\t\t%s\n", params->inputfname);
    printf("\tOutputfile fitness:\t%s\n", params->savefname ? params->savefname : "not saved");
    printf("\tOutputfile generation:\t%s\n", params->genfname ? params->genfname : "not saved");    
    printf("\tExtended Write:\t\t%s\n", params->extended_write ? "yes" : "no");
    printf("\tPopsize:\t\t%ld\n", params->popsize);
    printf("\tTournsize:\t\t%ld\n", params->tournsize);
    printf("\tMutationprob:\t\t%.4f\n", params->mutpb);
    printf("\tMax evals:\t\t%ld\n", params->max_evals);
    printf("\tModel changes:\t\t%ld\n", params->model_changes);
    printf("\tVerbose:\t\t%s\n", params->verbose ? "yes" : "no");
    printf("\tFilecheck:\t\t%s\n", params->filecheck ? "yes" : "no");
    printf("\tContinuous:\t\t%s\n", params->continuous ? "yes" : "no");
    printf("\tLocalsearch:\t\t%s\n", params->do_localsearch ? "yes" : "no");
    if(params->do_localsearch)
        printf("\tLocalsearch k-value:\t%ld\n", params->ls_k );
    printf("\tPopulation Injection:\t%s\n", params->do_pi ? "yes" : "no");    
    if(params->do_pi)
    {
        printf("\tInjection width:\t%ld\n", params->pi_width);
        printf("\tInjection threshold:\t%.4f\n", params->pi_threshold);
        printf("\tInjection size:\t\t%ld\n", params->pi_size);
    }
    printf("</Experiment parameters>\n");
}

/*
 * Prints the string representation of the individual including values and fitness value.
 */
void print_individual(Individual *ind)
{
    printf("Individual: [");
    long i;
    for(i=0; i<ind->size; i++)
    {
        if(i) printf(" ");
        printf("%d", ind->values[i]);
    }
    printf("], Fitness: %ld (%p)\n", ind->fitness, ind);
}

/*
 * Performs a bitflip mutation with probability 'p' on each gene of individual 'ind'.
 * Returns the number of flipped bits
 */
int bitflip_mutation(Individual *ind, float p)
{
    long i, c=0;
    for(i = 0; i<ind->size; i++)
    {
        if(frand(1.0) < p)
        {
            c++;
            ind->values[i] = !ind->values[i];
        }
    }
    return c;
}

/*
 * Performs an uniform crossover on parents 'p1' and 'p2' and saves the result in children 'c1' and 'c2'.
 */
void uniform_crossover(Individual *p1, Individual *p2, Individual *c1, Individual *c2)
{
    long i;
    for(i = 0; i<p1->size; i++)
    {
        if(rand() % 2)
        {
            c1->values[i] = p1->values[i];
            c2->values[i] = p2->values[i];
        }
        else
        {
            c1->values[i] = p2->values[i];
            c2->values[i] = p1->values[i];
        }
    }
}

/*
 * Performs a tournament selection on population 'pop'.
 * Parameter 'tournsize' determines the size of the tournament in which the individuals participate.
 * Returns the index of the selected individual within the population.
 * TODO: implement finding distinct individuals
 */
int tournament_selection(Population *pop, int tournsize)
{
    int best = rand() % pop->size;
    long i;
    for(i=1; i<tournsize; i++)
    {
        int choice = rand() % pop->size;
        if(pop->ind[choice].fitness < pop->ind[best].fitness)
            best = choice;
    }
    return best;
}


/*
 * Calculates and returns the mean of the fitness of population 'pop' of size 'size'.
 */
double population_fitness_mean(Population *pop)
{
    Fitness sum = 0;
    long i;
    for(i=0; i<pop->size; i++)
        sum += pop->ind[i].fitness;
    return (sum / (double)pop->size);
}

/*
 * Calculates and returns the minimum of the fitness of population 'pop' of size 'size'.
 */
double population_fitness_min(Population *pop)
{
    double min = pop->ind[0].fitness;
    int i;
    for(i=1; i<pop->size; i++)
    {
        if(pop->ind[i].fitness < min)
            min = pop->ind[i].fitness;
    }
    return min;
}

/*
 * Calculates and returns the maximum of the fitness of population 'pop' of size 'size'.
 */
double population_fitness_max(Population *pop)
{
    double max = pop->ind[0].fitness;
    long i;
    for(i=1; i<pop->size; i++)
    {
        if(pop->ind[i].fitness > max)
            max = pop->ind[i].fitness;
    }
    return max;
}


/*
 * Calculates mean of an arbitrary double array
 */
double fitness_mean(double *arr, int size)
{
    double sum = 0;
    for(int i=0; i<size; i++)
        sum += arr[i];
    return (sum / (double)size);
}



/*
 * Calculates the standard deviation of an arbitrary double array
 */
double fitness_stdev(double *arr, int size)
{
    int i;
    double sum = 0;
    double mean = fitness_mean(arr, size);
    for(i=0; i<size; i++)
        sum += pow(arr[i] - mean, 2);
    return sqrt(sum / (double)size);
}

/*
 * Calculates and returns the standard deviation of the fitness of population 'pop' of size 'size'.
 */
double population_fitness_stdev(Population *pop)
{
    long i;
    double sum = 0;
    double m = population_fitness_mean(pop);
    for(i=0; i<pop->size; i++)
        sum += pow(pop->ind[i].fitness - m, 2);
    return sqrt(sum / (double)pop->size);
}

/*
 * Prints the statistics of the given population 'pop' of size 'size' for generation 'gen'
 */
void print_stats(Population *pop, int gen, long nevals, long nevals_ls, int print_ls, int print_pi)
{
    printf("gen %4d", gen);
    printf(" - nevals %4ld", nevals);
    if(print_ls)
        printf(" - ls nevals %4ld", nevals_ls);
    printf(" - mean %4.2f", population_fitness_mean(pop));
    printf(" - stdev %4.2f", population_fitness_stdev(pop));
    printf(" - min %4.2f", population_fitness_min(pop));
    printf(" - max %4.2f", population_fitness_max(pop));
    if(print_pi)
        printf(" - PI %4d", print_pi);
    printf("\n");
}

/*
 * Print the current network model
 */
void print_model(NetworkModel *model)
{
    int i;
    printf("<NetworkModel>\n");
    printf("\t<ID>%ld</ID>\n", model->id);
    printf("\t<Vcount>%ld</Vcount>\n", model->vcount);
    printf("\t<Ecount>%ld</Ecount>\n", model->ecount);
    printf("\t<Vertices>\n");
    for(i=0; i<model->vcount; i++)
        printf("\t\t<Vertex id=%d active=%d />\n", model->vertices[i].id, model->vertices[i].active);
    printf("\t</Vertices>\n");
    printf("\t<Edges>\n");
    for(i=0; i<model->ecount; i++)
        printf("\t\t<Edge src=%d dst=%d w=%d active=%d/>\n", model->edges[i].src, model->edges[i].dst, model->edges[i].w, model->edges[i].active);
    printf("\t</Edges>\n");
    printf("</NetworkModel>\n");
}

/*
 * Clean up the memory used by the population and the model
 */
int cleanup(Population *pop, NetworkModel *model, Fitness *fitvals, Diversity *div, Fitness_ext *fitvals_ext)
{
    free_population(pop);
    free_model(model);
    if(fitvals_ext)
        free(fitvals_ext);
    if(fitvals)
        free(fitvals);
    if(div->values)
        free(div->values);
    return 1;
}

/*
 * Performs check if all necessary files are present (from basefile_0 to basefile_[changes])
 * Returns 1 if all files are accessible and have a proper format, -1 otherwise
 */
int check_model_files(char* basefile, int changes, int verbose, int continuous)
{
    int i;
    int check;
    // First read basefile
    NetworkModel base;
    if(read_full_file(&base, basefile) == -1)
    {
        if(verbose)
            printf("ERROR: Reading file \"%s\" failed!\n", basefile);
        free_model(&base);
        return -1;
    }
    // Then, check all other files
    for(i=0; i<=changes; i++)
    {
        char* fname = replace_model(basefile, i);
        if(!fname)
        {
            free_model(&base);
            return -1;
        }

        if(continuous)
            check = read_diff_file(&base, fname);
        else
        {
            NetworkModel model;
            check = read_full_file(&model, fname);
            free_model(&model);
        }
        if(check == -1)
        {
            if(verbose)
                printf("ERROR: Reading file \"%s\" failed!\n", fname);
            free_model(&base);
            return -1;
        }
    }
    free_model(&base);
    return 1;
}

/*
 * Checks whether population injection needs to be triggered
 * Returns 1 if yes, 0 if not
 */
int pi_necessary(Diversity *div, double threshold)
{
    return fitness_stdev(div->values, div->_memsize) < threshold;
}

/*
 * Performs a local search on 1 individual.
 * For the neighborhood, 'k' individuals are considered.
 * The process is iteratively repeated until no better neighbor was found.
 * Returns the new number of evaluations to the caller, which can then calculate the
 * number of evaluations of the LS method or use it for further processing.
 * Returns in case we have a model change or the maximum number of allowed evaluations is hit.
 */
long localsearch(Individual *ind, int k, NetworkModel *model, Fitness *fitvals, long nevals, long max_evals, long change_evals, int continuous, char* inputfname, Fitness_ext *fitvals_ext)
{
    if(ind->size < k)
    {
        printf("WARNING: k-value too large for element, reducing to individuals genome size: %ld\n", ind->size);
        k = ind->size;
    }
    int neighbors[ind->size];
    long i;
    int foundbetter;
    Fitness best_fitness = ind->fitness;
    do
    {
        distinct_random_values(neighbors, ind->size);
        foundbetter = 0;
        for(i=0; i<k; i++)
        {
            ind->values[neighbors[i]] = !ind->values[neighbors[i]];
            Fitness fit = fitness(ind, model);
            if(fitvals_ext)
                fitvals_ext[nevals++] = fitness_ext(ind, model);
            else if(fitvals)
                fitvals[nevals++] = fit;
            else
                nevals++;
            if(continuous)
                read_diff_file(model, replace_model(inputfname, model->id + 1));
            if(fit >= best_fitness)
                ind->values[neighbors[i]] = !ind->values[neighbors[i]];
            else
            {
                ind->fitness = fit;
                best_fitness = fit;
                foundbetter = 1;
            }
            // TODO: Following statement is improvable :-)
            if(continuous)
            {
                if(nevals >= max_evals)
                    return nevals;
            }
            else
            {
                if(nevals >= max_evals || (nevals % change_evals) == 0)
                    return nevals;
            }                
        }
    }while(foundbetter);
    return nevals;
}

/*
 * Runs the continuous version of the monitor selection optimizer
 */
int run_continuous(struct ea_parameters* params)
{
    long i = 0;
    if(params->filecheck)
    {    
        if(params->verbose)
            printf("checking model files...\n");
        if(check_model_files(params->inputfname, params->max_evals, params->verbose, params->continuous) == -1)
        {
            printf("ERROR: not all files are present, aborting!\n");
            exit(EXIT_FAILURE);
        }
    }
    NetworkModel model;
    if(read_full_file(&model, params->inputfname) == -1)
    {
        printf("Error reading file '%s', aborting\n", params->inputfname);
        exit(EXIT_FAILURE);
    }
    if(model.ecount == 0 || model.vcount == 0)
    {
        printf("Either no edge or vertex in file '%s', aborting\n", params->inputfname);
        exit(EXIT_FAILURE);
    }
    if(read_diff_file(&model, replace_model(params->inputfname, 0)) == -1)
    {
        printf("Error reading file '%s', aborting\n", replace_model(params->inputfname, 0));
        exit(EXIT_FAILURE);
    }
    if(params->verbose)
    {
        print_config(params);
        int active = 0;
        int k;
        for(k=0; k<model.vcount; k++)
            if(model.vertices[k].active)
                active++;
        printf("Read file '%s' having |V| = %ld (%d active), |E| = %ld @ %d evals\n", replace_model(params->inputfname, 0), model.vcount, active, model.ecount, 0);
    }
    Fitness* fitvals = NULL;
    Fitness_ext* fitvals_ext = NULL;
    Generation_data* gendata = NULL;
    int gendatacount = 0;
    if(params->savefname)
    {
        if(params->extended_write)
            fitvals_ext = malloc(sizeof(Fitness_ext) * params->max_evals);
        else
            fitvals = malloc(sizeof(Fitness) * params->max_evals);
    }
    if(params->genfname)
    {
        gendata = malloc(sizeof(Generation_data) * params->max_evals);
    }
    int nevals = 0;
    Population pop;
    pop._memsize = params->popsize + params->popsize + params->pi_size;
    pop.size = 0;
    pop.ind = malloc(sizeof(Individual) * pop._memsize);
    time_t run_start_time = time(NULL);
    //#pragma omp parallel for
    for(i=0; i<params->popsize; i++)
    {
        Individual ind;
        create_random_individual(&ind, model.vcount);
        pop.ind[pop.size++] = ind;
    }
    for(i=0; i<pop.size; i++)
    {
        pop.ind[i].fitness = fitness(&pop.ind[i], &model);
        if(params->genfname)
            gendata[gendatacount++] = create_generation_data(&pop, &model);
        read_diff_file(&model, replace_model(params->inputfname, model.id + 1));
        if(params->savefname)
        {
            if(params->extended_write)
                fitvals_ext[nevals++] = fitness_ext(&pop.ind[i], &model);
            else
                fitvals[nevals++] = pop.ind[i].fitness;
        }
        else
            nevals++;
    }
    // Variables for PI
    Diversity diversity;
    diversity.values = NULL;
    diversity._memsize = 0;
    if(params->do_pi)
    {
        diversity._memsize = params->pi_width;
        diversity.values = malloc(sizeof(Fitness) * diversity._memsize);
        int k;
        for(k=0; k<diversity._memsize; k++)
            diversity.values[k] = 0;
        diversity.next = 0;
    }

    // print initial population statistics
    if(params->verbose)
        print_stats(&pop, 0, nevals, 0, params->do_localsearch, 0);

    long pi_triggered = 0;
    // start of the generational process
    for(i=0; ; i++)
    {
        int j;
        int injected = 0;
        long ls_nevals = 0;
        // local search
        if(params->do_localsearch)
        {
            if(nevals >= params->max_evals)
            {
                if(params->verbose)
                {
                    printf("Final: ");
                    print_stats(&pop, i+1, nevals, ls_nevals, params->do_localsearch, injected);
                    printf("Reached %d evaluations, quitting after %ld seconds\n", nevals, time(NULL) - run_start_time);
                }
                if(params->savefname) 
                {
                    write_meta_header(time(NULL) - run_start_time, pi_triggered, params->savefname);
                    if(params->extended_write)
                        write_fitness_ext(fitvals_ext, nevals, params->savefname);
                    else
                        write_fitness(fitvals, nevals, params->savefname);
                }
                if(params->genfname)
                {
                    write_generation_data(gendata, gendatacount, params->genfname);
                    free(gendata);
                }
                cleanup(&pop, &model, fitvals, &diversity, fitvals_ext);
                return 0;
            }
            long idx = best_individual_idx(&pop);
            ls_nevals = nevals;
            nevals = localsearch(&pop.ind[idx], params->ls_k, &model, fitvals, nevals, params->max_evals, 1, params->continuous, params->inputfname, fitvals_ext);
            ls_nevals = nevals - ls_nevals;
        }
        // choose parents and perform crossover + mutation on children
        for(j=0; j<params->popsize; j = j + 2)
        {
            if(nevals >= params->max_evals)
            {
                if(params->verbose)
                {
                    printf("Final: ");
                    print_stats(&pop, i+1, nevals, ls_nevals, params->do_localsearch, injected);
                    printf("Reached %d evaluations, quitting after %ld seconds\n", nevals, time(NULL) - run_start_time);
                }
                if(params->savefname) 
                {
                    write_meta_header(time(NULL) - run_start_time, pi_triggered, params->savefname);
                    if(params->extended_write)
                        write_fitness_ext(fitvals_ext, nevals, params->savefname);
                    else
                        write_fitness(fitvals, nevals, params->savefname);
                }
                if(params->genfname)
                {
                    write_generation_data(gendata, gendatacount, params->genfname);
                    free(gendata);
                }
                cleanup(&pop, &model, fitvals, &diversity, fitvals_ext);
                return 0;
            }
            int p1;
            int p2;
            do
            {
                p1 = tournament_selection(&pop, params->tournsize);
                p2 = tournament_selection(&pop, params->tournsize);
            }
            while(p1 == p2);
            Individual c1;
            create_null_individual(&c1, model.vcount);
            Individual c2;
            create_null_individual(&c2, model.vcount);
            uniform_crossover(&pop.ind[p1], &pop.ind[p2], &c1, &c2);
            bitflip_mutation(&c1, params->mutpb);
            bitflip_mutation(&c2, params->mutpb);
            c1.fitness = fitness(&c1, &model);
            if(params->genfname)
                gendata[gendatacount++] = create_generation_data(&pop, &model);
            read_diff_file(&model, replace_model(params->inputfname, model.id + 1));
            if(params->savefname)
            {
                if(params->extended_write)
                    fitvals_ext[nevals++] = fitness_ext(&c1, &model);
                else
                    fitvals[nevals++] = c1.fitness;
            }
            else
                nevals++;
            pop.ind[pop.size++] = c1;
            if(nevals >= params->max_evals)
            {
                if(params->verbose)
                {
                    printf("Final: ");
                    print_stats(&pop, i+1, nevals, ls_nevals, params->do_localsearch, injected);
                    printf("Reached %d evaluations, quitting after %ld seconds\n", nevals, time(NULL) - run_start_time);
                }
                if(params->savefname) 
                {
                    write_meta_header(time(NULL) - run_start_time, pi_triggered, params->savefname);
                    if(params->extended_write)
                        write_fitness_ext(fitvals_ext, nevals, params->savefname);
                    else
                        write_fitness(fitvals, nevals, params->savefname);
                }
                if(params->genfname)
                {
                    write_generation_data(gendata, gendatacount, params->genfname);
                    free(gendata);
                }
                cleanup(&pop, &model, fitvals, &diversity, fitvals_ext);
                return 0;
            }
            c2.fitness = fitness(&c2, &model);
            if(params->genfname)
                gendata[gendatacount++] = create_generation_data(&pop, &model);
            read_diff_file(&model, replace_model(params->inputfname, model.id + 1));
            if(params->savefname)
            {
                if(params->extended_write)
                    fitvals_ext[nevals++] = fitness_ext(&c2, &model);
                else
                    fitvals[nevals++] = c2.fitness;
            }
            else
                nevals++;
            pop.ind[pop.size++] = c2;
        }
        // Determine survivor indices using selection operation
        // TODO: How to ensure uniqueness? Floyd Algorithm?
        int surv_idx[params->popsize];
        for(j=0; j<params->popsize; j++)
        {
            int s;
            do
            {
                s = tournament_selection(&pop, params->tournsize);
            }
            while(contains(surv_idx, j, s));
            surv_idx[j] = s;
        }
        // pick survivors from the population and free dying individuals memory
        for(j=pop.size-1; j>=0; j--)
        {
            if(!contains(surv_idx, params->popsize, j))
            {
                free_individual(&pop.ind[j]);
                if(j != pop.size-1)
                    pop.ind[j] = pop.ind[pop.size-1];
                pop.size--;
            }
        }
        // Place for possible population injection
        if(params->do_pi)
        {
            diversity.values[diversity.next++] = population_fitness_stdev(&pop);
            diversity.next = diversity.next % diversity._memsize;
            if(pi_necessary(&diversity, params->pi_threshold))
            {
                pi_triggered += 1;
                int k;
                for(k=0; k<params->pi_size; k++)
                {
                    if(nevals >= params->max_evals)
                        break;
                    Individual ind;
                    create_random_individual(&ind, model.vcount);
                    ind.fitness = fitness(&ind, &model);
                    if(params->genfname)
                        gendata[gendatacount++] = create_generation_data(&pop, &model);
                    read_diff_file(&model, replace_model(params->inputfname, model.id + 1));
                    if(params->savefname)
                    {
                        if(params->extended_write)
                            fitvals_ext[nevals++] = fitness_ext(&ind, &model);
                        else
                            fitvals[nevals++] = ind.fitness;
                    }
                    else
                        nevals++;
                    pop.ind[pop.size++] = ind;
                    injected++;
                    if(nevals >= params->max_evals)
                        break;
                }    
            }
        }
        // print statistics of the generation
        if(params->verbose) 
            print_stats(&pop, i+1, nevals, ls_nevals, params->do_localsearch, injected);
    }
    return 0;
}

/*
 * Runs the default version of the monitor selection optimizer
 */
int run_default(struct ea_parameters* params)
{
    if(params->genfname)
        printf("##### WARNING: Generation output feature not implemented yet in default EA mode! Use continuous mode to use it!\n");
    long i = 0;
    long change_eval;
    char *current_model_fname;
    if(params->model_changes == 0)
    {
        change_eval = params->max_evals+1;
        current_model_fname = params->inputfname;
    }
    else
    {
        if(params->filecheck)
        {
            if(check_model_files(params->inputfname, params->model_changes, params->verbose, params->continuous) == -1)
            {
                printf("ERROR: not all files are present, aborting!\n");
                exit(EXIT_FAILURE);
            }
        }
        change_eval = (int)(params->max_evals / params->model_changes);
        current_model_fname = replace_model(params->inputfname, 0);
        if(!current_model_fname)
        {
            printf("ERROR: Reading file failed, aborting.\n");
            exit(EXIT_FAILURE);
        }
    }
    if(change_eval <= params->popsize)
    {
        printf("ERROR: popsize is larger than model change size. Reevaluation will cost all evaluations, no LS/EA is possible. Aborting\n");
        exit(EXIT_FAILURE);
    }

    NetworkModel model;
    if(read_full_file(&model, current_model_fname) == -1)
    {
        printf("Error reading file '%s', aborting\n", current_model_fname);
        exit(EXIT_FAILURE);
    }
    if(model.ecount == 0 || model.vcount == 0)
    {
        printf("Either no edge or vertex in file '%s', aborting\n", current_model_fname);
        exit(EXIT_FAILURE);
    }
    if(params->verbose)
    {
        print_config(params);
        int active = 0;
        int k;
        for(k=0; k<model.vcount; k++)
            if(model.vertices[k].active)
                active++;
        printf("Read file '%s' having |V| = %ld (%d active), |E| = %ld @ %d evals\n", current_model_fname, model.vcount, active, model.ecount, 0);
    }
    Fitness *fitvals = NULL;
    Fitness_ext *fitvals_ext = NULL;
    if(params->savefname)
    {
        if(params->extended_write)
            fitvals_ext = malloc(sizeof(Fitness_ext) * params->max_evals);
        else
            fitvals = malloc(sizeof(Fitness) * params->max_evals);
    }
    int nevals = 0;
    Population pop;
    pop._memsize = params->popsize + params->popsize + params->pi_size;
    pop.size = 0;
    pop.ind = malloc(sizeof(Individual) * pop._memsize);
    time_t run_start_time = time(NULL);
    //#pragma omp parallel for
    for(i=0; i<params->popsize; i++)
    {
        Individual ind;
        create_random_individual(&ind, model.vcount);
        ind.fitness = fitness(&ind, &model);
        if(params->savefname)
        {
            if(params->extended_write)
                fitvals_ext[nevals++] = fitness_ext(&ind, &model);
            else
                fitvals[nevals++] = ind.fitness;
        }
        pop.ind[pop.size++] = ind;
    }
    // Variables for PI
    Diversity diversity;
    diversity.values = NULL;
    diversity._memsize = 0;
    if(params->do_pi)
    {
        diversity._memsize = params->pi_width;
        diversity.values = malloc(sizeof(Fitness) * diversity._memsize);
        int k;
        for(k=0; k<diversity._memsize; k++)
            diversity.values[k] = 0;
        diversity.next = 0;
    }

    // print initial population statistics
    if(params->verbose)
        print_stats(&pop, 0, nevals, 0, params->do_localsearch, 0);

    long pi_triggered = 0;
    // start of the generational process
    for(i=0; ; i++)
    {
        int j;
        int injected = 0;
        long ls_nevals = 0;
        // local search
        if(params->do_localsearch)
        {
            if(nevals >= params->max_evals)
            {
                if(params->verbose)
                {
                    printf("Final: ");
                    print_stats(&pop, i+1, nevals, ls_nevals, params->do_localsearch, injected);
                    printf("Reached %d evaluations, quitting after %ld seconds\n", nevals, time(NULL) - run_start_time);
                }
                if(params->savefname) 
                {
                    write_meta_header(time(NULL) - run_start_time, pi_triggered, params->savefname);
                    if(params->extended_write)
                        write_fitness_ext(fitvals_ext, nevals, params->savefname);
                    else
                        write_fitness(fitvals, nevals, params->savefname);
                }
                cleanup(&pop, &model, fitvals, &diversity, fitvals_ext);
                return 0;
            }
            long idx = best_individual_idx(&pop);
            ls_nevals = nevals;
            nevals = localsearch(&pop.ind[idx], params->ls_k, &model, fitvals, nevals, params->max_evals, change_eval, params->continuous, params->inputfname, fitvals_ext);
            ls_nevals = nevals - ls_nevals;
        }
        if((nevals % change_eval) == 0)
        {
            int next_model = (int)(nevals / change_eval);
            current_model_fname = replace_model(params->inputfname, next_model);
            free_model(&model);
            if(read_full_file(&model, current_model_fname) == -1)
            {   
                printf("Error reading file '%s', aborting\n", current_model_fname);
                exit(EXIT_FAILURE);
            }   
            if(model.ecount == 0 || model.vcount == 0)
            {   
                printf("Either no edge or vertex in file '%s', aborting\n", current_model_fname);
                exit(EXIT_FAILURE);
            }   
            if(params->verbose)
            {   
                int active = 0;
                int k;
                for(k=0; k<model.vcount; k++)
                    active += model.vertices[k].active;
                printf("Read file '%s' having |V| = %ld (%d active), |E| = %ld @ %d evals\n", current_model_fname, model.vcount, active, model.ecount, nevals);
            }
            int k;
            for(k=0; k<pop.size; k++)
            {
                if(nevals >= params->max_evals)
                    break;
                pop.ind[k].fitness = fitness(&pop.ind[k], &model);
                if(params->savefname)
                {
                    if(params->extended_write)
                        fitvals_ext[nevals++] = fitness_ext(&pop.ind[k], &model);
                    else
                        fitvals[nevals++] = pop.ind[k].fitness;
                }
            }
        }
        // choose parents and perform crossover + mutation on children
        for(j=0; j<params->popsize; j = j + 2)
        {
            if(nevals >= params->max_evals)
            {
                if(params->verbose)
                {
                    printf("Final: ");
                    print_stats(&pop, i+1, nevals, ls_nevals, params->do_localsearch, injected);
                    printf("Reached %d evaluations, quitting after %ld seconds\n", nevals, time(NULL) - run_start_time);
                }
                if(params->savefname) 
                {
                    write_meta_header(time(NULL) - run_start_time, pi_triggered, params->savefname);
                    if(params->extended_write)
                        write_fitness_ext(fitvals_ext, nevals, params->savefname);
                    else
                        write_fitness(fitvals, nevals, params->savefname);
                }
                cleanup(&pop, &model, fitvals, &diversity, fitvals_ext);
                return 0;
            }
            int p1;
            int p2;
            do
            {
                p1 = tournament_selection(&pop, params->tournsize);
                p2 = tournament_selection(&pop, params->tournsize);
            }
            while(p1 == p2);
            Individual c1;
            create_null_individual(&c1, model.vcount);
            Individual c2;
            create_null_individual(&c2, model.vcount);
            uniform_crossover(&pop.ind[p1], &pop.ind[p2], &c1, &c2);
            bitflip_mutation(&c1, params->mutpb);
            bitflip_mutation(&c2, params->mutpb);
            c1.fitness = fitness(&c1, &model);
            if(params->savefname)
            {
                if(params->extended_write)
                    fitvals_ext[nevals++] = fitness_ext(&c1, &model);
                else
                    fitvals[nevals++] = c1.fitness;
            }
            pop.ind[pop.size++] = c1;
            // TODO: Model change!
            if((nevals % change_eval) == 0)
            {
                int next_model = (int)(nevals / change_eval);
                current_model_fname = replace_model(params->inputfname, next_model);
                free_model(&model);
                if(read_full_file(&model, current_model_fname) == -1)
                {   
                    printf("Error reading file '%s', aborting\n", current_model_fname);
                    exit(EXIT_FAILURE);
                }   
                if(model.ecount == 0 || model.vcount == 0)
                {   
                    printf("Either no edge or vertex in file '%s', aborting\n", current_model_fname);
                    exit(EXIT_FAILURE);
                }   
                if(params->verbose)
                {   
                    int active = 0;
                    int k;
                    for(k=0; k<model.vcount; k++)
                        active += model.vertices[k].active;
                    printf("Read file '%s' having |V| = %ld (%d active), |E| = %ld @ %d evals\n", current_model_fname, model.vcount, active, model.ecount, nevals);
                }
                int k;
                for(k=0; k<pop.size; k++)
                {
                    if(nevals >= params->max_evals)
                        break;
                    pop.ind[k].fitness = fitness(&pop.ind[k], &model);
                    if(params->savefname)
                    {
                        if(params->extended_write)
                            fitvals_ext[nevals++] = fitness_ext(&pop.ind[k], &model);
                        else
                            fitvals[nevals++] = pop.ind[k].fitness;
                    }
                }
            }
            if(nevals >= params->max_evals)
            {
                if(params->verbose)
                {
                    printf("Final: ");
                    print_stats(&pop, i+1, nevals, ls_nevals, params->do_localsearch, injected);
                    printf("Reached %d evaluations, quitting after %ld seconds\n", nevals, time(NULL) - run_start_time);
                }
                if(params->savefname) 
                {
                    write_meta_header(time(NULL) - run_start_time, pi_triggered, params->savefname);
                    if(params->extended_write)
                        write_fitness_ext(fitvals_ext, nevals, params->savefname);
                    else
                        write_fitness(fitvals, nevals, params->savefname);
                }
                cleanup(&pop, &model, fitvals, &diversity, fitvals_ext);
                return 0;
            }
            c2.fitness = fitness(&c2, &model);
            if(params->savefname)
            {
                if(params->extended_write)
                    fitvals_ext[nevals++] = fitness_ext(&c2, &model);
                else
                    fitvals[nevals++] = c2.fitness;
            }
            pop.ind[pop.size++] = c2;
            // TODO: Model change!
            if((nevals % change_eval) == 0)
            {
                int next_model = (int)(nevals / change_eval);
                current_model_fname = replace_model(params->inputfname, next_model);
                free_model(&model);
                if(read_full_file(&model, current_model_fname) == -1)
                {
                    printf("Error reading file '%s', aborting\n", current_model_fname);
                    exit(EXIT_FAILURE);
                }
                if(model.ecount == 0 || model.vcount == 0)
                {
                    printf("Either no edge or vertex in file '%s', aborting\n", current_model_fname);
                    exit(EXIT_FAILURE);
                }
                if(params->verbose)
                {
                    int active = 0;
                    int k;
                    for(k=0; k<model.vcount; k++)
                        active += model.vertices[k].active;
                    printf("Read file '%s' having |V| = %ld (%d active), |E| = %ld @ %d evals\n", current_model_fname, model.vcount, active, model.ecount, nevals);
                }
                int k;
                for(k=0; k<pop.size; k++)
                {
                    if(nevals >= params->max_evals)
                        break;
                    pop.ind[k].fitness = fitness(&pop.ind[k], &model);
                    if(params->savefname)
                    {
                        if(params->extended_write)
                            fitvals_ext[nevals++] = fitness_ext(&pop.ind[k], &model);
                        else
                            fitvals[nevals++] = pop.ind[k].fitness;
                    }
                }
            }
        }
        // Determine survivor indices using selection operation
        // TODO: How to ensure uniqueness? Floyd Algorithm?
        int surv_idx[params->popsize];
        for(j=0; j<params->popsize; j++)
        {
            int s;
            do
            {
                s = tournament_selection(&pop, params->tournsize);
            }
            while(contains(surv_idx, j, s));
            surv_idx[j] = s;
        }
        // pick survivors from the population and free dying individuals memory
        for(j=pop.size-1; j>=0; j--)
        {
            if(!contains(surv_idx, params->popsize, j))
            {
                free_individual(&pop.ind[j]);
                if(j != pop.size-1)
                    pop.ind[j] = pop.ind[pop.size-1];
                pop.size--;
            }
        }
        // Place for possible population injection
        if(params->do_pi)
        {
            diversity.values[diversity.next++] = population_fitness_stdev(&pop);
            diversity.next = diversity.next % diversity._memsize;
            if(pi_necessary(&diversity, params->pi_threshold))
            {
                pi_triggered += 1;
                int k;
                for(k=0; k<params->pi_size; k++)
                {
                    if(nevals >= params->max_evals)
                        break;
                    Individual ind;
                    create_random_individual(&ind, model.vcount);
                    ind.fitness = fitness(&ind, &model);
                    if(params->savefname)
                    {
                        if(params->extended_write)
                            fitvals_ext[nevals++] = fitness_ext(&ind, &model);
                        else
                            fitvals[nevals++] = ind.fitness;
                    }
                    pop.ind[pop.size++] = ind;
                    injected++;
                    if(nevals >= params->max_evals)
                        break;
                    // model change
                    if((nevals % change_eval) == 0)
                    {
                        int next_model = (int)(nevals / change_eval);
                        current_model_fname = replace_model(params->inputfname, next_model);
                        free_model(&model);
                        if(read_full_file(&model, current_model_fname) == -1)
                        {    
                            printf("Error reading file '%s', aborting\n", current_model_fname);
                            exit(EXIT_FAILURE);
                        }    
                        if(model.ecount == 0 || model.vcount == 0)
                        {    
                            printf("Either no edge or vertex in file '%s', aborting\n", current_model_fname);
                            exit(EXIT_FAILURE);
                        }    
                        if(params->verbose)
                        {    
                            int active = 0; 
                            int k;
                            for(k=0; k<model.vcount; k++) 
                                active += model.vertices[k].active;
                            printf("Read file '%s' having |V| = %ld (%d active), |E| = %ld @ %d evals\n", current_model_fname, model.vcount, active, model.ecount, nevals);
                        }    
                        // TODO: Double declaration of k???
                        int k;
                        for(k=0; k<pop.size; k++) 
                        {    
                            if(nevals >= params->max_evals)
                                break;
                            pop.ind[k].fitness = fitness(&pop.ind[k], &model);
                            if(params->savefname)
                            {
                                if(params->extended_write)
                                    fitvals_ext[nevals++] = fitness_ext(&pop.ind[k], &model);
                                else
                                    fitvals[nevals++] = pop.ind[k].fitness;
                            }
                            else
                                nevals++;
                        }
                    }
                }    
            }
        }
        // print statistics of the generation
        if(params->verbose) 
            print_stats(&pop, i+1, nevals, ls_nevals, params->do_localsearch, injected);
    }
    return 0;
}

int main(int argc, char**argv)
{
    srand(time(NULL));
    struct ea_parameters params;
    params.inputfname = NULL;
    params.savefname = NULL;
    params.genfname = NULL;
    params.extended_write = 0;
    params.popsize = 100;
    params.tournsize = 5;
    params.mutpb = 0.05;
    params.max_evals = 100000;
    params.model_changes = 0;
    params.verbose = 0;
    params.filecheck = 0;
    params.continuous = 0;
    params.do_localsearch = 0;
    params.ls_k = 50;
    params.do_pi = 0;
    params.pi_width = 1;
    params.pi_threshold = 0.0;
    params.pi_size = 100;

    argp_parse(&argp, argc, argv, 0, 0, &params);
    if(params.continuous)
        run_continuous(&params);
    else
        run_default(&params);
    return 0;
}
