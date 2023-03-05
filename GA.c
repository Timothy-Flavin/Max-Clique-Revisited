#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

//#define DEBUG_UTIL
//#define DEBUG_FITNESS
//#define DEBUG_CROSSOVER
//#define DEBUG_ROULETTE
//#define DEBUG_ONE_GENE_EXCHANGE
#define DEBUG_GA

// Utilities for the GA code
// Genome structs for parents and children
struct genome{
  int glen;
  float fitness;
  int* genes;
  int graph_size;
  char* gene_mask;
};
struct genome_pair{
  struct genome* genome_1;
  struct genome* genome_2;
};

// Project required functions

// Fitness function for a genome
float fitness(struct genome* g, int* adj_mat);

// Two crossovers
struct genome_pair* single_point(struct genome* genome_1, struct genome* genome_2);
struct genome_pair order_1(struct genome* genome_1, struct genome* genome_2);

// Two selection algorithms
int* roulette(struct genome* parents, int n);
int* rank(struct genome* parents, int n);

// Two mutation operations
void one_gene_exchange(struct genome* g); //make a bit string of size g
void max_degree_exchange(struct genome* g); // Same thing but 50% chance 
                                                    // of choosing max available degree

// Genome and Graph utility functions:
struct genome* copy_genome(struct genome* g);
struct genome* random_genome(int gl, int n);
struct genome* random_population(int p_size, int gl, int n);
void free_population(struct genome* pop, int p_size); // To clear any memory leaks
int* generate_graph(int n, int m_clique, float con_prob);
// nxn adj_matrix with rows 0:m-1 fully connected and connection probability con_prob
// for all other rows
int* row_sums(int* adj_mat, int n); // gets degree of nodes

// -------------------- General utility functions --------------------------------

// Float comparison function for qsort later
int compare_floats(const void * a, const void * b){
  float fa = *(const float*) a;
  float fb = *(const float*) b;
  return (fa>fb) - (fb>fa);
}
// Genome comparison function for qsort later
int compare_genomes(const void * a, const void * b){
  float fa = (*(struct genome *)a).fitness;
  float fb = (*(struct genome *)b).fitness;
  return (fa>fb) - (fb>fa);
}
// Test if number is already in a genome. 1 if true, 0 if false
int int_in(int g, int* genes, int s){
  int in=0;
  for(int i=0; i<s; ++i){
    if(genes[i] == g) in=1;
  }
  return in;
}

// Used to visualize genomes
void print_genome(struct genome* g){
  printf("Printing Genome at address: %p\n  Length: %d, Fitness: %f\n  Chromosome: [", g, g->glen, g->fitness);
  for(int i=0; i<g->glen; ++i){
    printf("%d, ",g->genes[i]);
  }
  printf("]\n  Gene Mask: [");
  for(int i=0; i<g->graph_size; ++i){
    printf("%d, ",g->gene_mask[i]);
  }
  printf("]\n  Size of genome: %lu\n", sizeof(*g));
}
void print_adj_matrix(int* adj_mat, int n){
  printf("Printing %d x %d adjacency matrix: \n", n, n);
  for(int i=0; i<n; ++i){
    printf("[");
    for(int j=0; j<n; ++j){
      printf("%d, ",adj_mat[i*n + j]);
    }
    printf("]\n");
  }
}

// Rand number shortcuts
float randf(){
  return (float)rand() / ((float)RAND_MAX + 1);
}
float randi(start, end){
  int range=end-start;
  return rand()%range + start;
}


// Unit test for utility functions
void util_unit_tests(){
  //printf("size of int %lu, size of float %lu, size of int* %lu\n", sizeof(int), sizeof(float), sizeof(int*));

  printf("\n----------------Running Util unit tests... -------------\n\n");

  float* compare_test = malloc(100*sizeof(float));
  printf("Before Sorting: \n");
  for(int i=0; i<10; ++i){
    compare_test[i] = (float)rand() / ((float)RAND_MAX + 1);
    printf("%f, ", compare_test[i]);
  }
  printf("\nAfter Sorting: \n");
  qsort(compare_test, 10, sizeof(float), compare_floats);
  for(int i=0; i<10; ++i){
    printf("%f, ", compare_test[i]);
  }
  printf("\n");

  printf("Random Genome: \n");
  int glen=10;
  int pop_size = 5;
  int graph_size = 15;
  struct genome* g1 = random_genome(glen, graph_size);//{glen, 0.5f, (int*)calloc(glen,sizeof(int)), graph_size, (char*)calloc(graph_size,sizeof(char))};
  print_genome(g1);
  printf("Copy of Random Genome: \n");
  struct genome* g2 = copy_genome(g1);
  print_genome(g2);
  printf("One Gene Exchange: \n");
  one_gene_exchange(g2);
  print_genome(g2);


  printf("\nBefore sorting genomes: \n");
  struct genome* glist = random_population(pop_size, glen, graph_size);
  for(int i=0; i<pop_size; ++i){
    glist[i].fitness = randf();
    printf("Pop number: %d ", i);
    print_genome(&glist[i]);
  }

  qsort(glist, pop_size, sizeof(struct genome), compare_genomes);


  printf("\nAfter sorting genomes: \n");
  for(int i=0; i<pop_size; ++i){
    printf("Pop number: %d ", i);
    print_genome(&glist[i]);
  }

  printf("\nTesting int_in: \n");
  int tryin[10] = {0,1,2,3,4,5,6,7,8,9};
  printf("7: %d, 3: %d, 10: %d, -1: %d\n\n", int_in(7, tryin, 10), int_in(3, tryin, 10), int_in(10, tryin, 10), int_in(-1, tryin, 10));

  printf("Generating 5x5 graph with 3x3 cliqe and 0.4 connection prob\n");
  int* adj = generate_graph(5, 3, 0.4f);
  print_adj_matrix(adj, 5);
  printf("Row sums: ");
  int* rums = row_sums(adj, 5);
  for(int i=0; i<5; ++i){
    printf("%d, ", rums[i]);
  }
  printf("\n\n");

}

void fitness_unit_tests(){
  printf("\n----------------Running Fitness unit tests... -------------\n\n");
  printf("Testing Fitness function on three genomes with adj_mat:\n");
  int n=5;
  int adj_mat[25] = {1,1,1,1,1, 1,1,1,1,0, 1,1,1,1,0, 1,1,1,1,0, 1,0,0,0,0};
  print_adj_matrix(&adj_mat[0], 5);
  int genes1[3] = {0,1,2};
  int genes2[4] = {0,1,2,3};
  int genes3[5] = {0,1,2,3,4};
  char gmask1[5] = {1,1,1,0,0};
  char gmask2[5] = {1,1,1,1,0};
  char gmask3[5] = {1,1,1,1,1};
  struct genome g1 = {3, 0.0f, &genes1[0], n, &gmask1[0]};
  struct genome g2 = {4, 0.0f, &genes2[0], n, &gmask2[0]};
  struct genome g3 = {5, 0.0f, &genes3[0], n, &gmask3[0]};
  g1.fitness = fitness(&g1, &adj_mat[0]);
  g2.fitness = fitness(&g2, &adj_mat[0]);
  g3.fitness = fitness(&g3, &adj_mat[0]);

  print_genome(&g1);
  print_genome(&g2);
  print_genome(&g3);
}

void crossover_unit_test(){
  
  printf("\n----------------Running Crossover unit tests... -------------\n\n");
  int genes1[5] = {1,2,5,6,7};
  int genes2[5] = {0,1,2,3,4};
  char gmask1[8] = {0,1,1,0,0,1,1,1};
  char gmask2[8] = {1,1,1,1,1,0,0,0};
  struct genome g1 = {5, 0.0f, &genes1[0], 8, &gmask1[0]};
  struct genome g2 = {5, 0.0f, &genes2[0], 8, &gmask2[0]};
  struct genome_pair parents = {&g1, &g2};

  printf("\nGenomes before crossover: \n");
  print_genome(&g1);
  print_genome(&g2);

  struct genome_pair* children = single_point(parents.genome_1, parents.genome_2);

  printf("\nGenomes after crossover: \n");
  print_genome(children->genome_1);
  print_genome(children->genome_2);
}

void roulette_unit_test(){
  printf("\n----------------Running Roulette unit tests... -------------\n\n");
  int n=5;
  int adj_mat[25] = {1,1,0,1,1, 1,1,1,1,0, 0,1,1,1,0, 1,1,1,1,0, 1,0,0,0,0};
  printf("Testing Roulette Wheel, Parents: \n");
  print_adj_matrix(adj_mat, n);

  int npar = 8;
  struct genome* parents = random_population(npar, 3, n);
  for(int i=0; i<npar; ++i){  
    parents[i].fitness = fitness(&parents[i], adj_mat);
    print_genome(&parents[i]);
  }
  printf("Selection: \n");
  int* selection = roulette(parents, npar);
  for(int i=0; i<npar; ++i){  
    printf("%d, ", selection[i]);
  }
  printf("\n");
}


// Project Function implementations: 
// Genome and Graph utility functions:
//struct genome* copy_genome(struct genome* g){}

// gl is the genome length and n is the graph size
struct genome* random_genome(int gl, int n){
  int* gen = malloc(gl*sizeof(int));
  char* gen_mask = malloc(n*sizeof(char));
  for(int i=0; i<n; ++i){
    gen_mask[i] = 0;
  }
  for(int i=0; i<gl; ++i){
    int in=1;
    int g = 0;
    while(in){
      g = randi(0, n);
      in = gen_mask[g];//int_in(g, gen, i+1); //dont need to check all elements of course
    }  
    gen[i] = g;
    gen_mask[g] = 1;
  }

  struct genome* new_gen = malloc(sizeof(struct genome));
  new_gen->fitness=0.0f;
  new_gen->genes = gen;
  new_gen->glen = gl;
  new_gen->graph_size = n;
  new_gen->gene_mask = gen_mask;

  return new_gen;
}

struct genome* random_population(int p_size, int gl, int n){
  struct genome* parents = malloc(p_size*sizeof(struct genome));
  for(int i=0; i<p_size; ++i){
    parents[i] = *random_genome(gl, n);
  }
  return parents;
}

void free_population(struct genome* pop, int p_size){
  for(int i=0; i<p_size; ++i){
    free(pop[i].genes);
    free(pop[i].gene_mask);
  }
  free(pop);
}


// Fitness function for a genome
float fitness(struct genome* g, int* adj_mat){
  int tot_p_con = g->glen * (g->glen-1);
  int tot_con = 0;
  int n = g->graph_size;
  //for each gene
  for(int i = 0; i<g->glen; ++i){
    int gene = g->genes[i];
    #ifdef DEBUG_FITNESS
    printf("\n Gene %d: ", gene);
    #endif
    for(int c=0; c<n; ++c){
      if(c==gene) continue;
      if(adj_mat[gene*n + c] && g->gene_mask[c]){
        ++tot_con;
        #ifdef DEBUG_FITNESS
        printf("c: %d,", c);
        #endif
      }
    }
  }
  #ifdef DEBUG_FITNESS
  printf("\nTotal connections possible: %d, connections present: %d\n", tot_p_con, tot_con);
  #endif
  return (float)tot_con / tot_p_con;
}

// Two crossovers
struct genome_pair* single_point(struct genome* genome_1, struct genome* genome_2){
  int gl = genome_1->glen;
  int n = genome_1->graph_size;
  int cross_point = randi(1,gl);
  #ifdef DEBUG_CROSSOVER
  printf("Cross Point: %d", cross_point);
  #endif



  int* child1 = malloc(gl*sizeof(int));
  int* child2 = malloc(gl*sizeof(int));
  char* c1_mask = calloc(n,sizeof(char));
  char* c2_mask = calloc(n,sizeof(char));


  struct genome* ch1 = malloc(sizeof(struct genome));
  struct genome* ch2 = malloc(sizeof(struct genome));

  ch1->glen = gl;
  ch1->genes = child1;
  ch1->fitness = 0.0f;
  ch1->graph_size = n;
  ch1->gene_mask = c1_mask;

  ch2->glen = gl;
  ch2->genes = child2;
  ch2->fitness = 0.0f;
  ch2->graph_size = n;
  ch2->gene_mask = c2_mask;
  
  int fixup1=0, fixup2=0;

  for(int i=0; i<cross_point; ++i){
    child1[i] = genome_1->genes[i];
    c1_mask[genome_1->genes[i]] = 1;

    child2[i] = genome_2->genes[i];
    c2_mask[genome_2->genes[i]] = 1;
  }
  for(int i=cross_point; i<gl; ++i){
    int c_conflict = c1_mask[genome_2->genes[i]]; // dont need to check all spots because most are -1
    if(c_conflict){
      while(c_conflict){
        c_conflict = c1_mask[genome_2->genes[fixup1]];
        ++fixup1;
      }
      child1[i] = genome_2->genes[fixup1-1];
      c1_mask[genome_2->genes[fixup1-1]] = 1;
    }
    else{
      child1[i] = genome_2->genes[i];
      c1_mask[genome_2->genes[i]] = 1;
    }

    c_conflict = c2_mask[genome_1->genes[i]];
    if(c_conflict){
      while(c_conflict){
        c_conflict = c2_mask[genome_1->genes[fixup2]];
        ++fixup2;
      }
      child2[i] = genome_1->genes[fixup2-1];
      c2_mask[genome_1->genes[fixup2-1]] = 1;
    }
    else{
      child2[i] = genome_1->genes[i];
      c2_mask[genome_1->genes[i]] = 1;
    }

  }

  #ifdef DEBUG_CROSSOVER
    printf("\n");
    for(int i=0; i<n; ++i){
      printf("%d", c1_mask[i]);
    }
    printf("\n");
    for(int i=0; i<n; ++i){
      printf("%d", c2_mask[i]);
    }
    printf("\n");
  #endif
  struct genome_pair* children = malloc(sizeof(struct genome_pair));
  children->genome_1 = ch1;
  children->genome_2 = ch2;
  return children;
}
//struct genome_pair uniform(struct genome* genome_1, struct genome* genome_2);

// Two selection algorithms
int* roulette(struct genome* parents, int n){
  int* children = malloc(n*sizeof(int));
  
  float tot_fit = 0.0f;
  for(int i=0; i<n; ++i){
    tot_fit += parents[i].fitness;
  }
  #ifdef DEBUG_ROULETTE
    printf("Parents, tot_fit: %f", tot_fit);
  #endif
  for(int i=0; i<n; ++i){
    float sel_fit = randf() * tot_fit;
    #ifdef DEBUG_ROULETTE
      float og_fit = sel_fit;
    #endif
    int j=0;
    while(sel_fit > parents[j].fitness){
      sel_fit -= parents[j].fitness;
      ++j;
    }
    children[i] = j;
    #ifdef DEBUG_ROULETTE
    printf("[%d, %f], ",j, og_fit);
    #endif
  }
  #ifdef DEBUG_ROULETTE
    printf("\n");
  #endif
  return children;
}
//int* rank(int* fitness);

// Two mutation operations
void one_gene_exchange(struct genome* g){
  int off = randi(0,g->glen);
  int on = randi(0,g->graph_size - g->glen);

  #ifdef DEBUG_ONE_GENE_EXCHANGE
  printf("One Gene Exchange off: %d, on: %d\n", off, on);
  #endif

  int new_gene = -1;
  int i=0;
  while(new_gene == -1){
    #ifdef DEBUG_ONE_GENE_EXCHANGE
    printf("i: %d, new_gene: %d, on: %d\n", i, new_gene, on);
    #endif
    if(!g->gene_mask[i]){
      if(!on){
        new_gene = i;
        g->gene_mask[i] = 1;
        g->gene_mask[g->genes[off]] = 0;
      }
      else{
        --on;
        ++i;
      }
    }
    else{
      ++i;
    }
  }

  g->genes[off] = new_gene;
} 
//struct genome max_degree_exchange(struct genome* g); // Same thing but 50% chance 
                                                    // of choosing max available degree

// Genome and Graph utility functions:
struct genome* copy_genome(struct genome* g){
  int* gen = malloc(g->glen*sizeof(int));
  char* gen_mask = malloc(g->graph_size*sizeof(char));
  for(int i=0; i<g->graph_size; ++i){
    gen_mask[i] = g->gene_mask[i];
  }
  for(int i=0; i<g->glen; ++i){
    gen[i] = g->genes[i];
  }

  struct genome* new_gen = malloc(sizeof(struct genome));
  new_gen->fitness=g->fitness;
  new_gen->genes = gen;
  new_gen->glen = g->glen;
  new_gen->graph_size = g->graph_size;
  new_gen->gene_mask = gen_mask;
  return new_gen;
}


int* generate_graph(int n, int m_clique, float con_prob){
  int* adj_mat = calloc(n*n, sizeof(int));
  
  for(int i=0; i<n; ++i){
    for(int j=0; j<i; ++j){
      if(i<m_clique){
        adj_mat[i*n+j] = 1;
        adj_mat[j*n+i] = 1;
      }
      else if(randf() < con_prob){
        adj_mat[i*n+j] = 1;
        adj_mat[j*n+i] = 1;
      }
    }
  }
  return adj_mat;
}
// nxn adj_matrix with rows 0:m-1 fully connected and connection probability con_prob
// for all other rows
int* row_sums(int* adj_mat, int n){
  int* rsums = calloc(n,sizeof(int));
  for(int i=0; i<n; ++i){
    for(int j=0; j<n; ++j){
      rsums[i] += adj_mat[i*n+j];
    }
  }
  return rsums;
}




int main(int argc, char** argv){
  float mutation_rate = ((float) atoi(argv[1])) / 100;
  int popsize = atoi(argv[2]);
  int clique_size = atoi(argv[3]);
  int graph_size = atoi(argv[4]);
  float con_prob = ((float) atoi(argv[5])) / 100;

  int min_clique_size = 2;
  int max_clique_size = graph_size-1;
  int cur_clique_size = 5;//min_clique_size;

  printf("GA Initialized with \nMutation Rate: %f\nPop Size: %d\nClique Size: %d\n", mutation_rate, popsize, clique_size);

  #ifdef DEBUG_UTIL
  util_unit_tests();
  #endif
  #ifdef DEBUG_FITNESS
  fitness_unit_tests();
  #endif
  #ifdef DEBUG_CROSSOVER
  crossover_unit_test();
  #endif
  #ifdef DEBUG_ROULETTE
  roulette_unit_test();
  #endif
  int* adj = generate_graph(graph_size, clique_size, con_prob);

  #ifdef DEBUG_GA
    print_adj_matrix(adj, graph_size);
  #endif

  struct genome* population = random_population(popsize, cur_clique_size, graph_size);
  int generation = 0;
  float max_fit = 0.0f;

  while(max_fit < 1.0f){
    float tot_fit = 0.0f;
    max_fit = 0.0f;
    int fittest_individual = -1;
    for(int p=0; p<popsize; ++p){
      population[p].fitness = fitness(&population[p], adj);
      tot_fit += population[p].fitness;
      if(population[p].fitness > max_fit){
        max_fit = population[p].fitness;
        fittest_individual = p;
      }
    }
    #ifdef DEBUG_GA
      printf("Average fitness of population %d: %f, max_fit: %f, fittest_individual: %d\n", generation, tot_fit / popsize, max_fit, fittest_individual);
      print_genome(&population[fittest_individual]);
    #endif
    if(max_fit >=1.0f){
      break;
    }
    int* selected = roulette(population, popsize);
    struct genome* child_pool = malloc(popsize * sizeof(struct genome));
    for(int i=0; i<popsize; i+=2){
      struct genome_pair* children = single_point(&population[selected[i]], &population[selected[i+1]]);
      child_pool[i] = *(children->genome_1);
      child_pool[i+1] = *(children->genome_2);
      free(children);
    }

    for(int i=0; i<popsize; ++i){
      if(randf() < mutation_rate){
        one_gene_exchange(&child_pool[i]);
      }
    }

    free_population(population, popsize);
    population = child_pool;
    ++generation; 
  }
  
  free_population(population, popsize);
  free(adj);

  return 0;
}