#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

//#define DEBUG_UTIL
//#define DEBUG_FITNESS
//#define DEBUG_CROSSOVER
#define DEBUG_ROULETTE

// Utilities for the GA code
// Genome structs for parents and children
struct genome{
  int glen;
  float fitness;
  int* genes;
};
struct genome_pair{
  struct genome* genome_1;
  struct genome* genome_2;
};

// Project required functions

// Fitness function for a genome
float fitness(struct genome* g, int* adj_mat, int n);

// Two crossovers
struct genome_pair* single_point(struct genome* genome_1, struct genome* genome_2);
struct genome_pair order_1(struct genome* genome_1, struct genome* genome_2);

// Two selection algorithms
int* roulette(struct genome* parents, int n);
int* rank(struct genome* parents, int n);

// Two mutation operations
struct genome one_gene_exchange(struct genome* g); //make a bit string of size g
struct genome max_degree_exchange(struct genome* g); // Same thing but 50% chance 
                                                    // of choosing max available degree

// Genome and Graph utility functions:
struct genome* copy_genome(struct genome* g);
struct genome* random_genome(int n, int gl);
int* generate_graph(int n, int m_clique, float con_prob);
// nxn adj_matrix with rows 0:m-1 fully connected and connection probability con_prob
// for all other rows
int* row_sums(int* adj_mat, int n);

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
  printf("Printing Genome:\n  Length: %d, Fitness: %f\n  Chromosome: ", g->glen, g->fitness);
  for(int i=0; i<g->glen; ++i){
    printf("%d, ",g->genes[i]);
  }
  printf("\n  Size of genome: %lu\n", sizeof(*g));
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
  printf("size of int %lu, size of float %lu, size of int* %lu\n", sizeof(int), sizeof(float), sizeof(int*));

  printf("Running unit tests... \n");

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

  int glen=10;
  int pop_size = 5;
  struct genome g1 = {glen, 0.5f, (int*)malloc(glen*sizeof(int))};
  print_genome(&g1);

  printf("\nBefore sorting genomes: \n");
  struct genome* glist = (struct genome*)malloc(pop_size*sizeof(struct genome));
  for(int i=0; i<pop_size; ++i){
    glist[i].glen = glen;
    glist[i].fitness = randf();
    glist[i].genes = (int*)malloc(glen*sizeof(int));
    for(int j=0; j<glen; ++j){
      glist[i].genes[j] = randi(0,10);
    }
    printf("Gene number: %d ", i);
    print_genome(&glist[i]);
  }

  qsort(glist, pop_size, sizeof(struct genome), compare_genomes);


  printf("\nAfter sorting genomes: \n");
  for(int i=0; i<pop_size; ++i){
    printf("Gene number: %d ", i);
    print_genome(&glist[i]);
  }

  printf("\nTesting int_in: \n");
  int tryin[10] = {0,1,2,3,4,5,6,7,8,9};
  printf("7: %d, 3: %d, 10: %d, -1: %d\n\n", int_in(7, tryin, 10), int_in(3, tryin, 10), int_in(10, tryin, 10), int_in(-1, tryin, 10));

}

void fitness_unit_tests(){
  printf("Testing Fitness function on three genomes with adj_mat:\n");
  int n=5;
  int adj_mat[25] = {1,1,1,1,1, 1,1,1,1,0, 1,1,1,1,0, 1,1,1,1,0, 1,0,0,0,0};
  print_adj_matrix(&adj_mat[0], 5);
  int genes1[3] = {0,1,2};
  int genes2[4] = {0,1,2,3};
  int genes3[5] = {0,1,2,3,4};
  struct genome g1 = {3, 0.0f, &genes1[0]};
  struct genome g2 = {4, 0.0f, &genes2[0]};
  struct genome g3 = {5, 0.0f, &genes3[0]};
  g1.fitness = fitness(&g1, &adj_mat[0], n);
  g2.fitness = fitness(&g2, &adj_mat[0], n);
  g3.fitness = fitness(&g3, &adj_mat[0], n);

  print_genome(&g1);
  print_genome(&g2);
  print_genome(&g3);
}

void crossover_unit_test(){
  int genes1[5] = {1,2,5,6,7};
  int genes2[5] = {0,1,2,3,4};
  struct genome g1 = {5, 0.0f, &genes1[0]};
  struct genome g2 = {5, 0.0f, &genes2[0]};
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
  int n=5;
  int adj_mat[25] = {1,1,0,1,1, 1,1,1,1,0, 0,1,1,1,0, 1,1,1,1,0, 1,0,0,0,0};
  printf("Testing Roulette Wheel");
  print_adj_matrix(adj_mat, n);

  int npar = 8;
  struct genome* parents = malloc(npar*sizeof(struct genome));
  for(int i=0; i<npar; ++i){
    parents[i] = *random_genome(3, 5);
    parents[i].fitness = fitness(&parents[i], adj_mat, n);
    print_genome(&parents[i]);
  }

  int* selection = roulette(parents, npar);

}


// Project Function implementations: 
// Genome and Graph utility functions:
//struct genome* copy_genome(struct genome* g){}

// n is the number of genes, gl is the graph length
struct genome* random_genome(int n, int gl){
  int* gen = malloc(n*sizeof(int));
  for(int i=0; i<n; ++i){
    gen[i] = -1;
  }
  for(int i=0; i<n; ++i){
    int in=1;
    int g = 0;
    while(in){
      g = randi(0, gl);
      in = int_in(g, gen, i+1); //dont need to check all elements of course
    }  
    gen[i] = g;
  }

  struct genome* new_gen = malloc(sizeof(struct genome));
  new_gen->fitness=0.0f;
  new_gen->genes = gen;
  new_gen->glen = n;

  return new_gen;
}
//int* generate_graph(int n, int m_clique, float con_prob){}
// nxn adj_matrix with rows 0:m-1 fully connected and connection probability con_prob
// for all other rows
//int* row_sums(int* adj_mat, int n){}


// Fitness function for a genome
float fitness(struct genome* g, int* adj_mat, int n){
  int tot_p_con = g->glen * (g->glen-1);
  int tot_con = 0;
  //for each gene
  for(int i = 0; i<g->glen; ++i){
    int gene = g->genes[i];
    #ifdef DEBUG_FITNESS
    printf("\n Gene %d: ", gene);
    #endif
    for(int c=0; c<n; ++c){
      if(c==gene) continue;
      if(adj_mat[gene*n + c] && int_in(c, g->genes, g->glen)){
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
  int cross_point = randi(1,gl);
  #ifdef DEBUG_CROSSOVER
  printf("Cross Point: %d", cross_point);
  #endif



  int* child1 = malloc(gl*sizeof(int));
  int* child2 = malloc(gl*sizeof(int));
  struct genome* ch1 = malloc(sizeof(struct genome));
  struct genome* ch2 = malloc(sizeof(struct genome));

  ch1->glen = gl;
  ch1->genes = child1;
  ch1->fitness = 0.0f;
  ch2->glen = gl;
  ch2->genes = child2;
  ch2->fitness = 0.0f;

  for(int i=0; i<gl; ++i){
    child1[i] = -1;
    child2[i] = -1;    
  }
  
  int fixup1=0, fixup2=0;

  for(int i=0; i<cross_point; ++i){
    child1[i] = genome_1->genes[i];
    child2[i] = genome_2->genes[i];
  }
  for(int i=cross_point; i<gl; ++i){
    int c_conflict = int_in(genome_2->genes[i], child1, i); // dont need to check all spots because most are -1
    if(c_conflict){
      while(c_conflict){
        c_conflict = int_in(genome_2->genes[fixup1], child1, i);
        ++fixup1;
      }
      child1[i] = genome_2->genes[fixup1-1];
    }
    else{
      child1[i] = genome_2->genes[i];
    }

    c_conflict = int_in(genome_1->genes[i], child2, i);
    if(c_conflict){
      while(c_conflict){
        c_conflict = int_in(genome_1->genes[fixup2], child2, i);
        ++fixup2;
      }
      child2[i] = genome_1->genes[fixup2-1];
    }
    else{
      child2[i] = genome_1->genes[i];
    }

  }
  struct genome_pair* children = malloc(sizeof(struct genome_pair));
  children->genome_1 = ch1;
  children->genome_2 = ch2;
  return children;
}
//struct genome_pair order_1(struct genome* genome_1, struct genome* genome_2);

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
//struct genome one_gene_exchange(struct genome* g); //make a bit string of size g
//struct genome max_degree_exchange(struct genome* g); // Same thing but 50% chance 
                                                    // of choosing max available degree

// Genome and Graph utility functions:
//struct genome* copy_genome(struct genome* g);
//int* generate_graph(int n, int m_clique, float con_prob);
// nxn adj_matrix with rows 0:m-1 fully connected and connection probability con_prob
// for all other rows
//int* row_sums(int* adj_mat, int n);




int main(int argc, char** argv){
  float mutation_rate = ((float) atoi(argv[1])) / 100;
  int popsize = atoi(argv[2]);
  int clique_size = atoi(argv[3]);
  int ut = atoi(argv[4]);

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


  return 0;
}