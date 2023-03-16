#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>



//#define DEBUG_UTIL
//#define DEBUG_FITNESS
//#define DEBUG_CROSSOVER
//#define DEBUG_ROULETTE
//#define DEBUG_ONE_GENE_EXCHANGE
//#define DEBUG_GA

// Utilities for the GA code
// Genome structs for parents and children
struct genome{
  int glen;
  float fitness;
  int* genes;
  int graph_size;
  char* gene_mask;
};

int* degree = NULL;

// Project required functions
// Fitness function for a genome
float fitness(struct genome* g, int* adj_mat);
int run_SA(struct genome* S, int g_len, int T0, int num_iter, float Alpha, float Beta, int* graph, int graph_size, int max_time, void (*mutation) (struct genome*));

// Two mutation operations
void one_gene_exchange(struct genome* g); //make a bit string of size g
void max_degree_exchange(struct genome* g); // Same thing but 50% chance 
                                                    // of choosing max available degree

// Genome and Graph utility functions:
void copy_genome(struct genome* g, struct genome* new_g);
struct genome* random_genome(int gl, int n);
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
  return ((float)rand() + 1) / ((float)RAND_MAX + 2);
}
int randi(int start, int end){
  int range=end-start;
  return rand()%range + start;
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

void random_genome_inplace(int gl, int n, struct genome* new_gen){
  new_gen->genes = malloc(gl*sizeof(int));
  new_gen->gene_mask = malloc(n*sizeof(char));
  for(int i=0; i<n; ++i){
    new_gen->gene_mask[i] = 0;
  }
  for(int i=0; i<gl; ++i){
    int in=1;
    int g = 0;
    while(in){
      g = randi(0, n);
      in = new_gen->gene_mask[g];//int_in(g, gen, i+1); //dont need to check all elements of course
    }  
    new_gen->genes[i] = g;
    new_gen->gene_mask[g] = 1;
  }

  new_gen->fitness=0.0f;
  new_gen->glen = gl;
  new_gen->graph_size = n;
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
void max_degree_exchange(struct genome* g){
  if(randf() > 0.5f) {one_gene_exchange(g); return;}

  int max_deg = -1;
  int gene_i = 0;
  for(int i=0; i< g->graph_size; ++i){
    if(g->gene_mask[i]) continue;

    if(degree[i] > max_deg){
      max_deg = degree[i];
      gene_i = i;
    }
  }

  int swap = randi(0,g->glen);
  g->gene_mask[g->genes[swap]] = 0;
  g->genes[swap] = gene_i;
  g->gene_mask[gene_i] = 1;

} 

// Genome and Graph utility functions:
void copy_genome(struct genome* g, struct genome* new_g){
  new_g->fitness = g->fitness;
  new_g->glen = g->glen;
  new_g->graph_size = g->graph_size;
  for(int i=0; i<g->graph_size; ++i){
    new_g->gene_mask[i] = g->gene_mask[i];
  }
  for(int i=0; i<g->glen; ++i){
    new_g->genes[i] = g->genes[i];
  }
}
void free_genome(struct genome* g){
  free(g->genes);
  free(g->gene_mask);
  free(g);
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


int run_SA(struct genome* S, int g_len, int T0, int num_iter, float Alpha, float Beta, int* graph, int graph_size, int max_time, void (*mutation) (struct genome*)){
  struct genome* new_S = random_genome(g_len, graph_size);
  copy_genome(S, new_S);
  float T = T0;
  int iterations = num_iter;

  time_t start = time(NULL);
  while( time(NULL) - start < max_time && iterations > 1){
    int i=0;
    while(i < iterations){
      copy_genome(S, new_S);
      mutation(new_S);
      float f1 = fitness(S, graph), f2 = fitness(new_S, graph);
      
      
      // If we found the solution then just stop there is no point to go on
      if(f1 >= 1.0f){
        return 1;
      }
      if(f2 >= 1.0f){
        copy_genome(new_S, S);
        return 1;
      }
      
      double f3 = (double)(f2-f1)/T;
      //Otherwise we do the algorithm. 
      if(f1 < f2){// || randf() < exp(f3)){
        copy_genome(new_S, S);
      }
      ++i;
    }
    T = Alpha*T;
    iterations = (int) (Beta*iterations);
  }

  return 0;
}

int main(int argc, char** argv){
  srand(time(NULL)); //time(NULL)
  float Alpha = ((float) atoi(argv[1])) / 100;
  float Beta = ((float) atoi(argv[2])) / 100;
  int clique_size = atoi(argv[3]);
  int graph_size = atoi(argv[4]);
  float con_prob = ((float) atoi(argv[5])) / 100;

  int num_iter = atoi(argv[6]);
  float T0 = ((float) atoi(argv[7])) / 100;
  int mut_no = atoi(argv[8]);

  int max_time = atoi(argv[9]);
  int max_in_time = atoi(argv[10]);

  // Choosing operators
  void (*mutation) (struct genome*) = one_gene_exchange;
  if(mut_no == 1){ mutation = max_degree_exchange; }

  int min_clique_size = 2;
  int max_clique_size = graph_size-1;
  int cur_clique_size = min_clique_size;

  printf("SA Initialized with \nAlpha: %f\nBeta: %f\nClique Size: %d\nGraph Size: %d\nCon_Prob: %f\nN-iter: %d\nT0: %f", Alpha, Beta, clique_size, graph_size, con_prob, num_iter, T0);

  int* adj = generate_graph(graph_size, clique_size, con_prob);
  degree = row_sums(adj, graph_size);

  time_t start = time(NULL);
  time_t best = time(NULL);
  struct genome* best_genome = random_genome(cur_clique_size, graph_size);
  struct genome* temp_best = random_genome(cur_clique_size, graph_size);

  while(clique_size < graph_size && time(NULL) - start < max_time){
    free(temp_best->genes);
    free(temp_best->gene_mask);
    free(temp_best);
    temp_best = random_genome(cur_clique_size, graph_size);
    
    int res = run_SA(temp_best, cur_clique_size, T0, num_iter, Alpha, Beta, adj, graph_size, max_in_time, mutation);
    if(res > 0){
      printf("Found solution for clique size %d\n\n", cur_clique_size);
      free(best_genome->genes);
      free(best_genome->gene_mask);
      free(best_genome);
      best_genome = random_genome(cur_clique_size, graph_size);
      copy_genome(temp_best, best_genome);
      cur_clique_size++;
      best = time(NULL);
    }
    else{
      printf("Did not find solution for size %d\n", cur_clique_size);
    }
  }


  printf("Done getting genes...");

  char filename[11] = {'S','A','f','0','.','t','x','t','\0'};
  filename[3] = mut_no+ '0';

  printf("Set File Name...\n");
  printf("%s",filename);

  FILE * f = fopen(filename, "a");
  for(int i=0; i<graph_size; ++i)
    fputc(best_genome->gene_mask[i] + '0', f);

  printf("\nPrinted Best Genome...\n");

  char str[256];
  sprintf(str, "%lu", best - start);

  fputs(" Time_Taken ",f);
  fputs(str,f);

  sprintf(str, "%d", graph_size);
  fputs(" Graph_Size ",f);
  fputs(str,f);

  sprintf(str, "%f", Alpha);
  fputs(" Alpha ",f);
  fputs(str,f);

  sprintf(str, "%f", Beta);
  fputs(" Beta ",f);
  fputs(str,f);

  sprintf(str, "%d", clique_size);
  fputs(" Max_Clique_Present ",f);
  fputs(str,f);

  sprintf(str, "%d", cur_clique_size-1);
  fputs(" Max_Clique_Found ",f);
  fputs(str,f);
  fputc('\n',f);
  
  fclose(f);


  free_genome(best_genome);
  free_genome(temp_best);
  printf("Best clique found in: %lu s total", best-start);
  free(adj);

  return 0;
}