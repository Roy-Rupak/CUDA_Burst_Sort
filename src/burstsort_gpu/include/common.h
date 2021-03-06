#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <inttypes.h>
#include <assert.h>

#define MEMORY_EXHAUSTED   "Out of memory"
#define BAD_INPUT          "Can not open or read file"
#define TO_MB 1000000
#define CACHE_LINE_SIZE 128

typedef struct timeval timer;

#define false 0
#define true 1
#define MIN_RANGE (char)32
#define MAX_RANGE (char)126
#define TRIE_SIZE 1024

#define SKIPPING /* dont change */
#define MASK     /* best not to turn off mask */


#define _32_BYTES 32
#define _64_BYTES 64

//    bytes=bytes>>2;
//     // cudaMalloc((void **) &dest_device,10*sizeof(uint32_t));
//      // cudaMalloc((void **) &src_device,10*sizeof(uint32_t)); 
//
double perform_insertion(char *to_insert);
double perform_search(char *to_search);
extern "C" {
void fatal(char *str); 
}
extern "C" {
int32_t scmp(const char  *s1, const char  *s2);
int32_t sncmp(const char *s1, const char *s2, uint64_t, uint64_t);
}
int32_t get_inserted();
int32_t get_found();
__global__ void set_terminator(char *buffer, int length);
int slen(char *word);
void node_cpy(uint32_t *dest, uint32_t *src, uint32_t bytes);
int insert(char *word);

