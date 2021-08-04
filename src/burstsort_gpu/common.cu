#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "include/common.h"
#include <cuda_profiler_api.h>
/* the following code deals with the user interface */
static int total_searched=0;
static int total_inserted=0;
static int inserted=0;
static int found=0;
char *buffer_dev;
uint32_t *src_device;
uint32_t *dest_device;
char *len_dev;
char *len;
char *word_dev;
static int ii=0;
int bytesy;
/* display an error message and exit the program */
extern "C" {
void fatal(char *str) { puts(str); exit(1); }
}
__global__
void insert_device_1()
{
  printf("cuda kernel\n");
}

__global__
void node_cpy_device(uint32_t *dest,uint32_t *src,uint32_t bytess)
{ 
  uint32_t tid=threadIdx.x;
  //bytess=bytess>>2;
 // printf("src in cuda%d\n",src);
 //printf("IN CUDA KERNEL");
  int i=0;
 // printf("bytes %d\n",bytes);
 // printf("tid %d",tid);
//  printf("%d",src[2]);
 /* if(tid<bytes)
  {
   
   *(dest+tid)=*(src+tid);
   
  printf("IN CUDA KERNEL");
     
   // i++;
  }*/
  
 /*if(tid<bytess)
  {
    *(dest+tid)=*(src+tid);
     printf("in cuda");   

  }*/
  if(tid<bytess)
  {
    *(dest+tid)=*(src+tid);
    
  //  printf("in cuda");

  }
  
  *(dest+tid+1)='\0';
  __syncthreads();
//  *(src+tid+1)='\0';

 //printf("dwdwedwe\n");


    
 //printf("dest in cuda%d\n",dest); 
}
/* copy a block of memory, a word at a time */
void node_cpy(uint32_t *dest, uint32_t *src, uint32_t bytes)
{
 // uint32_t *dest_device;
 // uint32_t *src_device;
  bytes=bytes>>2;
// printf("%d\n",bytes);
  //cudaMalloc((void **) &dest_device,bytes*sizeof(uint32_t));
  //cudaMalloc((void **) &src_device,bytes*sizeof(uint32_t)); 
  bytesy=bytes;
/*  cudaMemcpy(dest_device,dest,bytesy*sizeof(uint32_t), cudaMemcpyHostToDevice);

  cudaMemcpy(src_device,src,bytesy*sizeof(uint32_t), cudaMemcpyHostToDevice);
 
 // node_cpy_device<<<1,16>>>((char*)dest_device,(char*)src_device,bytes);
 // cudaMemcpyAsync(dest,dest_device,100*sizeof(char), cudaMemcpyDeviceToHost);
 // cudaMemcpyAsync(src,src_device,100*sizeof(char), cudaMemcpyDeviceToHost);
  
  node_cpy_device<<<1,bytesy>>>(dest_device, src_device, bytesy);
  cudaMemcpy(dest,dest_device,bytesy*sizeof(uint32_t), cudaMemcpyDeviceToHost);*/
   
 //  cudaMemcpyAsync(src,src_device,(bytesy)*sizeof(uint32_t), cudaMemcpyDeviceToHost);
  while(bytes != 0)
  {
    *dest++=*src++;
    bytes--;

  }
  //*dest='\0'; 
 //cudaMemcpyAsync(dev_string1,tmp,200*sizeof(char), cudaMemcpyHostToDevice);
}

/* string compare routine with string lengths provided */
int32_t sncmp(const char  *s1, const char  *s2, uint64_t s1_len, uint64_t s2_len)
{
  while ( *s1 == *s2 )
  {
    *s1++;
    *s2++;
    --s1_len;
    --s2_len;

    if ( s1_len == 0 && s2_len ==0)  return 0;
    if ( s1_len == 0 ) return -1;
    if ( s2_len == 0 ) return 1;
  }
  return ( *s1 - *s2);
}

/*
 * scan an array of characters and replace '\n' characters
 * with '\0'
 */
__global__ void set_terminator(char *buffer, int length)
{
 /* register int32_t i=0;
  for(; i<length; ++i)  
  {
    if( *(buffer+i) == '\n' )   
    {
      *(buffer+i) = '\0';
    }
  }*/
 
  register int32_t i=0;
 /* cudaMalloc((void**)&buffer_dev,150*sizeof(char));
  cudaMemcpy(buffer_dev,buffer,150*sizeof(char), cudaMemcpyHostToDevice);
  print<<<1,10>>>(length,buffer_dev);*/
   
   int tid=blockIdx.x * blockDim.x+threadIdx.x+100; 
   //step=step+tid;
     // if(tid+e<length){
    
   // for(i=0;i<50;i++){
      if(tid<(length)){
    	for(i=0;i<100;i++){
	if(*( buffer+tid) == '\n' )   
        {
          *(buffer+tid) = '\0'; 
           
      
        }
      
      }
   }
   //   step=step+i;
    // }	
	
//__syncthreads(); 

  /* int tid=(blockIdx.x * blockDim.x+threadIdx.x)*100;
   //step=step+tid;
     // if(tid+e<length){

   // for(i=0;i<50;i++){
      if(tid<(length)){
        for(int i=0;i<100;i++){
        if(*( buffer+tid+i) == '\n' )
        {
          *(buffer+tid+i) = '\0';


        }

      }
   //   step=step+i;
     }*/

   
}
__global__ void counting_len(char *word,char* len)
{
  char *x=word;
  int tid=threadIdx.x;
  int s=0;
  //for(; *x != '\0'; ++x);
 if(tid!=50 || (*(x+tid)=='\0'))
 // if(tid!=70)
  {
   //    	if(*(x+tid)=='\0')
        {
           s=tid;
           len[0]=s+'0';
          // return; 
          }
  }
   
  
  
   
  
	
}

/* string length routine */
int32_t slen(char *word)
{
 char *x=word;
 char *y=word;
 int32_t f;
  for(; *x != '\0'; ++x);
  
 // cudaMalloc((void**)&word_dev,50*sizeof(char));
 /* cudaProfilerStart();
  cudaMemcpyAsync(word_dev,word,50*sizeof(char), cudaMemcpyHostToDevice);
 // cudaMemcpyAsync(len_dev,word,1*sizeof(char), cudaMemcpyHostToDevice);
  counting_len<<<1,50>>>(word_dev,len_dev);
  cudaMemcpyAsync(y,len_dev,1*sizeof(char), cudaMemcpyDeviceToHost);
  cudaProfilerStop();
 // printf("length %d",y[0]-'0');
  f=y[0]-'0';*/
 // printf("%d\n",f);
  return x-word ;
  
  
}
void reset_counters()
{
  total_searched=total_inserted=inserted=found=0;
}

int32_t get_inserted()
{
  return inserted;
}

int32_t get_found()
{
  return found;
}

/* access the data structure to insert the strings found in the 
 * filename that is provided as a parameter. The file supplied must
 * be smaller than 2GB in size, otherwise, the code below has to be
 * modified to support large files, i.e., open64(), lseek64().
 * This should not be required, however, since the caller to this
 * function should be designed to handle multiple files, to allow you 
 * to break a large file into smaller pieces.  
 */
double perform_insertion(char *to_insert)
{ 
   int32_t  input_file=0;
   int32_t  return_value=0;
   uint32_t input_file_size=0;
   uint32_t read_in_so_far=0;
   

   char *buffer=0;
   char *buffer_start=0;
   char *buff_dev=0;
   cudaStream_t stream1;
  cudaStreamCreate(&stream1);
  
 /* cudaStreamCreateWithFlags(&mystream1,cudaStreamNonBlocking);
   cudaStreamCreateWithFlags(&mystream2,cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&mystream3,cudaStreamNonBlocking);*/
   
   timer start, stop;
   double insert_real_time=0.0;
   cudaMalloc((void**)&word_dev,100*sizeof(char));
   cudaMalloc((void**)&len_dev,100*sizeof(int));
   

   /* open the file for reading */
   if( (input_file=(int32_t) open(to_insert, O_RDONLY))<=0) 
     fatal(BAD_INPUT);  
     
   /* get the size of the file in bytes */
   input_file_size=lseek(input_file, 0, SEEK_END);
     
   /* allocate a buffer in memory to store the file */
   if( (buffer = (char *)calloc(1, input_file_size+1 )) == NULL) 
     fatal(MEMORY_EXHAUSTED);
     
   /* keep a pointer to the start of the buffer */
   buffer_start=buffer;
   
   /* read the file into memory and close the file pointer */
   lseek(input_file, 0, SEEK_SET);

   /* attempt to read the entire file into memory */
   while(read_in_so_far < input_file_size)
   {
     return_value=read(input_file, buffer, input_file_size);
     assert(return_value>=0);
     read_in_so_far+=return_value;
   }
   close(input_file);
  // thrust::device_vector<char>buffer_dev;
  /* for(int i=0;i<sizeof(buffer);i++)
   {
	buffer_dev[i]=buffer[i];
   
   //d_array_stringVals = thrust::raw_pointer_cast(buffer_dev.data());  
   /* make sure that all strings are null terminated */
  // cudaProfilerStart();
   cudaMalloc((void**)&buffer_dev,input_file_size*sizeof(char));
//    cudaHostGetDevicePointer(&buffer_dev, buffer, 0 );
  cudaMemcpyAsync(buffer_dev,buffer,input_file_size*sizeof(char), cudaMemcpyHostToDevice,stream1);
  //print<<<1,1>>>(input_file_size,buffer_dev);

   set_terminator<<<612,850,0,stream1>>>(buffer_dev, input_file_size);
  // terminate<<<1,1>>>(10);

  cudaMemcpyAsync(buffer,buffer_dev,input_file_size*sizeof(char), cudaMemcpyDeviceToHost,stream1);
  // cudaProfilerStop();
   /* start the timer for insertion */  
   gettimeofday(&start, NULL);
//   cudaMalloc((void **) &dest_device,100*sizeof(uint32_t));
//   cudaMalloc((void **) &src_device,100*sizeof(uint32_t));

    cudaMemcpyAsync(buffer_dev,buffer,input_file_size*sizeof(char), cudaMemcpyHostToDevice,stream1);
   /* main insertion loop */
   //insert_device<<<1,1>>>();
   //print<<<1,1>>>();
   time_loop_insert: 
   
   insert_device<<<1,100>>>(buffer_dev);
   /* insert the first null-terminated string in the buffer */
  /* if(insert(buffer))
   {
     inserted++;
   }*/
   total_inserted++;

   /* point to the next string in the buffer */
 // for(; *buffer != '\0'; buffer++);
  int32_t r=slen(buffer);
   buffer=buffer+r;
   buffer++;

   /* if the buffer pointer has been incremented to beyond the size of the file,
    * then all strings have been processed, and the insertion is complete. 
    */   
   if(buffer - buffer_start >= input_file_size) goto insertion_complete;
  //  insert_device<<<1,1>>>(buffer_dev);
 
   goto time_loop_insert;

   insertion_complete:

   /* stop the insertion timer */
   gettimeofday(&stop, NULL);

   /* do the math to compute the time required for insertion */   
   insert_real_time = 1000.0 * ( stop.tv_sec - start.tv_sec ) + 0.001  
   * (stop.tv_usec - start.tv_usec );
   insert_real_time = insert_real_time/1000.0;

   /* free the temp buffer used to store the file in memory */
   free(buffer_start);
   
   /* return the elapsed insertion time */
   return insert_real_time;
}

