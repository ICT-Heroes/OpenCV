/***********************************************************************/
/*                                                                     */
/*   svm_learn_main.c                                                  */
/*                                                                     */
/*   Command line interface to the learning module of the              */
/*   Support Vector Machine.                                           */
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 02.07.02                                                    */
/*                                                                     */
/*   Copyright (c) 2000  Thorsten Joachims - All rights reserved       */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/


/* if svm-learn is used out of C++, define it as extern "C" */
#ifdef __cplusplus
extern "C" {
#endif

# include "svm_common.h"
# include "svm_learn.h"

#ifdef __cplusplus
}
#endif

char docfile[200];           /* file with training examples */
char modelfile[200];         /* file for resulting classifier */
char restartfile[200];       /* file with initial alphas */

void   read_input_parameters(int, char **, char *, char *, char *, long *, 
			     LEARN_PARM *, KERNEL_PARM *);
void   wait_any_key();
void   print_help();



int Learn_Main (int argc, char* argv[])
{  
  DOC **docs;  /* training examples */
  long totwords,totdoc,i;
  double *target;
  double *alpha_in=NULL;
  KERNEL_CACHE *kernel_cache;
  LEARN_PARM learn_parm;
  KERNEL_PARM kernel_parm;
  MODEL *model=(MODEL *)my_malloc(sizeof(MODEL));

  read_input_parameters(argc,argv,docfile,modelfile,restartfile,&verbosity,
			&learn_parm,&kernel_parm);
  read_documents(docfile,&docs,&target,&totwords,&totdoc);
  if(restartfile[0]) alpha_in=read_alphas(restartfile,totdoc);

  if(kernel_parm.kernel_type == LINEAR)		kernel_cache=NULL;
  else           kernel_cache=kernel_cache_init(totdoc,learn_parm.kernel_cache_size);

  if(learn_parm.type == CLASSIFICATION) {
    svm_learn_classification(docs,target,totdoc,totwords,&learn_parm,
			     &kernel_parm,kernel_cache,model,alpha_in);
  }

  if(kernel_cache) {
    /* Free the memory used for the cache. */
    kernel_cache_cleanup(kernel_cache);
  }

  /* Warning: The model contains references to the original data 'docs'.
     If you want to free the original data, and only keep the model, you 
     have to make a deep copy of 'model'. */
  /* deep_copy_of_model=copy_model(model); */
  write_model(modelfile,model);

  free(alpha_in);
  free_model(model,0);
  for(i=0;i<totdoc;i++) 
    free_example(docs[i],1);
  free(docs);
  free(target);

  return(0);
}

/*---------------------------------------------------------------------------*/

void read_input_parameters(int argc,char *argv[],char *docfile,char *modelfile,
			   char *restartfile,long *verbosity,
			   LEARN_PARM *learn_parm,KERNEL_PARM *kernel_parm)
{
  long i;
  char type[100];
  
  /* set default */
  set_learning_defaults(learn_parm, kernel_parm);
  strcpy_s (modelfile, 10, "svm_model");
  strcpy_s (restartfile, 1, "");
  (*verbosity)=1;
  strcpy_s(type, 2,"c");

  for(i=1;(i<argc) && ((argv[i])[0] == '-');i++) {
    switch ((argv[i])[1]) 
      { 
      case '?': print_help(); exit(0);
      default: printf("\nUnrecognized option %s!\n\n",argv[i]);
	       print_help();
	       exit(0);
      }
  }
  if(i>=argc) {
    printf("\nNot enough input parameters!\n\n");
    wait_any_key();
    print_help();
    exit(0);
  }
  strcpy(docfile, argv[i]);
  if((i+1)<argc) {
	  strcpy(modelfile, argv[i + 1]);
  }
  if(learn_parm->svm_iter_to_shrink == -9999) {
    if(kernel_parm->kernel_type == LINEAR) 
      learn_parm->svm_iter_to_shrink=2;
    else
      learn_parm->svm_iter_to_shrink=100;
  }
  if(strcmp(type,"c")==0) {
    learn_parm->type=CLASSIFICATION;
  }
  else if(strcmp(type,"r")==0) {
    learn_parm->type=REGRESSION;
  }
  else if(strcmp(type,"p")==0) {
    learn_parm->type=RANKING;
  }
  else if(strcmp(type,"o")==0) {
    learn_parm->type=OPTIMIZATION;
  }
  else if(strcmp(type,"s")==0) {
    learn_parm->type=OPTIMIZATION;
    learn_parm->sharedslack=1;
  }
  else {
    printf("\nUnknown type '%s': Valid types are 'c' (classification), 'r' regession, and 'p' preference ranking.\n",type);
    wait_any_key();
    print_help();
    exit(0);
  }    
  if (!check_learning_parms(learn_parm, kernel_parm)) {
     wait_any_key();
     print_help();
     exit(0);
  }
}

void wait_any_key()
{
  printf("\n(more)\n");
  (void)getc(stdin);
}

void print_help(){}


