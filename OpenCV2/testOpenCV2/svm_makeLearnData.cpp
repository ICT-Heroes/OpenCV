#include "svm_makeLearnData.h"

void MakeLearnData(){
	char* trainingfile = "data.dat";		// file with training examples
	char* outputfile = "learning.dat";		// file for resulting classifier
	char* restartfile = "";					// file with initial alphas
	DOC **docs;  // training examples
	long totwords, totdoc, i;
	double *target;
	LEARN_PARM learn_parm;
	KERNEL_PARM kernel_parm;
	MODEL *model = (MODEL *)my_malloc(sizeof(MODEL));
	set_learning_defaults(&learn_parm, &kernel_parm);
	verbosity = 1;
	learn_parm.svm_iter_to_shrink = 2;
	learn_parm.type = CLASSIFICATION;
	read_documents(trainingfile, &docs, &target, &totwords, &totdoc);
	svm_learn_classification(docs, target, totdoc, totwords, &learn_parm, &kernel_parm, NULL, model, NULL);
	write_model(outputfile, model);
	//«ÿ¡¶
	free_model(model, 0);
	for (i = 0; i < totdoc; i++)	free_example(docs[i], 1);
	free(docs);
	free(target);
}