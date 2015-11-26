
#include "svm_match.h"

void MatchData(){

	char* testFile, *learningFile, *outputFile, *line, *comment;
	DOC *doc;   /* test example */
	WORD *words;
	long max_docs, max_words_doc, lld, totdoc = 0, queryid, slackid, correct = 0,
		incorrect = 0, no_accuracy = 0, res_a = 0, res_b = 0, res_c = 0, res_d = 0, wnum, pred_format;
	double t1, runtime = 0, dist, doc_label, costfactor;
	FILE *predfl, *docfl;
	MODEL *model;

	verbosity = 2;
	pred_format = 1;
	testFile = "test.dat";
	learningFile = "learning.dat";
	outputFile = "output.dat";

	nol_ll(testFile, &max_docs, &max_words_doc, &lld); /* scan size of input file */
	max_words_doc += 2;
	lld += 2;

	line = (char *)my_malloc(sizeof(char)*lld);
	words = (WORD *)my_malloc(sizeof(WORD)*(max_words_doc + 10));

	model = read_model(learningFile);
	add_weight_vector_to_linear_model(model);

	printf("Classifying test examples..");
	fflush(stdout);

	if ((docfl = fopen(testFile, "r")) == NULL)	{ perror(testFile); exit(1); }
	if ((predfl = fopen(outputFile, "w")) == NULL){ perror(outputFile); exit(1); }

	while ((!feof(docfl)) && fgets(line, (int)lld, docfl)) {
		if (line[0] == '#') continue;  /* line contains comments */
		parse_document(line, words, &doc_label, &queryid, &slackid, &costfactor, &wnum, max_words_doc, &comment);
		totdoc++;
		if (model->kernel_parm.kernel_type == LINEAR) {/* For linear kernel,     */
			for (long j = 0; (words[j]).wnum != 0; j++) {     /* check if feature numbers   */
				if ((words[j]).wnum > model->totwords)   /* are not larger than in     */
					(words[j]).wnum = 0;                  /* model. Remove feature if   */
			}                                       /* necessary.                 */
		}
		doc = create_example(-1, 0, 0, 0.0, create_svector(words, comment, 1.0));
		t1 = get_runtime();
		dist = classify_example_linear(model, doc);
		runtime += (get_runtime() - t1);
		free_example(doc, 1);

		if (dist > 0) {
			if (pred_format == 0) { /* old weired output format */
				fprintf(predfl, "%.8g:+1 %.8g:-1\n", dist, -dist);
			}
			if (doc_label > 0) correct++; else incorrect++;
			if (doc_label > 0) res_a++; else res_b++;
		}
		else {
			if (pred_format == 0) { /* old weired output format */
				fprintf(predfl, "%.8g:-1 %.8g:+1\n", -dist, dist);
			}
			if (doc_label < 0) correct++; else incorrect++;
			if (doc_label > 0) res_c++; else res_d++;
		}
		if (pred_format == 1) { /* output the value of decision function */
			fprintf(predfl, "%.8g\n", dist);
		}
		if ((int)(0.01 + (doc_label*doc_label)) != 1){
			no_accuracy = 1;
		} /* test data is not binary labeled */
		if (verbosity >= 2) {
			if (totdoc % 100 == 0) {
				printf("%ld..", totdoc); fflush(stdout);
			}
		}
	}
	fclose(predfl);
	fclose(docfl);
	free(line);
	free(words);
	free_model(model, 1);

	if (verbosity >= 2) {
		printf("done\n");
		printf("Runtime (without IO) in cpu-seconds: %.2f\n",
			(float)(runtime / 100.0));
	}
	if ((!no_accuracy) && (verbosity >= 1)) {
		printf("Accuracy on test set: %.2f%% (%ld correct, %ld incorrect, %ld total)\n", (float)(correct)*100.0 / totdoc, correct, incorrect, totdoc);
		printf("Precision/recall on test set: %.2f%%/%.2f%%\n", (float)(res_a)*100.0 / (res_a + res_b), (float)(res_a)*100.0 / (res_a + res_c));
	}
}

void MatchData(float* image){

	char* testFile, *learningFile, *outputFile, *line, *comment;
	DOC *doc;   /* test example */
	WORD *words;
	long max_docs, max_words_doc, lld, totdoc = 0, queryid, slackid, correct = 0,
		incorrect = 0, no_accuracy = 0, res_a = 0, res_b = 0, res_c = 0, res_d = 0, wnum, pred_format;
	double t1, runtime = 0, dist, doc_label, costfactor;
	FILE *predfl, *docfl;
	MODEL *model;

	verbosity = 2;
	pred_format = 1;
	testFile = "test.dat";
	learningFile = "learning.dat";
	outputFile = "output.dat";
	
	nol_ll(testFile, &max_docs, &max_words_doc, &lld); /* scan size of input file */
	max_words_doc += 2;
	lld += 2;

	line = (char *)my_malloc(sizeof(char)*lld);
	words = (WORD *)my_malloc(sizeof(WORD)*(max_words_doc + 10));

	model = read_model(learningFile);
	add_weight_vector_to_linear_model(model);

	printf("Classifying test examples..");
	fflush(stdout);

	if ((docfl = fopen(testFile, "r")) == NULL)	{ perror(testFile); exit(1); }
	if ((predfl = fopen(outputFile, "w")) == NULL){ perror(outputFile); exit(1); }

	while ((!feof(docfl)) && fgets(line, (int)lld, docfl) ) {
		if (line[0] == '#') continue;  /* line contains comments */
		parse_document(line, words, &doc_label, &queryid, &slackid, &costfactor, &wnum, max_words_doc, &comment);
		totdoc++;
		if (model->kernel_parm.kernel_type == LINEAR) {/* For linear kernel,     */
			for (long j = 0; (words[j]).wnum != 0; j++) {     /* check if feature numbers   */
				if ((words[j]).wnum > model->totwords)   /* are not larger than in     */
					(words[j]).wnum = 0;                  /* model. Remove feature if   */
			}                                       /* necessary.                 */
		}
		doc = create_example(-1, 0, 0, 0.0, create_svector(words, comment, 1.0));
		t1 = get_runtime();
		dist = classify_example_linear(model, doc);
		runtime += (get_runtime() - t1);
		free_example(doc, 1);

		if (dist > 0) {
			if (pred_format == 0) { /* old weired output format */
				fprintf(predfl, "%.8g:+1 %.8g:-1\n", dist, -dist);
			}
			if (doc_label > 0) correct++; else incorrect++;
			if (doc_label > 0) res_a++; else res_b++;
		}
		else {
			if (pred_format == 0) { /* old weired output format */
				fprintf(predfl, "%.8g:-1 %.8g:+1\n", -dist, dist);
			}
			if (doc_label < 0) correct++; else incorrect++;
			if (doc_label > 0) res_c++; else res_d++;
		}
		if (pred_format == 1) { /* output the value of decision function */
			fprintf(predfl, "%.8g\n", dist);
		}
		if ((int)(0.01 + (doc_label*doc_label)) != 1){
			no_accuracy = 1;
		} /* test data is not binary labeled */
		if (verbosity >= 2) {
			if (totdoc % 100 == 0) {
				printf("%ld..", totdoc); fflush(stdout);
			}
		}
	}
	fclose(predfl);
	fclose(docfl);
	free(line);
	free(words);
	free_model(model, 1);

	if (verbosity >= 2) {
		printf("done\n");
		printf("Runtime (without IO) in cpu-seconds: %.2f\n",
			(float)(runtime / 100.0));
	}
	if ((!no_accuracy) && (verbosity >= 1)) {
		printf("Accuracy on test set: %.2f%% (%ld correct, %ld incorrect, %ld total)\n", (float)(correct)*100.0 / totdoc, correct, incorrect, totdoc);
		printf("Precision/recall on test set: %.2f%%/%.2f%%\n", (float)(res_a)*100.0 / (res_a + res_b), (float)(res_a)*100.0 / (res_a + res_c));
	}
}