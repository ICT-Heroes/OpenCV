/***********************************************************************/
/*                                                                     */
/*   svm_loqo.c                                                        */
/*                                                                     */
/*   Interface to the PR_LOQO optimization package for SVM.            */
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 19.07.99                                                    */
/*                                                                     */
/*   Copyright (c) 1999  Universitaet Dortmund - All rights reserved   */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/

# include <math.h>
# include "svm_common.h"

/* Common Block Declarations */

long verbosity;

/* /////////////////////////////////////////////////////////////// */

#define	max(A, B)	((A) > (B) ? (A) : (B))
#define	min(A, B)	((A) < (B) ? (A) : (B))
#define sqr(A)          ((A) * (A))
#define	ABS(A)  	((A) > 0 ? (A) : (-(A)))

#define PREDICTOR 1
#define CORRECTOR 2

/*****************************************************************
replace this by any other function that will exit gracefully
in a larger system
***************************************************************/

void nrerror(char error_text[])
{
	printf("ERROR: terminating program - %s\n", error_text);
	exit(1);
}

/*****************************************************************
taken from numerical recipes and modified to accept pointers
moreover numerical recipes code seems to be buggy (at least the
ones on the web)

cholesky solver and backsubstitution
leaves upper right triangle intact (rows first order)
***************************************************************/

void choldc(double a[], int n, double p[])
{
	void nrerror(char error_text[]);
	int i, j, k;
	double sum;

	for (i = 0; i < n; i++){
		for (j = i; j < n; j++) {
			sum = a[n*i + j];
			for (k = i - 1; k >= 0; k--) sum -= a[n*i + k] * a[n*j + k];
			if (i == j) {
				if (sum <= 0.0)
					nrerror("choldc failed, matrix not positive definite");
				p[i] = sqrt(sum);
			}
			else a[n*j + i] = sum / p[i];
		}
	}
}

void cholsb(double a[], int n, double p[], double b[], double x[])
{
	int i, k;
	double sum;

	for (i = 0; i<n; i++) {
		sum = b[i];
		for (k = i - 1; k >= 0; k--) sum -= a[n*i + k] * x[k];
		x[i] = sum / p[i];
	}

	for (i = n - 1; i >= 0; i--) {
		sum = x[i];
		for (k = i + 1; k<n; k++) sum -= a[n*k + i] * x[k];
		x[i] = sum / p[i];
	}
}

/*****************************************************************
sometimes we only need the forward or backward pass of the
backsubstitution, hence we provide these two routines separately
***************************************************************/

void chol_forward(double a[], int n, double p[], double b[], double x[])
{
	int i, k;
	double sum;

	for (i = 0; i<n; i++) {
		sum = b[i];
		for (k = i - 1; k >= 0; k--) sum -= a[n*i + k] * x[k];
		x[i] = sum / p[i];
	}
}

void chol_backward(double a[], int n, double p[], double b[], double x[])
{
	int i, k;
	double sum;

	for (i = n - 1; i >= 0; i--) {
		sum = b[i];
		for (k = i + 1; k<n; k++) sum -= a[n*k + i] * x[k];
		x[i] = sum / p[i];
	}
}

/*****************************************************************
solves the system | -H_x A' | |x_x| = |c_x|
|  A   H_y| |x_y|   |c_y|

with H_x (and H_y) positive (semidefinite) matrices
and n, m the respective sizes of H_x and H_y

for variables see pg. 48 of notebook or do the calculations on a
sheet of paper again

predictor solves the whole thing, corrector assues that H_x didn't
change and relies on the results of the predictor. therefore do
_not_ modify workspace

if you want to speed tune anything in the code here's the right
place to do so: about 95% of the time is being spent in
here. something like an iterative refinement would be nice,
especially when switching from double to single precision. if you
have a fast parallel cholesky use it instead of the numrec
implementations.

side effects: changes H_y (but this is just the unit matrix or zero anyway
in our case)
***************************************************************/

void solve_reduced(int n, int m, double h_x[], double h_y[],
	double a[], double x_x[], double x_y[],
	double c_x[], double c_y[],
	double workspace[], int step)
{
	int i, j, k;

	double *p_x;
	double *p_y;
	double *t_a;
	double *t_c;
	double *t_y;

	p_x = workspace;		/* together n + m + n*m + n + m = n*(m+2)+2*m */
	p_y = p_x + n;
	t_a = p_y + m;
	t_c = t_a + n*m;
	t_y = t_c + n;

	if (step == PREDICTOR) {
		choldc(h_x, n, p_x);	/* do cholesky decomposition */

		for (i = 0; i<m; i++)         /* forward pass for A' */
			chol_forward(h_x, n, p_x, a + i*n, t_a + i*n);

		for (i = 0; i<m; i++)         /* compute (h_y + a h_x^-1A') */
			for (j = i; j<m; j++)
				for (k = 0; k<n; k++)
					h_y[m*i + j] += t_a[n*j + k] * t_a[n*i + k];

		choldc(h_y, m, p_y);	/* and cholesky decomposition */
	}

	chol_forward(h_x, n, p_x, c_x, t_c);
	/* forward pass for c */

	for (i = 0; i<m; i++) {		/* and solve for x_y */
		t_y[i] = c_y[i];
		for (j = 0; j<n; j++)
			t_y[i] += t_a[i*n + j] * t_c[j];
	}

	cholsb(h_y, m, p_y, t_y, x_y);

	for (i = 0; i<n; i++) {		/* finally solve for x_x */
		t_c[i] = -t_c[i];
		for (j = 0; j<m; j++)
			t_c[i] += t_a[j*n + i] * x_y[j];
	}

	chol_backward(h_x, n, p_x, t_c, x_x);
}

/*****************************************************************
matrix vector multiplication (symmetric matrix but only one triangle
given). computes m*x = y
no need to tune it as it's only of O(n^2) but cholesky is of
O(n^3). so don't waste your time _here_ although it isn't very
elegant.
***************************************************************/

void matrix_vector(int n, double m[], double x[], double y[])
{
	int i, j;

	for (i = 0; i<n; i++) {
		y[i] = m[(n + 1) * i] * x[i];

		for (j = 0; j<i; j++)
			y[i] += m[i + n*j] * x[j];

		for (j = i + 1; j<n; j++)
			y[i] += m[n*i + j] * x[j];
	}
}
# define DEF_PRECISION_LINEAR    1E-8
# define DEF_PRECISION_NONLINEAR 1E-14
int pr_loqo(int n, int m, double c[], double h_x[], double a[], double b[],
	double l[], double u[], double primal[], double dual[],
	int verb, double sigfig_max, int counter_max,
	double margin, double bound, int restart)
{
	/* the knobs to be tuned ... */
	/* double margin = -0.95;	   we will go up to 95% of the
	distance between old variables and zero */
	/* double bound = 10;		   preset value for the start. small
	values give good initial
	feasibility but may result in slow
	convergence afterwards: we're too
	close to zero */
	/* to be allocated */
	double *workspace;
	double *diag_h_x;
	double *h_y;
	double *c_x;
	double *c_y;
	double *h_dot_x;
	double *rho;
	double *nu;
	double *tau;
	double *sigma;
	double *gamma_z;
	double *gamma_s;

	double *hat_nu;
	double *hat_tau;

	double *delta_x;
	double *delta_y;
	double *delta_s;
	double *delta_z;
	double *delta_g;
	double *delta_t;

	double *d;

	/* from the header - pointers into primal and dual */
	double *x;
	double *y;
	double *g;
	double *z;
	double *s;
	double *t;

	/* auxiliary variables */
	double b_plus_1;
	double c_plus_1;

	double x_h_x;
	double primal_inf;
	double dual_inf;

	double sigfig;
	double primal_obj, dual_obj;
	double mu;
	double alfa, step;
	int counter = 0;

	int status = STILL_RUNNING;

	int i, j, k;

	/* memory allocation */
	workspace = (double*)malloc((n*(m + 2) + 2 * m)*sizeof(double));
	diag_h_x = (double*)malloc(n*sizeof(double));
	h_y = (double*)malloc(m*m*sizeof(double));
	c_x = (double*)malloc(n*sizeof(double));
	c_y = (double*)malloc(m*sizeof(double));
	h_dot_x = (double*)malloc(n*sizeof(double));

	rho = (double*)malloc(m*sizeof(double));
	nu = (double*)malloc(n*sizeof(double));
	tau = (double*)malloc(n*sizeof(double));
	sigma = (double*)malloc(n*sizeof(double));

	gamma_z = (double*)malloc(n*sizeof(double));
	gamma_s = (double*)malloc(n*sizeof(double));

	hat_nu = (double*)malloc(n*sizeof(double));
	hat_tau = (double*)malloc(n*sizeof(double));

	delta_x = (double*)malloc(n*sizeof(double));
	delta_y = (double*)malloc(m*sizeof(double));
	delta_s = (double*)malloc(n*sizeof(double));
	delta_z = (double*)malloc(n*sizeof(double));
	delta_g = (double*)malloc(n*sizeof(double));
	delta_t = (double*)malloc(n*sizeof(double));

	d = (double*)malloc(n*sizeof(double));

	/* pointers into the external variables */
	x = primal;			/* n */
	g = x + n;			/* n */
	t = g + n;			/* n */

	y = dual;			/* m */
	z = y + m;			/* n */
	s = z + n;			/* n */

	/* initial settings */
	b_plus_1 = 1;
	c_plus_1 = 0;
	for (i = 0; i<n; i++) c_plus_1 += c[i];

	/* get diagonal terms */
	for (i = 0; i<n; i++) diag_h_x[i] = h_x[(n + 1)*i];

	/* starting point */
	if (restart == 1) {
		/* x, y already preset */
		for (i = 0; i<n; i++) {	/* compute g, t for primal feasibility */
			g[i] = max(ABS(x[i] - l[i]), bound);
			t[i] = max(ABS(u[i] - x[i]), bound);
		}

		matrix_vector(n, h_x, x, h_dot_x); /* h_dot_x = h_x * x */

		for (i = 0; i<n; i++) {	/* sigma is a dummy variable to calculate z, s */
			sigma[i] = c[i] + h_dot_x[i];
			for (j = 0; j<m; j++)
				sigma[i] -= a[n*j + i] * y[j];

			if (sigma[i] > 0) {
				s[i] = bound;
				z[i] = sigma[i] + bound;
			}
			else {
				s[i] = bound - sigma[i];
				z[i] = bound;
			}
		}
	}
	else {			/* use default start settings */
		for (i = 0; i<m; i++)
			for (j = i; j<m; j++)
				h_y[i*m + j] = (i == j) ? 1 : 0;

		for (i = 0; i<n; i++) {
			c_x[i] = c[i];
			h_x[(n + 1)*i] += 1;
		}

		for (i = 0; i<m; i++)
			c_y[i] = b[i];

		/* and solve the system [-H_x A'; A H_y] [x, y] = [c_x; c_y] */
		solve_reduced(n, m, h_x, h_y, a, x, y, c_x, c_y, workspace, PREDICTOR);

		/* initialize the other variables */
		for (i = 0; i<n; i++) {
			g[i] = max(ABS(x[i] - l[i]), bound);
			z[i] = max(ABS(x[i]), bound);
			t[i] = max(ABS(u[i] - x[i]), bound);
			s[i] = max(ABS(x[i]), bound);
		}
	}

	for (i = 0, mu = 0; i<n; i++)
		mu += z[i] * g[i] + s[i] * t[i];
	mu = mu / (2 * n);

	/* the main loop */
	if (verb >= STATUS) {
		printf("counter | pri_inf  | dual_inf  | pri_obj   | dual_obj  | ");
		printf("sigfig | alpha  | nu \n");
		printf("-------------------------------------------------------");
		printf("---------------------------\n");
	}

	while (status == STILL_RUNNING) {
		/* predictor */

		/* put back original diagonal values */
		for (i = 0; i<n; i++)
			h_x[(n + 1) * i] = diag_h_x[i];

		matrix_vector(n, h_x, x, h_dot_x); /* compute h_dot_x = h_x * x */

		for (i = 0; i<m; i++) {
			rho[i] = b[i];
			for (j = 0; j<n; j++)
				rho[i] -= a[n*i + j] * x[j];
		}

		for (i = 0; i<n; i++) {
			nu[i] = l[i] - x[i] + g[i];
			tau[i] = u[i] - x[i] - t[i];

			sigma[i] = c[i] - z[i] + s[i] + h_dot_x[i];
			for (j = 0; j<m; j++)
				sigma[i] -= a[n*j + i] * y[j];

			gamma_z[i] = -z[i];
			gamma_s[i] = -s[i];
		}

		/* instrumentation */
		x_h_x = 0;
		primal_inf = 0;
		dual_inf = 0;

		for (i = 0; i<n; i++) {
			x_h_x += h_dot_x[i] * x[i];
			primal_inf += sqr(tau[i]);
			primal_inf += sqr(nu[i]);
			dual_inf += sqr(sigma[i]);
		}
		for (i = 0; i<m; i++)
			primal_inf += sqr(rho[i]);
		primal_inf = sqrt(primal_inf) / b_plus_1;
		dual_inf = sqrt(dual_inf) / c_plus_1;

		primal_obj = 0.5 * x_h_x;
		dual_obj = -0.5 * x_h_x;
		for (i = 0; i<n; i++) {
			primal_obj += c[i] * x[i];
			dual_obj += l[i] * z[i] - u[i] * s[i];
		}
		for (i = 0; i<m; i++)
			dual_obj += b[i] * y[i];

		sigfig = log10(ABS(primal_obj) + 1) -
			log10(ABS(primal_obj - dual_obj));
		sigfig = max(sigfig, 0);

		/* the diagnostics - after we computed our results we will
		analyze them */

		if (counter > counter_max) status = ITERATION_LIMIT;
		if (sigfig  > sigfig_max)  status = OPTIMAL_SOLUTION;
		if (primal_inf > 10e100)   status = PRIMAL_INFEASIBLE;
		if (dual_inf > 10e100)     status = DUAL_INFEASIBLE;
		if ((primal_inf > 10e100) & (dual_inf > 10e100)) status = PRIMAL_AND_DUAL_INFEASIBLE;
		if (ABS(primal_obj) > 10e100) status = PRIMAL_UNBOUNDED;
		if (ABS(dual_obj) > 10e100) status = DUAL_UNBOUNDED;

		/* write some nice routine to enforce the time limit if you
		_really_ want, however it's quite useless as you can compute
		the time from the maximum number of iterations as every
		iteration costs one cholesky decomposition plus a couple of
		backsubstitutions */

		/* generate report */
		if ((verb >= FLOOD) | ((verb == STATUS) & (status != 0)))
			printf("%7i | %.2e | %.2e | % .2e | % .2e | %6.3f | %.4f | %.2e\n",
			counter, primal_inf, dual_inf, primal_obj, dual_obj,
			sigfig, alfa, mu);

		counter++;

		if (status == 0) {		/* we may keep on going, otherwise
								it'll cost one loop extra plus a
								messed up main diagonal of h_x */
			/* intermediate variables (the ones with hat) */
			for (i = 0; i<n; i++) {
				hat_nu[i] = nu[i] + g[i] * gamma_z[i] / z[i];
				hat_tau[i] = tau[i] - t[i] * gamma_s[i] / s[i];
				/* diagonal terms */
				d[i] = z[i] / g[i] + s[i] / t[i];
			}

			/* initialization before the cholesky solver */
			for (i = 0; i<n; i++) {
				h_x[(n + 1)*i] = diag_h_x[i] + d[i];
				c_x[i] = sigma[i] - z[i] * hat_nu[i] / g[i] -
					s[i] * hat_tau[i] / t[i];
			}
			for (i = 0; i<m; i++) {
				c_y[i] = rho[i];
				for (j = i; j<m; j++)
					h_y[m*i + j] = 0;
			}

			/* and do it */
			solve_reduced(n, m, h_x, h_y, a, delta_x, delta_y, c_x, c_y, workspace,
				PREDICTOR);

			for (i = 0; i<n; i++) {
				/* backsubstitution */
				delta_s[i] = s[i] * (delta_x[i] - hat_tau[i]) / t[i];
				delta_z[i] = z[i] * (hat_nu[i] - delta_x[i]) / g[i];

				delta_g[i] = g[i] * (gamma_z[i] - delta_z[i]) / z[i];
				delta_t[i] = t[i] * (gamma_s[i] - delta_s[i]) / s[i];

				/* central path (corrector) */
				gamma_z[i] = mu / g[i] - z[i] - delta_z[i] * delta_g[i] / g[i];
				gamma_s[i] = mu / t[i] - s[i] - delta_s[i] * delta_t[i] / t[i];

				/* (some more intermediate variables) the hat variables */
				hat_nu[i] = nu[i] + g[i] * gamma_z[i] / z[i];
				hat_tau[i] = tau[i] - t[i] * gamma_s[i] / s[i];

				/* initialization before the cholesky */
				c_x[i] = sigma[i] - z[i] * hat_nu[i] / g[i] - s[i] * hat_tau[i] / t[i];
			}

			for (i = 0; i<m; i++) {	/* comput c_y and rho */
				c_y[i] = rho[i];
				for (j = i; j<m; j++)
					h_y[m*i + j] = 0;
			}

			/* and do it */
			solve_reduced(n, m, h_x, h_y, a, delta_x, delta_y, c_x, c_y, workspace,
				CORRECTOR);

			for (i = 0; i<n; i++) {
				/* backsubstitution */
				delta_s[i] = s[i] * (delta_x[i] - hat_tau[i]) / t[i];
				delta_z[i] = z[i] * (hat_nu[i] - delta_x[i]) / g[i];

				delta_g[i] = g[i] * (gamma_z[i] - delta_z[i]) / z[i];
				delta_t[i] = t[i] * (gamma_s[i] - delta_s[i]) / s[i];
			}

			alfa = -1;
			for (i = 0; i<n; i++) {
				alfa = min(alfa, delta_g[i] / g[i]);
				alfa = min(alfa, delta_t[i] / t[i]);
				alfa = min(alfa, delta_s[i] / s[i]);
				alfa = min(alfa, delta_z[i] / z[i]);
			}
			alfa = (margin - 1) / alfa;

			/* compute mu */
			for (i = 0, mu = 0; i<n; i++)
				mu += z[i] * g[i] + s[i] * t[i];
			mu = mu / (2 * n);
			mu = mu * sqr((alfa - 1) / (alfa + 10));

			for (i = 0; i<n; i++) {
				x[i] += alfa * delta_x[i];
				g[i] += alfa * delta_g[i];
				t[i] += alfa * delta_t[i];
				z[i] += alfa * delta_z[i];
				s[i] += alfa * delta_s[i];
			}

			for (i = 0; i<m; i++)
				y[i] += alfa * delta_y[i];
		}
	}
	if ((status == 1) && (verb >= STATUS)) {
		printf("----------------------------------------------------------------------------------\n");
		printf("optimization converged\n");
	}

	/* free memory */
	free(workspace);
	free(diag_h_x);
	free(h_y);
	free(c_x);
	free(c_y);
	free(h_dot_x);

	free(rho);
	free(nu);
	free(tau);
	free(sigma);
	free(gamma_z);
	free(gamma_s);

	free(hat_nu);
	free(hat_tau);

	free(delta_x);
	free(delta_y);
	free(delta_s);
	free(delta_z);
	free(delta_g);
	free(delta_t);

	free(d);

	/* and return to sender */
	return status;
}


double *optimize_qp();
//double *optimize_qp(QP* qp, double* epsilon_crit, long nx, double* threshold, LEARN_PARM *learn_parm);
double *primal=0,*dual=0;
double init_margin=0.15;
long   init_iter=500,precision_violations=0;
double model_b;
double opt_precision=DEF_PRECISION_LINEAR;

/* /////////////////////////////////////////////////////////////// */

void *my_malloc();

double *optimize_qp(QP* qp, double* epsilon_crit, long nx, double* threshold, LEARN_PARM *learn_parm)
/* start the optimizer and return the optimal values */
{
  register long i,j,result;
  double margin,obj_before,obj_after;
  double sigdig,dist,epsilon_loqo;
  int iter;
 
  if(!primal) { /* allocate memory at first call */
    primal=(double *)my_malloc(sizeof(double)*nx*3);
    dual=(double *)my_malloc(sizeof(double)*(nx*2+1));
  }
  
  if(verbosity>=4) { /* really verbose */
    printf("\n\n");
    for(i=0;i<qp->opt_n;i++) {
      printf("%f: ",qp->opt_g0[i]);
      for(j=0;j<qp->opt_n;j++) {
	printf("%f ",qp->opt_g[i*qp->opt_n+j]);
      }
      printf(": a%ld=%.10f < %f",i,qp->opt_xinit[i],qp->opt_up[i]);
      printf(": y=%f\n",qp->opt_ce[i]);
    }
    for(j=0;j<qp->opt_m;j++) {
      printf("EQ-%ld: %f*a0",j,qp->opt_ce[j]);
      for(i=1;i<qp->opt_n;i++) {
	printf(" + %f*a%ld",qp->opt_ce[i],i);
      }
      printf(" = %f\n\n",-qp->opt_ce0[0]);
    }
}

  obj_before=0; /* calculate objective before optimization */
  for(i=0;i<qp->opt_n;i++) {
    obj_before+=(qp->opt_g0[i]*qp->opt_xinit[i]);
    obj_before+=(0.5*qp->opt_xinit[i]*qp->opt_xinit[i]*qp->opt_g[i*qp->opt_n+i]);
    for(j=0;j<i;j++) {
      obj_before+=(qp->opt_xinit[j]*qp->opt_xinit[i]*qp->opt_g[j*qp->opt_n+i]);
    }
  }

  result=STILL_RUNNING;
  qp->opt_ce0[0]*=(-1.0);
  /* Run pr_loqo. If a run fails, try again with parameters which lead */
  /* to a slower, but more robust setting. */
  for(margin=init_margin,iter=init_iter;
      (margin<=0.9999999) && (result!=OPTIMAL_SOLUTION);) {
    sigdig=-log10(opt_precision);

    result=pr_loqo((int)qp->opt_n,(int)qp->opt_m,
		   (double *)qp->opt_g0,(double *)qp->opt_g,
		   (double *)qp->opt_ce,(double *)qp->opt_ce0,
		   (double *)qp->opt_low,(double *)qp->opt_up,
		   (double *)primal,(double *)dual, 
		   (int)(verbosity-2),
		   (double)sigdig,(int)iter, 
		   (double)margin,(double)(qp->opt_up[0])/4.0,(int)0);

    if(isNan(dual[0])) {     /* check for choldc problem */
      if(verbosity>=2) {
	printf("NOTICE: Restarting PR_LOQO with more conservative parameters.\n");
      }
      if(init_margin<0.80) { /* become more conservative in general */
	init_margin=(4.0*margin+1.0)/5.0;
      }
      margin=(margin+1.0)/2.0;
      (opt_precision)*=10.0;   /* reduce precision */
      if(verbosity>=2) {
	printf("NOTICE: Reducing precision of PR_LOQO.\n");
      }
    }
    else if(result!=OPTIMAL_SOLUTION) {
      iter+=2000; 
      init_iter+=10;
      (opt_precision)*=10.0;   /* reduce precision */
      if(verbosity>=2) {
	printf("NOTICE: Reducing precision of PR_LOQO due to (%ld).\n",result);
      }      
    }
  }

  if(qp->opt_m)         /* Thanks to Alex Smola for this hint */
    model_b=dual[0];
  else
    model_b=0;

  /* Check the precision of the alphas. If results of current optimization */
  /* violate KT-Conditions, relax the epsilon on the bounds on alphas. */
  epsilon_loqo=1E-10;
  for(i=0;i<qp->opt_n;i++) {
    dist=-model_b*qp->opt_ce[i]; 
    dist+=(qp->opt_g0[i]+1.0);
    for(j=0;j<i;j++) {
      dist+=(primal[j]*qp->opt_g[j*qp->opt_n+i]);
    }
    for(j=i;j<qp->opt_n;j++) {
      dist+=(primal[j]*qp->opt_g[i*qp->opt_n+j]);
    }
    /*  printf("LOQO: a[%d]=%f, dist=%f, b=%f\n",i,primal[i],dist,dual[0]); */
    if((primal[i]<(qp->opt_up[i]-epsilon_loqo)) && (dist < (1.0-(*epsilon_crit)))) {
      epsilon_loqo=(qp->opt_up[i]-primal[i])*2.0;
    }
    else if((primal[i]>(0+epsilon_loqo)) && (dist > (1.0+(*epsilon_crit)))) {
      epsilon_loqo=primal[i]*2.0;
    }
  }

  for(i=0;i<qp->opt_n;i++) {  /* clip alphas to bounds */
    if(primal[i]<=(0+epsilon_loqo)) {
      primal[i]=0;
    }
    else if(primal[i]>=(qp->opt_up[i]-epsilon_loqo)) {
      primal[i]=qp->opt_up[i];
    }
  }

  obj_after=0;  /* calculate objective after optimization */
  for(i=0;i<qp->opt_n;i++) {
    obj_after+=(qp->opt_g0[i]*primal[i]);
    obj_after+=(0.5*primal[i]*primal[i]*qp->opt_g[i*qp->opt_n+i]);
    for(j=0;j<i;j++) {
      obj_after+=(primal[j]*primal[i]*qp->opt_g[j*qp->opt_n+i]);
    }
  }

  /* if optimizer returned NAN values, reset and retry with smaller */
  /* working set. */
  if(isNan(obj_after) || isNan(model_b)) {
    for(i=0;i<qp->opt_n;i++) {
      primal[i]=qp->opt_xinit[i];
    }     
    model_b=0;
    if(learn_parm->svm_maxqpsize>2) {
      learn_parm->svm_maxqpsize--;  /* decrease size of qp-subproblems */
    }
  }

  if(obj_after >= obj_before) { /* check whether there was progress */
    (opt_precision)/=100.0;
    precision_violations++;
    if(verbosity>=2) {
      printf("NOTICE: Increasing Precision of PR_LOQO.\n");
    }
  }

  if(precision_violations > 500) { 
    (*epsilon_crit)*=10.0;
    precision_violations=0;
    if(verbosity>=1) {
      printf("\nWARNING: Relaxing epsilon on KT-Conditions.\n");
    }
  }	  

  (*threshold)=model_b;

  if(result!=OPTIMAL_SOLUTION) {
    printf("\nERROR: PR_LOQO did not converge. \n");
    return(qp->opt_xinit);
  }
  else {
    return(primal);
  }
}

