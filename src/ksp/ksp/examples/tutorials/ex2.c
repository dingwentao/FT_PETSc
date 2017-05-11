	
static char help[] = "Solves a linear system in parallel with KSP.\n\
Input parameters include:\n\
  -random_exact_sol : use a random exact solution vector\n\
  -view_exact_sol   : write exact solution vector to stdout\n\
  -m <mesh_x>       : number of mesh points in x-direction\n\
  -n <mesh_n>       : number of mesh points in y-direction\n\n";

/*T
   Concepts: KSP^basic parallel example;
   Concepts: KSP^Laplacian, 2d
   Concepts: Laplacian, 2d
   Processors: n
T*/

/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/
#include <time.h>
#include <petscksp.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Vec            x,b,u;  /* approx solution, RHS, exact solution */
  Mat            A;        /* linear system matrix */
  KSP            ksp;     /* linear solver context */
  PetscRandom    rctx;     /* random number generator context */
  PetscReal      norm;     /* norm of solution error */
  PetscInt       its;
  PetscErrorCode ierr;
  PetscBool      flg = PETSC_FALSE;
  char           file[PETSC_MAX_PATH_LEN];     /* input file name */
  PetscViewer    view;
  PetscInt		 N;
  PetscMPIInt    rank,size;
  PetscReal		 normu;
  PetscInt		solver_type, error_type;		/* solver_type: 0 - Original, 1 - Algorithm 1, 3 - Algorithm 2 */
  PetscInt		itv_c, itv_d;
  PetscInt		inj_itr, inj_times, inj_num;

  PetscInitialize(&argc,&args,(char*)0,help);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  ierr = PetscOptionsGetString(NULL,"-f",file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate binary file with the -f option");
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&view);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSBAIJ);CHKERRQ(ierr);
  ierr = MatLoad(A,view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  ierr = MatGetSize(A,&N,NULL);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = VecSetRandom(u,rctx);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = MatMult(A,u,b);CHKERRQ(ierr);
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  
  solver_type = atoi(args[argc-3]);
  error_type = atoi(args[argc-2]);
  inj_itr = atoi(args[argc-1]);
//  solver_type = atoi(args[argc-7]);
//  error_type = atoi(args[argc-6]);
//  itv_c = atoi(args[argc-5]);
//  itv_d = atoi(args[argc-4]);
//  inj_itr = atoi(args[argc-3]);
//  inj_times = atoi(args[argc-2]);
//  inj_num = atoi(args[argc-1]);
//  if (solver_type==4 || solver_type==5)
//  {
//	  srand (time(NULL));
//	  inj_itr = itv_c*itv_d+rand()/(RAND_MAX/1000+1);
//  }
//
//  ierr = KSPSetOnlineABFT(ksp,solver_type,error_type,itv_c,itv_d,inj_itr,inj_times, inj_num);
  ierr = KSPSetOnlineABFT(ksp,0,error_type,54,2,inj_itr,0,0);
  ierr = KSPSetTolerances(ksp,1.e-2/N,1.e-50,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  struct timespec start, end;
  long long int local_diff, global_diff;
  clock_gettime(CLOCK_REALTIME, &start);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  clock_gettime(CLOCK_REALTIME, &end);
  local_diff = 1000000000L*(end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec);\
  MPI_Reduce(&local_diff, &global_diff, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
//  PetscPrintf(MPI_COMM_WORLD,"Solver type = %d\n",solver_type);
//  PetscPrintf(MPI_COMM_WORLD,"0 - Original, 1- Offline-residual, 4 - Algorithm 1, 5 - Algorithm 2\n");
//  PetscPrintf(MPI_COMM_WORLD,"Error type = %d\n",error_type);
//  PetscPrintf(MPI_COMM_WORLD,"0 - No error, 1 - One error in W, 2 - Two errors in W, 3 - One error in P, 4 - Two errors in P, 5 - One error in X\n");
//  PetscPrintf(MPI_COMM_WORLD,"6 - Two errors in X, 7 - One error in R, 8 - Two errors in R, 9 - One error in Z, 10 - Two errors in Z\n");
  PetscPrintf(MPI_COMM_WORLD,"Elapsed time of solver = % lf nanoseconds\n", (double)(global_diff)/size);
  ierr = VecAXPY(x,-1.0,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = VecNorm(u,NORM_2,&normu);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Relative norm of error = %lf\n",norm/normu);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %d\n",its);CHKERRQ(ierr);	
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
