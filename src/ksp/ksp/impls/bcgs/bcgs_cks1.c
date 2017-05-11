
#include <../src/ksp/ksp/impls/bcgs/bcgsimpl.h>       /*I  "petscksp.h"  I*/

#undef __FUNCT__
#define __FUNCT__ "KSPSetFromOptions_BCGS"
PetscErrorCode KSPSetFromOptions_BCGS(PetscOptions *PetscOptionsObject,KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"KSP BCGS Options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetUp_BCGS"
PetscErrorCode KSPSetUp_BCGS(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPSetWorkVecs(ksp,12);CHKERRQ(ierr); /* add predefined vectors C1 and checkpoint vectors CKPX,CKPP,CKPR,CKPRP,CKPV*/
//  ierr = KSPSetWorkVecs(ksp,6);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "KSPSolve_BCGS"
PetscErrorCode KSPSolve_BCGS(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    rho,rhoold,alpha,beta,omega,omegaold,d1;
  Vec            X,B,V,P,R,RP,T,S;
  PetscReal      dp    = 0.0,d2;
  KSP_BCGS       *bcgs = (KSP_BCGS*)ksp->data;

  /* Dingwen */
  PetscInt		itv_d, itv_c;
  PetscScalar   CKSX1,CKSR1,CKSV1,CKST1,CKSS1,CKSP1;
  PetscScalar	CKSRP;
  Vec			C1;
  Vec			CKPX,CKPP,CKPR,CKPRP,CKPV;
  PetscInt		CKPi;
  PetscScalar	CKPrhoold,CKPalpha,CKPomegaold;
  Mat			A,CKPA;
  PetscBool		flag = PETSC_TRUE;
  PetscScalar	v;
  PetscInt		pos;
  /* Dingwen */

  PetscFunctionBegin;
  X  = ksp->vec_sol;
  B  = ksp->vec_rhs;
  R  = ksp->work[0];
  RP = ksp->work[1];
  V  = ksp->work[2];
  T  = ksp->work[3];
  S  = ksp->work[4];
  P  = ksp->work[5];

  /* Dingwen */
  C1 = ksp->work[6];
  CKPX = ksp->work[7];
  CKPP = ksp->work[8];
  CKPR = ksp->work[9];
  CKPRP = ksp->work[10];
  CKPV = ksp->work[11];
  itv_c = 2;
  itv_d = 20;

//  itv_c = ksp->itv_c;
//  itv_d = ksp->itv_d;
  /* Dingwen */

  /* Compute initial preconditioned residual */
  ierr = KSPInitialResidual(ksp,X,V,T,R,B);CHKERRQ(ierr);

  /* with right preconditioning need to save initial guess to add to final solution */
  if (ksp->pc_side == PC_RIGHT && !ksp->guess_zero) {
    if (!bcgs->guess) {
      ierr = VecDuplicate(X,&bcgs->guess);CHKERRQ(ierr);
    }
    ierr = VecCopy(X,bcgs->guess);CHKERRQ(ierr);
    ierr = VecSet(X,0.0);CHKERRQ(ierr);
  }

  /* Test for nothing to do */
  if (ksp->normtype != KSP_NORM_NONE) {
    ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);
  }
  ierr       = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
  ksp->its   = 0;
  ksp->rnorm = dp;
  ierr       = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
  ierr = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
  ierr = KSPMonitor(ksp,0,dp);CHKERRQ(ierr);
  ierr = (*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) PetscFunctionReturn(0);

  /* Make the initial Rp == R */
  ierr = VecCopy(R,RP);CHKERRQ(ierr);

  rhoold   = 1.0;
  alpha    = 1.0;
  omegaold = 1.0;
  ierr     = VecSet(P,0.0);CHKERRQ(ierr);
  ierr     = VecSet(V,0.0);CHKERRQ(ierr);

  /* Dingwen */
  /* Checksum coefficients initialization */
  PetscInt n;
  PetscInt *index;
  PetscScalar *v1;
  ierr = VecGetSize(B,&n);
  v1 	= (PetscScalar *)malloc(n*sizeof(PetscScalar));
  index	= (PetscInt *)malloc(n*sizeof(PetscInt));
  for (i=0; i<n; i++)
  {
	  index[i] = i;
	  v1[i] = 1.0;
  }
  ierr	= VecSetValues(C1,n,index,v1,INSERT_VALUES);CHKERRQ(ierr);

  /* Checksum Initialization */
  ierr = VecDot(C1,RP,&CKSRP);CHKERRQ(ierr);					/* Compute the checksum(RP) */
  ierr = VecDot(C1,X,&CKSX1);CHKERRQ(ierr);						/* Compute the initial checksum(X) */
  ierr = VecDot(C1,R,&CKSR1);CHKERRQ(ierr);						/* Compute the initial checksum(R) */
  ierr = VecDot(C1,V,&CKSV1);CHKERRQ(ierr);						/* Compute the initial checksum(V) */
  ierr = VecDot(C1,T,&CKST1);CHKERRQ(ierr);						/* Compute the initial checksum(T) */
  ierr = VecDot(C1,S,&CKSS1);CHKERRQ(ierr);						/* Compute the initial checksum(S) */
  ierr = VecDot(C1,P,&CKSP1);CHKERRQ(ierr);						/* Compute the initial checksum(P) */

  /* Checkpoint Matrix A and Vector RP */
  ierr = VecCopy(RP,CKPRP);CHKERRQ(ierr);
  ierr = KSPGetOperators(ksp,&A,NULL);
  ierr = MatCreate(PETSC_COMM_WORLD,&CKPA);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSBAIJ);CHKERRQ(ierr);
  ierr = MatSetSizes(CKPA,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatDuplicate(A,MAT_COPY_VALUES,&CKPA);CHKERRQ(ierr);
  /* Dingwen */

  i=0;
  do {
	  /* Dingwen */
	  if ((i>0) && (i%itv_d == 0))
	  {
		  PetscScalar	sumX1,sumR1,sumRP;
		  ierr = VecDot(C1,X,&sumX1);CHKERRQ(ierr);
		  ierr = VecDot(C1,R,&sumR1);CHKERRQ(ierr);
		  ierr = VecDot(C1,RP,&sumRP);CHKERRQ(ierr);
		  if ((PetscAbsScalar(sumX1-CKSX1)/(n*n) > 1.0e-6) || (PetscAbsScalar(sumR1-CKSR1)/(n*n) > 1.0e-6) || (PetscAbsScalar(sumRP-CKSRP)/(n*n) > 1.0e-6))
		  {
	 			  /* Rollback and Recovery */
	 			  PetscPrintf (MPI_COMM_WORLD,"Recovery start...\n");
	 			  PetscPrintf (MPI_COMM_WORLD,"Rollback from iteration-%d to iteration-%d\n",i,CKPi);
	 			  ierr = MatDuplicate(CKPA,MAT_COPY_VALUES,&A);CHKERRQ(ierr); /* Recovery matrix A from checkpoint */
	 			  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
	 			  i = CKPi;
	 			  ierr = VecCopy(CKPRP,RP);CHKERRQ(ierr);				/* Recovery vector RP from checkpoint */
	 			  ierr = VecCopy(CKPX,X);CHKERRQ(ierr);					/* Recovery vector X from checkpoint */
	 			  ierr = VecDot(C1,X,&CKSX1);CHKERRQ(ierr);				/* Recovery checksum(X) from X */
	 			  ierr = VecCopy(CKPR,R);CHKERRQ(ierr);					/* Recovery vector R from checkpoint */
	 			  ierr = VecDot(C1,R,&CKSR1);CHKERRQ(ierr);				/* Recovery checksum(R) from R */
	 			  ierr = VecCopy(CKPP,P);CHKERRQ(ierr);					/* Recovery vector P from checkpoint */
	 			  ierr = VecDot(C1,P,&CKSP1);CHKERRQ(ierr);				/* Recovery checksum(P) from P */
	 			  ierr = VecCopy(CKPV,V);CHKERRQ(ierr);					/* Recovery vector V from checkpoint */
	 			  ierr = VecDot(C1,V,&CKSV1);CHKERRQ(ierr);				/* Recovery checksum(V) from V */
	 			  rhoold	= CKPrhoold;
	 			  alpha		= CKPalpha;
	 			  omegaold	= CKPomegaold;
	 			  PetscPrintf (MPI_COMM_WORLD,"Recovery end.\n");
		  }
		  else if (i%(itv_c*itv_d) == 0)
		  {
 			  /* Checkpoint */
 			  PetscPrintf (MPI_COMM_WORLD,"Checkpoint start...\n");
 			  PetscPrintf (MPI_COMM_WORLD,"Checkpoint at iteration-%d\n",i);
			  ierr = VecCopy(X,CKPX);CHKERRQ(ierr);
			  ierr = VecCopy(P,CKPP);CHKERRQ(ierr);
			  ierr = VecCopy(R,CKPR);CHKERRQ(ierr);
			  ierr = VecCopy(V,CKPV);CHKERRQ(ierr);
			  CKPi 			= i;
 			  CKPrhoold		= rhoold;
 			  CKPalpha		= alpha;
 			  CKPomegaold	= omegaold;
			  PetscPrintf (MPI_COMM_WORLD,"Checkpoint end.\n");
		  }
	  }
	/* Dingwen */

    ierr = VecDot(R,RP,&rho);CHKERRQ(ierr);       /*   rho <- (r,rp)      */
    beta = (rho/rhoold) * (alpha/omegaold);
    ierr = VecAXPBYPCZ(P,1.0,-omegaold*beta,beta,R,V);CHKERRQ(ierr);  /* p <- r - omega * beta* v + beta * p */
    /* Dingwen */
    /* Update checksum(P) */
    CKSP1 = CKSR1 - omegaold*beta*CKSV1 + beta*CKSP1;
    /* Dingwen */

    ierr = KSP_PCApplyBAorAB(ksp,P,V,T);CHKERRQ(ierr);  /*   v <- K p           */
    /* Dingwen */
    /* Update checksum(V) */
    ierr = VecDot(C1,V,&CKSV1);CHKERRQ(ierr);
    /* Dingwen */

    ierr = VecDot(V,RP,&d1);CHKERRQ(ierr);
    if (d1 == 0.0) {
      if (ksp->errorifnotconverged) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"KSPSolve has not converged due to Nan or Inf inner product");
      else {
        ksp->reason = KSP_DIVERGED_NANORINF;
        break;
      }
    }
    alpha = rho / d1;                 /*   a <- rho / (v,rp)  */
    ierr  = VecWAXPY(S,-alpha,V,R);CHKERRQ(ierr);     /*   s <- r - a v       */
    /* Dingwen */
    /* Update checksum(S) */
    CKSS1 = CKSR1 - alpha*CKSV1;
    /* Dingwen */

    ierr  = KSP_PCApplyBAorAB(ksp,S,T,R);CHKERRQ(ierr); /*   t <- K s    */
    /* Dingwen */
    /* Update checksum(T) */
    ierr = VecDot(C1,T,&CKST1);CHKERRQ(ierr);
    /* Dingwen */

    ierr  = VecDotNorm2(S,T,&d1,&d2);CHKERRQ(ierr);
    if (d2 == 0.0) {
      /* t is 0.  if s is 0, then alpha v == r, and hence alpha p
         may be our solution.  Give it a try? */
      ierr = VecDot(S,S,&d1);CHKERRQ(ierr);
      if (d1 != 0.0) {
        ksp->reason = KSP_DIVERGED_BREAKDOWN;
        break;
      }
      ierr = VecAXPY(X,alpha,P);CHKERRQ(ierr);   /*   x <- x + a p       */
      /* Dingwen */
      /* Update checksum(X) */
      CKSX1 = CKSX1 + alpha*CKSP1;
      /* Dingwen */

      ierr = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
      ksp->its++;
      ksp->rnorm  = 0.0;
      ksp->reason = KSP_CONVERGED_RTOL;
      ierr = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
      ierr = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
      ierr = KSPMonitor(ksp,i+1,0.0);CHKERRQ(ierr);
      break;
    }
    omega = d1 / d2;                               /*   w <- (t's) / (t't) */
    ierr  = VecAXPBYPCZ(X,alpha,omega,1.0,P,S);CHKERRQ(ierr); /* x <- alpha * p + omega * s + x */
    /* Dingwen */
    /* Update checksum(X) */
    CKSX1 = alpha*CKSP1 + omega*CKSS1 + CKSX1;
    /* Dingwen */

    ierr  = VecWAXPY(R,-omega,T,S);CHKERRQ(ierr);     /*   r <- s - w t       */
    /* Dingwen */
    /* Update checksum(R) */
    CKSR1 = CKSS1 - omega*CKST1;
    /* Dingwen */

    if (ksp->normtype != KSP_NORM_NONE && ksp->chknorm < i+2) {
      ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);
    }

    rhoold   = rho;
    omegaold = omega;

    ierr = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
    ksp->its++;
    ksp->rnorm = dp;
    ierr = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
    ierr = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
    ierr = KSPMonitor(ksp,i+1,dp);CHKERRQ(ierr);
    ierr = (*ksp->converged)(ksp,i+1,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;
    if (rho == 0.0) {
      ksp->reason = KSP_DIVERGED_BREAKDOWN;
      break;
    }
    i++;

	/* Dingwen */
	/* Inject error */
	if ((i==50) && (flag))
	{
		pos		= 1000;
		v	 	= 10000000;
		ierr	= VecSetValues(X,1,&pos,&v,INSERT_VALUES);CHKERRQ(ierr);
		ierr	= VecAssemblyBegin(X);CHKERRQ(ierr);
		ierr	= VecAssemblyEnd(X);CHKERRQ(ierr);
		flag	= PETSC_FALSE;
		PetscPrintf(MPI_COMM_WORLD,"Inject an error at the end of iteration-%d\n", i);
	}
	/* Dingwen */

  } while (i<ksp->max_it);

  /* Dingwen */
//  PetscScalar	sumX1,sumR1;
//  ierr = VecDot(C1,X,&sumX1);CHKERRQ(ierr);
//  ierr = VecDot(C1,R,&sumR1);CHKERRQ(ierr);
//  PetscPrintf(MPI_COMM_WORLD,"checksum1(X) = %lf\n"	, CKSX1);
//  PetscPrintf(MPI_COMM_WORLD,"sum1(X) = %lf\n", sumX1);
//  PetscPrintf(MPI_COMM_WORLD,"checksum1(R) = %lf\n", CKSR1);
//  PetscPrintf(MPI_COMM_WORLD,"sum1(R) = %lf\n", sumR1);
  /* Dingwen */

  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;

  ierr = KSPUnwindPreconditioner(ksp,X,T);CHKERRQ(ierr);
  if (bcgs->guess) {
    ierr = VecAXPY(X,1.0,bcgs->guess);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPBuildSolution_BCGS"
PetscErrorCode KSPBuildSolution_BCGS(KSP ksp,Vec v,Vec *V)
{
  PetscErrorCode ierr;
  KSP_BCGS       *bcgs = (KSP_BCGS*)ksp->data;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_RIGHT) {
    if (v) {
      ierr = KSP_PCApply(ksp,ksp->vec_sol,v);CHKERRQ(ierr);
      if (bcgs->guess) {
        ierr = VecAXPY(v,1.0,bcgs->guess);CHKERRQ(ierr);
      }
      *V = v;
    } else SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Not working with right preconditioner");
  } else {
    if (v) {
      ierr = VecCopy(ksp->vec_sol,v);CHKERRQ(ierr); *V = v;
    } else *V = ksp->vec_sol;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPReset_BCGS"
PetscErrorCode KSPReset_BCGS(KSP ksp)
{
  KSP_BCGS       *cg = (KSP_BCGS*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&cg->guess);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDestroy_BCGS"
PetscErrorCode KSPDestroy_BCGS(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPReset_BCGS(ksp);CHKERRQ(ierr);
  ierr = KSPDestroyDefault(ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     KSPBCGS - Implements the BiCGStab (Stabilized version of BiConjugate Gradient Squared) method.

   Options Database Keys:
.   see KSPSolve()

   Level: beginner

   Notes: See KSPBCGSL for additional stabilization
          Supports left and right preconditioning but not symmetric

   References: van der Vorst, SIAM J. Sci. Stat. Comput., 1992.

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPBICG, KSPBCGSL, KSPFBICG, KSPSetPCSide()
M*/
#undef __FUNCT__
#define __FUNCT__ "KSPCreate_BCGS"
PETSC_EXTERN PetscErrorCode KSPCreate_BCGS(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_BCGS       *bcgs;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,&bcgs);CHKERRQ(ierr);

  ksp->data                = bcgs;
  ksp->ops->setup          = KSPSetUp_BCGS;
  ksp->ops->solve          = KSPSolve_BCGS;
  ksp->ops->destroy        = KSPDestroy_BCGS;
  ksp->ops->reset          = KSPReset_BCGS;
  ksp->ops->buildsolution  = KSPBuildSolution_BCGS;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = KSPSetFromOptions_BCGS;

  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
