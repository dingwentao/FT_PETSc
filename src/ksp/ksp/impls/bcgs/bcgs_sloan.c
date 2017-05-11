
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
	ierr = KSPSetWorkVecs(ksp,14);CHKERRQ(ierr);
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
	Vec			   Xr,Vr,Pr,Rr,Tr,Sr;
	PetscReal      dp    = 0.0,d2;
	KSP_BCGS       *bcgs = (KSP_BCGS*)ksp->data;
	/* Dingwen */
	Mat			Amat;
	Vec			CKSAmat;
	Vec			C1;
	PetscScalar	CKSV,sumV;
	PetscBool	flag = PETSC_FALSE;
	PetscScalar	v;
	PetscInt	pos;
	PetscInt	error_type;
	PetscInt	itv_c, itv_d;
	PetscInt	inj_itr;
	/* Dingwen */


	PetscFunctionBegin;
	/* Dingwen */
	error_type 	= ksp->error_type;
	itv_c 		= ksp->itv_c;
	itv_d 		= ksp->itv_d;
	inj_itr 	= ksp->inj_itr;
	/* Dingwen */

	X  = ksp->vec_sol;
	B  = ksp->vec_rhs;
	R  = ksp->work[0];
	RP = ksp->work[1];
	V  = ksp->work[2];
	T  = ksp->work[3];
	S  = ksp->work[4];
	P  = ksp->work[5];
	/* Dingwen */
	Xr = ksp->work[6];
	Rr = ksp->work[7];
	Vr = ksp->work[8];
	Tr = ksp->work[9];
	Sr = ksp->work[10];
	Pr = ksp->work[11];
	CKSAmat		= ksp->work[12];
	C1			= ksp->work[13];
	/* Dingwen */

	/* Dingwen */
	/* checksum coefficients initialization */
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
	ierr = KSPGetOperators(ksp,&Amat,NULL);
	ierr = KSP_MatMultTranspose(ksp,Amat,C1,CKSAmat);CHKERRQ(ierr);				/* Compute the initial checksum(A) */
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

	i=0;
	if (error_type == 1) flag = PETSC_TRUE;
	do {
		/* Dingwen */
		if ((i>0) && (i%(itv_c*itv_d) == 0) && (error_type==2)) {
			flag = PETSC_TRUE;
			inj_itr += itv_c*itv_d;
		}

		if (error_type==3) {
			flag = PETSC_TRUE;
			inj_itr = i;
		}

		ierr = VecDot(R,RP,&rho);CHKERRQ(ierr);       /*   rho <- (r,rp)      */
		beta = (rho/rhoold) * (alpha/omegaold);
		/* Dingwen */
		ierr = VecCopy(P,Pr);CHKERRQ(ierr);
		ierr = VecAXPBYPCZ(P,1.0,-omegaold*beta,beta,R,V);CHKERRQ(ierr);  /* p <- r - omega * beta* v + beta * p */
		ierr = VecAXPBYPCZ(Pr,1.0,-omegaold*beta,beta,R,V);CHKERRQ(ierr);	/* Redundancy */
		/* Dingwen */

		/* Dingwen */
		ierr = KSP_PCApplyBAorAB(ksp,P,V,T);CHKERRQ(ierr);  /*   v <- K p           */
		ierr = KSP_PCApplyBAorAB(ksp,Pr,Vr,Tr);CHKERRQ(ierr);	/* Redundancy */
		/* Dingwen */

		ierr = VecDot(CKSAmat,P,&CKSV);CHKERRQ(ierr);
		ierr = VecDot(C1,V,&sumV);CHKERRQ(ierr);

		/* Inject an error in V */
		if ((i==inj_itr) && (flag))
		{
			VecScatter	ctx;
			Vec			V_SEQ;
			PetscScalar	*V_ARR;
			VecScatterCreateToAll(V,&ctx,&V_SEQ);
			VecScatterBegin(ctx,V,V_SEQ,INSERT_VALUES,SCATTER_FORWARD);
			VecScatterEnd(ctx,V,V_SEQ,INSERT_VALUES,SCATTER_FORWARD);
			VecGetArray(V_SEQ,&V_ARR);
			pos = 0;
			v		= V_ARR[pos]*1.0;
			ierr	= VecSetValues(V,1,&pos,&v,INSERT_VALUES);CHKERRQ(ierr);
			VecDestroy(&V_SEQ);
			VecScatterDestroy(&ctx);
			VecAssemblyBegin(V);
			VecAssemblyEnd(V);
			if (error_type!=3) PetscPrintf(MPI_COMM_WORLD,"Inject an error in V after MVM at iteration-%d\n",i);
			flag	= PETSC_FALSE;

			VecScatterCreateToAll(V,&ctx,&V_SEQ);
			VecScatterBegin(ctx,V,V_SEQ,INSERT_VALUES,SCATTER_FORWARD);
			VecScatterEnd(ctx,V,V_SEQ,INSERT_VALUES,SCATTER_FORWARD);
			VecGetArray(V_SEQ,&V_ARR);
			pos = 0;
			v		= V_ARR[pos]/1.0;
			ierr = VecDot(C1,V,&sumV);CHKERRQ(ierr);
			ierr = VecDot(C1,V,&sumV);CHKERRQ(ierr);
			ierr = VecDot(C1,V,&sumV);CHKERRQ(ierr);
			ierr = VecDot(C1,V,&sumV);CHKERRQ(ierr);
			ierr = VecDot(C1,V,&sumV);CHKERRQ(ierr);
			ierr	= VecSetValues(V,1,&pos,&v,INSERT_VALUES);CHKERRQ(ierr);
			VecDestroy(&V_SEQ);
			VecScatterDestroy(&ctx);
			VecAssemblyBegin(V);
			VecAssemblyEnd(V);
		}

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
		ierr  = VecWAXPY(Sr,-alpha,V,R);CHKERRQ(ierr);		/* Redundancy */
		/* Dingwen */

		/* Dingwen */
		ierr  = VecCopy(S,Sr);CHKERRQ(ierr);
		ierr  = KSP_PCApplyBAorAB(ksp,S,T,R);CHKERRQ(ierr); /*   t <- K s    */
		ierr  = KSP_PCApplyBAorAB(ksp,Sr,Tr,Rr);CHKERRQ(ierr);		/* Redundancy */
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
			/* Dingwen */
			ierr = VecCopy(X,Xr);CHKERRQ(ierr);
			ierr = VecAXPY(X,alpha,P);CHKERRQ(ierr);   /*   x <- x + a p       */
			ierr = VecAXPY(Xr,alpha,P);CHKERRQ(ierr); /* Redundancy */
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
		/* Dingwen */
		ierr  = VecCopy(X,Xr);CHKERRQ(ierr);
		ierr  = VecAXPBYPCZ(X,alpha,omega,1.0,P,S);CHKERRQ(ierr); /* x <- alpha * p + omega * s + x */
		ierr  = VecAXPBYPCZ(Xr,alpha,omega,1.0,P,S);CHKERRQ(ierr);		/* Redundancy */
		ierr  = VecWAXPY(R,-omega,T,S);CHKERRQ(ierr);     /*   r <- s - w t       */
		ierr  = VecWAXPY(Rr,-omega,Tr,Sr);CHKERRQ(ierr);	/* Redundancy */
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
	} while (i<ksp->max_it);

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
