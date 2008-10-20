package linear;

public enum SolverType {
   /** L2-regularized logistic regression */
   L2_LR,

   /**	L2-loss support vector machines (dual) */
   L2LOSS_SVM_DUAL,

   /** L2-loss support vector machines (primal) */
   L2LOSS_SVM,

   /** L1-loss support vector machines (dual) */
   L1LOSS_SVM_DUAL,

   /** multi-class support vector machines by Crammer and Singer */
   MCSVM_CS;
}
