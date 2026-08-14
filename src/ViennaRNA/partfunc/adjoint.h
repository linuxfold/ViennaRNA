#ifndef VIENNA_RNA_PACKAGE_PARTFUNC_ADJOINT_H
#define VIENNA_RNA_PACKAGE_PARTFUNC_ADJOINT_H

#include <ViennaRNA/datastructures/basic.h>
#include <ViennaRNA/fold_compound.h>

/**
 *  @file     ViennaRNA/partfunc/adjoint.h
 *  @brief    Development oracle for partition-function adjoints
 *
 *  This interface is intentionally limited to short, unconstrained, single
 *  strands in the default dangles=2 model. It is a correctness oracle for
 *  development of the inside/outside GPU recurrence, not a production
 *  probability implementation.
 */

/**
 *  @brief Compute an independent CPU oracle for the adjoints of @c qb cells
 *
 *  The forward partition function must already be present in @p fc. For each
 *  nonzero paired inside cell @f$B_{ij}@f$, the oracle applies a known
 *  multiplicative probe @f$w@f$ to that pair and recomputes the root inside
 *  partition function. Since a base pair occurs at most once in a structure,
 *  the perturbed root is exactly linear in @f$w@f$:
 *
 *  @f[
 *    Q(w) = Q(1) + (w - 1) B_{ij} \bar B_{ij}.
 *  @f]
 *
 *  The returned triangle therefore contains @f$\bar B_{ij}@f$ without
 *  consulting the fold compound's base-pair probability matrix. Consequently
 *  @f$p_{ij} = B_{ij}\bar B_{ij}@f$ can be checked independently against
 *  vrna_pairing_probs().
 *
 *  The repeated scalar inside calculations make this routine deliberately
 *  expensive. Inputs longer than 60 nucleotides and model features outside
 *  the initial CUDA PF eligibility envelope are rejected.
 *
 *  @param  fc  Fold compound with completed forward PF matrices
 *  @return     Newly allocated triangular @c qb-adjoint matrix, or @c NULL if
 *              the input is unsupported or a probe calculation fails. The
 *              caller must release the result with free().
 */
FLT_OR_DBL *
vrna_pf_adjoint_oracle(vrna_fold_compound_t *fc);

#endif
