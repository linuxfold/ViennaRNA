#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include <stdlib.h>
#include <string.h>

#ifdef HAVE_DLFCN_H
# include <dlfcn.h>
#endif

#ifndef HAVE_CONFIG_H
# include "ViennaRNA/vrna_config.h"
#endif

#if VRNA_WITH_PTHREADS
# include <pthread.h>
#endif

#include "ViennaRNA/fold_compound.h"
#include "ViennaRNA/gpu/backend.h"
#include "ViennaRNA/partfunc/global.h"
#include "ViennaRNA/utils/basic.h"


static int
use_cuda_backend(void)
{
  const char *value = getenv("VRNA_PF_BACKEND");
  return (!value) || (strcmp(value, "cpu") != 0);
}


#ifdef HAVE_DLFCN_H
static void                 *cuda_handle = NULL;
static vrna_cuda_pf_batch_f cuda_pf_batch = NULL;


static void
initialize_cuda_backend(void)
{
  const char              *override = getenv("VRNA_CUDA_LIBRARY");
  vrna_cuda_backend_abi_f abi;

  cuda_handle = override && override[0] ?
                dlopen(override, RTLD_NOW | RTLD_LOCAL) :
                dlopen("libRNA_cuda.so", RTLD_NOW | RTLD_LOCAL);

# ifdef __APPLE__
  if (!cuda_handle)
    cuda_handle = dlopen("libRNA_cuda.dylib", RTLD_NOW | RTLD_LOCAL);
# endif

  if (!cuda_handle)
    return;

  abi           = (vrna_cuda_backend_abi_f)dlsym(cuda_handle, VRNA_CUDA_BACKEND_ABI_SYMBOL);
  cuda_pf_batch = (vrna_cuda_pf_batch_f)dlsym(cuda_handle, VRNA_CUDA_PF_BATCH_SYMBOL);
  if ((!abi) || (!cuda_pf_batch) || (abi() != VRNA_CUDA_BACKEND_ABI_VERSION)) {
    dlclose(cuda_handle);
    cuda_handle   = NULL;
    cuda_pf_batch = NULL;
  }
}


static int
run_cuda_backend(vrna_fold_compound_t **fc,
                 size_t               count,
                 unsigned char        *handled,
                 float                *energies,
                 unsigned int         flags)
{
# if VRNA_WITH_PTHREADS
  static pthread_once_t once = PTHREAD_ONCE_INIT;
  (void)pthread_once(&once, initialize_cuda_backend);
# else
  static int initialized = 0;
  if (!initialized) {
    initialize_cuda_backend();
    initialized = 1;
  }
# endif

  return cuda_pf_batch ? cuda_pf_batch(fc, count, handled, energies, flags) : 0;
}
#endif


PUBLIC int
vrna_pf_batch(vrna_fold_compound_t **fc,
              size_t               count,
              unsigned int         flags,
              float                *ensemble_energies)
{
  unsigned char *handled;

  if ((count > 0) && ((!fc) || (!ensemble_energies)))
    return 0;

  if (count == 0)
    return 1;

  handled = (unsigned char *)vrna_alloc(sizeof(*handled) * count);

  if (use_cuda_backend()) {
#ifdef _OPENMP
# pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < count; i++)
      if (fc[i])
        (void)vrna_fold_compound_prepare(fc[i], VRNA_OPTION_PF);

#ifdef HAVE_DLFCN_H
    (void)run_cuda_backend(fc, count, handled, ensemble_energies, flags);
#endif
  }

#ifdef _OPENMP
# pragma omp parallel for schedule(dynamic)
#endif
  for (size_t i = 0; i < count; i++)
    if (!handled[i])
      ensemble_energies[i] = fc[i] ? (float)vrna_pf(fc[i], NULL) : (float)(INF / 100.);

  free(handled);
  return 1;
}
