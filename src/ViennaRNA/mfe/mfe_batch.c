#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include <stdlib.h>
#include <string.h>

#ifdef HAVE_DLFCN_H
# include <dlfcn.h>
#endif

#include "ViennaRNA/vrna_config.h"

#if VRNA_WITH_PTHREADS
# include <pthread.h>
#endif

#include "ViennaRNA/fold_compound.h"
#include "ViennaRNA/mfe/global.h"
#include "ViennaRNA/backtrack/global.h"
#include "ViennaRNA/utils/basic.h"
#include "ViennaRNA/gpu/backend.h"


static vrna_backend_e
backend_from_environment(void)
{
  const char *value = getenv("VRNA_MFE_BACKEND");

  if (value) {
    if (strcmp(value, "cpu") == 0)
      return VRNA_BACKEND_CPU;

    if (strcmp(value, "cuda") == 0)
      return VRNA_BACKEND_CUDA;
  }

  return VRNA_BACKEND_AUTO;
}


#ifdef HAVE_DLFCN_H
static void                        *cuda_handle = NULL;
static vrna_cuda_backend_batch_f   cuda_batch = NULL;


static void *
open_cuda_backend(void)
{
  const char  *override = getenv("VRNA_CUDA_LIBRARY");
  void        *handle;

  if (override && override[0])
    return dlopen(override, RTLD_NOW | RTLD_LOCAL);

  handle = dlopen("libRNA_cuda.so", RTLD_NOW | RTLD_LOCAL);

# ifdef __APPLE__
  if (!handle)
    handle = dlopen("libRNA_cuda.dylib", RTLD_NOW | RTLD_LOCAL);
# endif

  return handle;
}


static void
initialize_cuda_backend(void)
{
  vrna_cuda_backend_abi_f abi;

  cuda_handle = open_cuda_backend();
  if (!cuda_handle)
    return;

  abi        = (vrna_cuda_backend_abi_f)dlsym(cuda_handle, VRNA_CUDA_BACKEND_ABI_SYMBOL);
  cuda_batch = (vrna_cuda_backend_batch_f)dlsym(cuda_handle, VRNA_CUDA_BACKEND_BATCH_SYMBOL);

  if ((!abi) ||
      (!cuda_batch) ||
      (abi() != VRNA_CUDA_BACKEND_ABI_VERSION)) {
    dlclose(cuda_handle);
    cuda_handle = NULL;
    cuda_batch  = NULL;
  }
}


static int
run_cuda_backend(vrna_fold_compound_t **fc,
                 size_t               count,
                 unsigned char        *handled,
                 int                  *energies,
                 unsigned int         flags)
{
#if VRNA_WITH_PTHREADS
  static pthread_once_t once = PTHREAD_ONCE_INIT;
  (void)pthread_once(&once, initialize_cuda_backend);
#else
  static int initialized = 0;
  if (!initialized) {
    initialize_cuda_backend();
    initialized = 1;
  }
#endif

  return cuda_batch ? cuda_batch(fc, count, handled, energies, flags) : 0;
}
#endif


PUBLIC int
vrna_mfe_batch(vrna_fold_compound_t **fc,
               size_t               count,
               char                 **structures,
               float                *energies)
{
  unsigned char   *handled;
  int             *cuda_energies;
  vrna_backend_e  backend;

  if (((count > 0) && (!fc || !energies)))
    return 0;

  if (count == 0)
    return 1;

  handled      = (unsigned char *)vrna_alloc(sizeof(unsigned char) * count);
  cuda_energies = (int *)vrna_alloc(sizeof(int) * count);
  backend      = backend_from_environment();

  if (backend != VRNA_BACKEND_CPU) {
    size_t prepared = 0;
    unsigned int flags = 0;

    if (structures)
      for (size_t i = 0; i < count; i++)
        if (structures[i] && fc[i] && fc[i]->params->model_details.backtrack)
          flags |= VRNA_CUDA_BACKEND_COPY_MATRICES;

#ifdef _OPENMP
# pragma omp parallel for reduction(+:prepared) schedule(static)
#endif
    for (size_t i = 0; i < count; i++)
      if (fc[i] && vrna_fold_compound_prepare(fc[i], VRNA_OPTION_MFE))
        prepared++;

#ifdef HAVE_DLFCN_H
    if (prepared > 0)
      (void)run_cuda_backend(fc, count, handled, cuda_energies, flags);
#else
    (void)prepared;
#endif
  }

#ifdef _OPENMP
# pragma omp parallel for schedule(dynamic)
#endif
  for (size_t i = 0; i < count; i++) {
    char *structure = structures ? structures[i] : NULL;

    if (handled[i]) {
      if (structure && fc[i]->params->model_details.backtrack)
        energies[i] = vrna_backtrack5(fc[i], fc[i]->length, structure);
      else
        energies[i] = (float)cuda_energies[i] / 100.;
    } else {
      energies[i] = vrna_mfe(fc[i], structure);
    }
  }

  free(cuda_energies);
  free(handled);

  return 1;
}
