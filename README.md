# Exact CUDA MFE acceleration

> [!IMPORTANT]
> **Experimental CUDA fork.** This branch adds an optional exact CUDA backend
> for batched MFE folding on NVIDIA GPUs. It is independent work and is not an
> official ViennaRNA release. Unsupported models and constraints, unavailable
> GPUs, and compact-energy range failures fall back to the original CPU path.
> See [the CUDA development guide](docs/cuda-development.md) for build steps,
> validation results, limitations, prior work, and the AI-development
> disclosure.

## Latest measured performance: up to 49.46× faster

Across seven measured iterations of a GPU-saturating batch of 256 public 900-nt
EternaFold sequences, the exact CUDA backend delivered **49.46× higher
energy-only throughput** and **32.66× higher structure-output throughput** on
an RTX PRO 6000 Blackwell.

| Backend | Energy-only | Speedup | Structures | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 32-thread CPU | 5.313528 s | 1× | 5.305571 s | 1× |
| RTX PRO 6000 Blackwell | 0.107429 s | **49.46×** | 0.162433 s | **32.66×** |

The 100× target has not been met. On this fixed CPU baseline it requires at
most 53.135 ms for energy-only output and 53.056 ms for structures, leaving a
further **2.02×** and **3.06×** reduction, respectively.

The benchmark uses the first 256 qualifying records from EternaFold's pinned
[`ExternalData_window900_uniq.fasta`](https://github.com/WaymentSteeleLab/EternaFold/blob/87b9aac55cee14fd562049d08f7b92d3131f10ce/datasets_in_fasta_form/test_datasets/ExternalData_window900_uniq.fasta).
Fold-compound construction is outside the timed region and one warm-up is
excluded. The CPU reference used all 32 hardware threads of a 16-core AMD Ryzen
Threadripper PRO 9955WX; each GPU was measured alone with CUDA 13.1. Energy and
structure checksums matched the CPU exactly. These are throughput results for
this workload, not a universal speedup or a single-sequence latency claim.

For comparison, commit
[`37e350fd`](https://github.com/linuxfold/ViennaRNA/tree/37e350fd080942dad391b1368e600c3b1dea76e8)
previously measured 27.51× energy-only and 21.38× structure throughput on an
RTX 4090. The table above was rerun for the current Blackwell changes.
See the [CUDA development guide](docs/cuda-development.md) for reproduction
commands, the dataset SHA-256, exactness tests, eligibility limits, and
fallback behavior.

## What changed

| Area | Major change | Exactness and safety |
| --- | --- | --- |
| Optional batch API | Added `vrna_mfe_batch()` and a runtime-loaded `libRNA_cuda.so` backend selected with `VRNA_MFE_BACKEND=auto\|cpu\|cuda`. Ordinary RNAlib remains CPU-only and has no mandatory CUDA dependency. | Inputs outside the documented model and constraint envelope stay on the authoritative CPU implementation. |
| Sparse multibranch folding | Replaced the long multibranch split scan with an exact candidate-sparse recurrence, retained only the two `M2` spans that paired cells can consume, and kept the sparse recurrence in normalized residual form. | A CPU oracle validates the sparse recurrence. Candidate-capacity overflow is detected per input and recomputed on the CPU. |
| Paired/internal-loop engine | Added exact pair bitsets, pairable-cell compaction, an exact lower bound for every legal `(u1,u2)` loop shape, cached outer-loop context, and deferred pair-type lookup until after the bound test. | The bound minimizes over every admitted pair type and nucleotide context, so it can only prune candidates that cannot improve the current exact integer minimum. The oracle measured a 79.8% reduction in full internal-energy evaluations on its validation workload. |
| GPU-specific state layout | Stored dynamic-programming energies as range-checked signed 16-bit residuals. Blackwell uses a packed span-major upper triangle, derived pair types, and bounded 32-bit packed indexing; Ada uses the faster measured dense-square layout plus three compact pair-type bitplanes. | Every compact store is range-checked. Any representability failure causes exact CPU recomputation rather than saturation or approximation. |
| Traceback and transfers | Added exact device traceback. Energy-only calls return only final energies; structure calls return energies and dot-bracket strings instead of exporting complete DP matrices. | Traceback follows CPU decision and tie-breaking order. Failure to reproduce a decision triggers CPU fallback. |
| Initialization and launch path | Precomputed exact hairpin-size penalties, removed redundant full-matrix initialization, cached hot mismatch terms, used a stream-ordered device arena, and tuned paired and sparse lane widths separately for Ada and Blackwell workloads. | Major feature groups expose diagnostic overrides, and architecture-specific defaults were retained only after controlled measurements. |
| Validation and profiling | Added cell-by-cell CUDA/CPU comparisons, sparse and internal-loop CPU oracles, model and constraint fallback tests, forced overflow tests, public FASTA benchmarking, and detailed recurrence/phase counters. | Tests compare exact `c`, `fML`, and `f5` cells, integer energies, and byte-identical structures on both supported GPU architectures. |

The thermodynamic parameter tables, integer energy representation, loop-size
limit, and CPU recurrences were not approximated. There is no fast-math, beam,
Tensor Core, reduced-precision energy, or saturating path.

[![GitHub release](https://img.shields.io/github/release/ViennaRNA/ViennaRNA.svg)](https://www.tbi.univie.ac.at/RNA/#download)
[![Build Status](https://github.com/ViennaRNA/ViennaRNA/actions/workflows/release.yaml/badge.svg)](https://github.com/ViennaRNA/ViennaRNA/actions)
[![Github All Releases](https://img.shields.io/github/downloads/ViennaRNA/ViennaRNA/total.svg)](https://github.com/ViennaRNA/ViennaRNA/releases)
[![Conda](https://img.shields.io/conda/v/bioconda/viennarna.svg)](https://anaconda.org/bioconda/viennarna)
[![Conda Downloads](https://img.shields.io/conda/dn/bioconda/viennarna.svg)](https://anaconda.org/bioconda/viennarna)
[![AUR](https://img.shields.io/aur/version/viennarna.svg)](https://aur.archlinux.org/packages/viennarna/)

# ViennaRNA Package

A C code library and several stand-alone programs for the prediction and comparison of RNA secondary structures.

Amongst other things, our implementations allow you to:

- predict minimum free energy secondary structures
- calculate the partition function for the ensemble of structures
- compute various equilibrium probabilities
- calculate suboptimal structures in a given energy range
- compute local structures in long sequences
- predict consensus secondary structures from a multiple sequence alignment
- predict melting curves
- search for sequences folding into a given structure
- compare two secondary structures 
- predict interactions between multiple RNA molecules

The package includes `Perl 5` and `Python` modules that give access to almost
all functions of the C library from within the respective scripting languages.

There is also a set of programs for analyzing sequence and distance
data using split decomposition, statistical geometry, and cluster methods.
They are not maintained any more and not built by default.

The code very rarely uses static arrays, and all programs should work for 
sequences up to a length of 32,700 (if you have huge amounts of memory that
is).

See the [NEWS][file_news] and [CHANGELOG.md][file_changelog] files for changes between versions.

----

## Table of Contents
1. [Availability](#availability)
2. [Documentation](#documentation)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Executable Programs](#executable-programs)
6. [Energy Parameters](#energy-parameters)
7. [References](#references)
8. [License](#license)
9. [Contact](#contact)

----

## Availability

The most recent source code should always be available through the
[official ViennaRNA website][vrna_website] and through
[github][vrna_github].

----

## Documentation

Executable programs shipped with the ViennaRNA Package are documented by
corresponding man pages, use e.g.:
```
man RNAfold
```
in a UNIX terminal to obtain the documentation for the `RNAfold` program.
HTML translations of all man pages can be found at [our official homepage][vrna_website_manpages].

We maintain a reference manual describing the `RNAlib` API that is automatically
generated with [doxygen][doxygen_website]. The HTML version of this reference
manual is available at [our official website][vrna_refman_website] and at
[ReadTheDocs][vrna_rtfm].

----

## Installation

For best portability the ViennaRNA package uses the GNU autoconf and automake
tools. The instructions below are for installing the ViennaRNA package from
source.

*See the file [INSTALL][file_install] for a more detailed description of the build
and installation process.*

### Quick Start

Usually you'll simply unpack the distribution tarball, configure and make:
```
tar -zxvf ViennaRNA-2.7.2.tar.gz
cd ViennaRNA-2.7.2
./configure
make
sudo make install
```

### User-dir Installation
If you do not have root privileges on your computer, you might want to install
the ViennaRNA Package to a location where you actually have write access to.
Use the `--prefix` option to set the installation prefix like so:
```
./configure --prefix=/home/username/ViennaRNA
make install
```

This will install everything into a new directory `ViennaRNA` directly into
the home directory of user `username`.

*Note, that the actual install destination paths are listed at the end
of the `./configure` output.*

### Install from git repository

If you attempt to build and install from our git repository, you need to
perform some additional steps __before__ actually running the `./configure`
script:

1. Unpack the `libsvm` archive to allow for SVM Z-score regression with the
   program `RNALfold`:
```
cd src
tar -xzf libsvm-3.35.tar.gz
cd ..
```

2. Unpack the `dlib` archive to allow for concentration dependency computations with the
   program `RNAmultifold`:
```
cd src
tar -xjf dlib-20.0.tar.bz2
cd ..
```

3. Install the autotools toolchain and the additional maintainer tools
   `gengetopt`, `help2man`,`flex`,`xxd`, and `swig` if necessary. For
   instance, in Debian based distributions, the following packages need
   to be installed:
    - `build-essential` (basic build tools, such as compiler, linker, etc.)
    - `autoconf`, `automake`, `libtool`, `pkg-config` (autotools toolchain)
    - `gengetopt` (to generate command line parameter parsers)
    - `help2man` (to generate the man pages) 
    - `bison` and `flex`` (to generate sources for RNAforester)
    - `vim-common` (for the `xxd` program)
    - `swig` (to generate the scripting language interfaces)
    - `liblapacke` (for `RNAxplorer`)
    - `liblapack`  (for `RNAxplorer`)
    - A fortran compiler, e.g. `gfortran` (for `RNAxplorer`)

4. Finally, run the autoconf/automake toolchain:
```
autoreconf -i
```

After that, you can compile and install the ViennaRNA Package as if obtained
from the distribution tarball.


### Binary packages
Binary packages for several Linux-based platforms, Microsoft Windows, and
Mac OS X are available at [our official website][vrna_website_packages].

### Bioconda
Installation is also possible through [bioconda][bioconda_website].
After successfully setting up the bioconda channels
```
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
conda config --set channel_priority strict
```
you can install the [viennarna bioconda package][vrna_bioconda_website]
through
```
conda install viennarna
```


### Python interface only
The Python 3 interface for the ViennaRNA Package library is
[available at PyPI][vrna_pypi_website] and can
be installed independently using Python's `pip`:
```
python -m pip install viennarna
```

#### Building a Python 3 sdist or wheel package
Our source tree allows for building/installing the Python 3
interface separately. For that, we provide the necessary packaging
files `pyproject.toml`, `setup.cfg`, `setup.py` and `MANIFEST.in`.

These files are created by our `autoconf` toolchain after a run
of `./configure`. Particular default compile-time features may be
(de-)activated by setting the corresponding boolean flags in
`setup.cfg`. See below for additional steps when building the
Python interface from a clean git clone.

Running
```
python -m build
```
will then create a source distribution (`sdist`) and a binary
package (`wheel`) in the `dist/` directory. These files can be
easily installed via Python's `pip`.

#### Howto prepare the Python 3 sdist/wheel build from git repository
If you are about to create the Python interface from a fresh
clone of our git repository, you require additional steps after
running `./configure` as described above. In particular, some
autogenerated static files that are compiled into RNAlib must
be generated. To do so, run
```
cd src/ViennaRNA/static
make
cd ../../..
```

Additionally, if building the reference manual is not explicitly
turned off, the Python interface requires docstrings to be generated.
They are taken from the doxygen xml output which can be created by
```
cd doc
make refman-html
cd ..
```

Finally, the swig wrapper must be build using
```
cd interfaces/Python
make RNA/RNA.py
cd ../..
```

After these steps, the Python sdist and wheel packages can
be build as usual.

----

## Configuration

This release includes the `RNAforester`, `Kinfold`, `Kinwalker`, `RNAlocmin`,
and `RNAxplorer` programs, which can also be obtained as independent packages.
Running `./configure` in the ViennaRNA directory will configure these packages
as well.
However, for detailed information and compile time options, see the README and
INSTALL files in the respective subdirectories.

A comprehensive description of configure options is available at our
[reference manual][vrna_refman_config].

See also
```
./configure --help
```
for a complete list of all `./configure` options and important environment
variables.

----

## Executable Programs

The ViennaRNA Package includes the following executable programs:

| Program         | Description                                                                                                               |
| --------------- | :-------------------------------------------------------------------------------------------------------------------------|
| `RNA2Dfold`     | Compute MFE structure, partition function and representative sample structures of k,l neighborhoods                       |
| `RNAaliduplex`  | Predict conserved RNA-RNA interactions between two alignments                                                             |
| `RNAalifold`    | Calculate secondary structures for a set of aligned RNA sequences                                                         |
| `RNAcofold`     | Calculate secondary structures of two RNAs with dimerization                                                              |
| `RNAdistance`   | Calculate distances between RNA secondary structures                                                                      |
| `RNAdos`        | Compute the density of states for the conformation space of a given RNA sequence                                          |
| `RNAduplex`     | Compute the structure upon hybridization of two RNA strands                                                               |
| `RNAeval`       | Evaluate free energy of RNA sequences with given secondary structure                                                      |
| `RNAfold`       | Calculate minimum free energy secondary structures and partition function of RNAs                                         |
| `RNAheat`       | Calculate the specific heat (melting curve) of an RNA sequence                                                            |
| `RNAinverse`    | Find RNA sequences with given secondary structure (sequence design)                                                       |
| `RNALalifold`   | Calculate locally stable secondary structures for a set of aligned RNAs                                                   |
| `RNALfold`      | Calculate locally stable secondary structures of long RNAs                                                                |
| `RNAmultifold`  | Compute secondary structures and probabilities for multiple interacting RNAs                                              |
| `RNApaln`       | RNA alignment based on sequence base pairing propensities                                                                 |
| `RNApdist`      | Calculate distances between thermodynamic RNA secondary structures ensembles                                              |
| `RNAparconv`    | Convert energy parameter files from ViennaRNA 1.8 to 2.0 format                                                           |
| `RNAPKplex`     | Predict RNA secondary structures including pseudoknots                                                                    |
| `RNAplex`       | Find targets of a query RNA                                                                                               |
| `RNAplfold`     | Calculate average pair probabilities for locally stable secondary structures                                              |
| `RNAplot`       | Draw RNA Secondary Structures in PostScript, SVG, or GML                                                                  |
| `RNApvmin`      | Calculate a perturbation vector that minimizes discripancies between predicted and observed pairing probabilities         |
| `RNAsnoop`      | Find targets of a query H/ACA snoRNA                                                                                      |
| `RNAsubopt`     | Calculate suboptimal secondary structures of RNAs                                                                         |
| `RNAup`         | Calculate the thermodynamics of RNA-RNA interactions                                                                      |
| `AnalyseSeqs`   | Analyse sequence data                                                                                                     |
| `AnalyseDists`  | Analyse distance matrices                                                                                                 |


A couple of useful utilities can be found in the src/Utils directory.

All executables read from stdin and write to stdout and use command line
switches rather than menus to be easily usable in pipes. For more detailed
information see the man pages. Perl utilities contain POD documentation
that can be read by typing e.g.
```
perldoc relplot.pl
```

Together with this version we also distribute the programs

  * `kinfold`,
  * `RNAforester`,
  * `RNAlocmin`,
  * `RNAxlorer`, and
  * `kinwalker`

See the README files in the respective sub-directories.

----

## References

If you use our software package, you may want to cite the follwing publications:

- [R. Lorenz et al. (2011)][vrna2_paper],
"ViennaRNA Package 2.0", Algorithms for Molecular Biology, 6:26

- [I.L. Hofacker (1994)][vrna_paper],
"Fast folding and comparison of RNA secondary structures",
Monatshefte fuer Chemie, Volume 125, Issue 2, pp 167-188


*Note, that the individual executable programs state their own list of references
in the corresponding man-pages.*

----

## Energy Parameters

### Default Parameters
Since version 2.0.0 the build-in energy parameters, also available as parameter
file [rna_turner2004.par][file_param_rna2004], are taken from:

- [D.H. Mathews et al. (2004)][param2004_paper],
"Incorporating chemical modification constraints into a dynamic programming
algorithm for prediction of RNA secondary structure",
Proc. Natl. Acad. Sci. USA: 101, pp 7287-7292

- [D.H. Turner et al. (2009)][nndb_paper], 
"NNDB: The nearest neighbor parameter database
for predicting stability of nucleic acid secondary structure",
Nucleic Acids Research: 38, pp 280-282.

### Deprecated Parameters
For backward compatibility we also provide energy parameters from Turner et al.
1999 in the file [rna_turner1999.par][file_param_rna1999].

### Trained Parameters
A set of trained RNA energy parameters from Andronescou et al. 2007,
[rna_andronescou2007.par][file_param_andronescu2007], a set of RNA
energy parameters obtained by graft and grow genetic programming from Langdon
et al. 2018, [rna_langdon2018.par][file_param_langdon2018] are
also included.

### DNA Parameters
To predict secondary structures for DNA, we additionally include two DNA
parameter sets:
  * [dna_mathews1999.par][file_param_dna1999] and
  * [dna_mathews2004.par][file_param_dna2004].


### RNA Base Modifications
Since version 2.6.0 several programs received support to predict structures
for sequences with modified bases. The corresponding energy parameters are
incomplete and mostly restricted to base pair stacking. The ViennaRNA package
currently includes paraneter sets for
  * [inosine][file_param_inosine]
  * [pseudouridine][file_param_pseudouridine]
  * [m1Ψ][file_param_m1Ψ]
  * [m5C][file_param_m5C]
  * [m6A][file_param_m6A]
  * [7DA][file_param_7DA]
  * [purine (a.k.a. nebularine)][file_param_purine]
  * [dihydrouridine][file_param_dihydrouridine]

### Paramers Set Availability
Energy parameter files are mostly provided for use with our executable
programs. All parameter sets are compiled-in to our `RNAlib` C-library.
Thus, when building upon our library, either through C/C++ or the
scripting language interface, these data are available through dedicated
functions and as constant strings. See, e.g.
[here][vrna_refman_epars] and [here][vrna_refman_mod_base].

----

## License

Please read the copyright notice in the file [COPYING][file_license]!

If you're a commercial user and find these programs useful, please consider
supporting further developments with a donation.

----

## Contact

We need your feedback! Send your comments, suggestions, and questions to
rna@tbi.univie.ac.at

Ivo Hofacker, Spring 2006


[file_news]: https://github.com/ViennaRNA/ViennaRNA/blob/master/NEWS
[file_changelog]: https://github.com/ViennaRNA/ViennaRNA/blob/master/CHANGELOG.md
[file_install]: https://github.com/ViennaRNA/ViennaRNA/blob/master/INSTALL
[file_license]: https://github.com/ViennaRNA/ViennaRNA/blob/master/COPYING
[file_param_rna2004]: https://github.com/ViennaRNA/ViennaRNA/blob/master/misc/rna_turner2004.par
[file_param_rna1999]: https://github.com/ViennaRNA/ViennaRNA/blob/master/misc/rna_turner1999.par
[file_param_dna2004]: https://github.com/ViennaRNA/ViennaRNA/blob/master/misc/dna_mathews2004.par
[file_param_dna1999]: https://github.com/ViennaRNA/ViennaRNA/blob/master/misc/dna_mathews1999.par
[file_param_andronescu2007]: https://github.com/ViennaRNA/ViennaRNA/blob/master/misc/rna_andronescou2007.par
[file_param_langdon2018]: https://github.com/ViennaRNA/ViennaRNA/blob/master/misc/rna_langdon2018.par
[file_param_inosine]: https://github.com/ViennaRNA/ViennaRNA/blob/master/misc/rna_mod_inosine_parameters.json
[file_param_pseudouridine]: https://github.com/ViennaRNA/ViennaRNA/blob/master/misc/rna_mod_pseudouridine_parameters.json
[file_param_m1Ψ]: https://github.com/ViennaRNA/ViennaRNA/blob/master/misc/rna_mod_n1methylpseudouridine_parameters.json
[file_param_m5C]: https://github.com/ViennaRNA/ViennaRNA/blob/master/misc/rna_mod_m5C_parameters.json
[file_param_m6A]: https://github.com/ViennaRNA/ViennaRNA/blob/master/misc/rna_mod_m6A_parameters.json
[file_param_7DA]: https://github.com/ViennaRNA/ViennaRNA/blob/master/misc/rna_mod_7DA_parameters.json
[file_param_purine]: https://github.com/ViennaRNA/ViennaRNA/blob/master/misc/rna_mod_purine_parameters.json
[file_param_dihydrouridine]: https://github.com/ViennaRNA/ViennaRNA/blob/master/misc/rna_mod_dihydrouridine_parameters.json
[vrna_website]: https://www.tbi.univie.ac.at/RNA
[vrna_github]: https://github.com/ViennaRNA/ViennaRNA
[vrna_website_manpages]: https://www.tbi.univie.ac.at/RNA/documentation.html#programs
[vrna_website_packages]: https://www.tbi.univie.ac.at/RNA/#binary_packages
[doxygen_website]: https://www.doxygen.nl
[vrna_refman_website]: https://www.tbi.univie.ac.at/RNA/ViennaRNA/refman
[vrna_refman_epars]: https://www.tbi.univie.ac.at/RNA/ViennaRNA/refman/group__energy__parameters__rw.html
[vrna_refman_mod_base]: https://www.tbi.univie.ac.at/RNA/ViennaRNA/refman/group__modified__bases.html
[vrna_rtfm]: https://viennarna.readthedocs.io
[bioconda_website]: https://bioconda.github.io
[vrna_bioconda_website]: http://bioconda.github.io/recipes/viennarna/README.html
[vrna_pypi_website]: https://pypi.org/project/ViennaRNA
[vrna_refman_config]: https://www.tbi.univie.ac.at/RNA/ViennaRNA/refman/install.html#configuration
[vrna2_paper]: https://almob.biomedcentral.com/articles/10.1186/1748-7188-6-26
[vrna_paper]: https://link.springer.com/article/10.1007/BF00818163
[nndb_paper]: https://dx.doi.org/10.1093/nar/gkp892
[param2004_paper]: https://doi.org/10.1073/pnas.0401799101
