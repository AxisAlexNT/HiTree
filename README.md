# HiCT library for interactive manual scaffolding using Hi-C contact maps

**Note**: this version is preliminary but provides an overview of essential implementation details for HiCT model.

## Overview

Hi-Tree actively uses Split/Merge tree structures (Treaps) to efficiently handle contig reverse and move operations without need for overwriting 2D data.

### Features
* Support for rearrangement operations (contig/scaffold reversal and translocation);
* Support for scaffolding operations (grouping multiple contigs into scaffold and ungrouping contigs from scaffold);
* Export of assembly in FASTA format;
* Export of selection context in FASTA format;
* Import of AGP assembly description;
* Saving/loading work state into the file.

#### W.I.P.
* The minimum assembly unit right now is **contig**, which cannot be split into parts;

## Operation instructions
You can try it by using [HiCT Server](https://github.com/ctlab/HiCT_Server) to visualize and edit Hi-C contact maps in [HiCT Web UI](https://github.com/ctlab/HiCT_WebUI).
It is recommended to use virtual environments provided by `venv` module to simplify dependency management.
This library uses HiCT format for the HiC data and you can convert Cooler's `.cool` or `.mcool` files to it using [HiCT utils](https://github.com/ctlab/HiCT_Utils)

## Building from source
You can run `rebuild.sh` script in source directory which will perform static type-checking of module using mypy (it may produce error messages), build library from source and reinstall it, deleting current version.
