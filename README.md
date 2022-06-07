# Hi-Tree library for interactive manual scaffolding

**Note**: this version is preliminary but provides an overview of essential implementation details for Hi-Tree model.

## Overview

Hi-Tree actively uses Split/Merge tree structures (Treaps) to efficiently handle contig reverse and move operations without need for overwriting 2D data.

## Demo
You can try it by using [Hi-Tree Server](https://github.com/AxisAlexNT/HiTree_Server) to visualize and edit Hi-C contact maps in [Hi-Tree Web UI](https://github.com/AxisAlexNT/HiTree_WebUI).

## Building from source
You can issue `python3 setup.py bdist_wheel` to build library from sources, then it could be installed using pip: `pip install dist\libhitree-1.0rc1.dev1-py3-none-any.whl`.