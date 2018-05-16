#!/bin/bash

perf stat -r 10 -e cache-references,cache-misses,branches,branch-misses python inference.py "$@"
