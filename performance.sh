#!/bin/bash

perf stat -e cache-references,cache-misses,branches,branch-misses python evaluate.py "$@"
