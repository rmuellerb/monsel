#!/usr/bin/fish

gcc -xc -std=c11 -O2 -g -Wall -o monsel monsel.c main.c tuning.c -lm
