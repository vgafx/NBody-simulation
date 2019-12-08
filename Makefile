CFLAGS	+= -Wall
CFLAGS	+= -O3
CFLAGS	+= -g2
#CFLAGS  += -fopenmp
#CFLAGS  += -msse3
#CFLAGS  += -march=native
#CFLAGS += -fopt-info-vec-all

#-fopt-info-vec-optimized

all: nbody-seq nbody-par

nbody-seq: nbody-seq.c
	gcc $(CFLAGS) -o nbody-seq nbody-seq.c -lm 
nbody-par: nbody-par.c
	mpicc $(CFLAGS) -o nbody-par nbody-par.c -lm


clean:
	rm -f *.o nbody-seq *~ *core
	rm -f *.o nbody-par *~ *core
	rm -f *.o nbody-par-bonus *~ *core
