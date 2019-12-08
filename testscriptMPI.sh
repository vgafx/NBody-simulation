#!/bin/bash
#prun -v -1 -np 1 ./
#preserve -t 01:00:00 -# 1 -np 1
#prun -v -1 -np 16 -sge-script $PRUN_ETC/prun-openmpi ./nbody-par-bonus 5000 0 ../nbody.ppm 1000


bodies=(400 800 1200 1600 2000 2400 2800 3200 3600 4000 4400 4800 5200 5600)
steps=(1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000)

echo "Running nbodyPAR"

for body in "${bodies[@]}"
do
	echo "------------------------------------------------"
	prun -v -1 -np 12 -reserve 225097 -sge-script $PRUN_ETC/prun-openmpi ./nbody-par $body 0 ../nbody.ppm 2000 > parbodies.out
	echo "run complete!"


done

echo "First set of Test COMPLETED"
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++"


for step in "${steps[@]}"
do
    	echo "------------------------------------------------"

	prun -v -1 -np 12 -reserve 225097 -sge-script $PRUN_ETC/prun-openmpi ./nbody-par 800 0 ../nbody.ppm $step > parsteps.out
	echo "run complete!"

done

