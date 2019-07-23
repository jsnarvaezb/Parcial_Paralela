echo "Multiplicación de matrices en OMP" > resultados.txt


for n in 1,2,3,4
do
for i in  8, 16, 32, 64, 128, 256, 512, 1024  #Imagenes
	do
		echo "Ejecucion con $n hosts con un N= $i." >> resultados.txt
					( time mpirun -np $n --hostfile mpi-hosts ./mpi "$i") &>> resultados.txt
  done
done
