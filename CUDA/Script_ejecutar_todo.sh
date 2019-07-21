echo "Multiplicación de matrices en OMP" > resultados.txt


for n in  8 16 32 64 128 256 512 1024  #Imagenes
	do
	for t in 16 64 256 1024 4096
		do
		for b in 1 8 16 32
			do
					echo "Ejecucion matrix $n x $n con $t hilos y $b bloques." >> resultados.txt
							(!time ./NxN $n $b $t) &>> resultados.txt
			done
		done
  done
