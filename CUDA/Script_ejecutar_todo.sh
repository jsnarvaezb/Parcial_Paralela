echo "Multiplicación de matrices en OMP" > resultados.txt


for n in  8 16 32 64 128 256 512 1024  #Matrix
	do
	for t in 16 64 256 1024 4096 #Hilos
		do
		for b in 8 16 32 64 #Blocks
			do
					echo "Ejecucion matrix $n x $n con $t hilos y $b bloques." >> resultados.txt
							(time ./cuda "$n" "$t" "$b") &>> resultados.txt
			done
		done
  done
