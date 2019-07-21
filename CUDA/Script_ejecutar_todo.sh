echo "Multiplicación de matrices en OMP" > resultados.txt


for i in  8 16 32 64 128 256 512 1024  #Imagenes
	do
		echo "Ejecucion con $i hilos." >> resultados.txt
					(time ./omp "$i") &>> resultados.txt
  done
