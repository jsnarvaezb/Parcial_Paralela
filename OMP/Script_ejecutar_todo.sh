echo "Multiplicación de matrices en OMP" > resultados.txt


for i in  8 16 32 64 128 256 512 1024  #Imagenes
  do
  for t in 1 2 4 8 16
   
	do
		echo "Ejecucion $i x $i con $t hilos." >> resultados.txt
					(time ./omp "$i" "$t") &>> resultados.txt
  done
  done 
