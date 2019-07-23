EXECS=mpi
MPICC?=mpiCC

all: ${EXECS}

mpi: mpi.c
	${MPICC} -I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib/ -o mpi mpi.c `pkg-config --cflags --libs opencv` -lopencv_core -lopencv_imgproc -lopencv_highgui

clean:
	rm -f ${EXECS}
