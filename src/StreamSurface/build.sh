nvcc -std=c++11 -c -arch=sm_50 StreamSurface.cu 
gcc -std=c++11 -c Activity.cpp 
gcc -std=c++11 -c CudaHelper.cpp 
gcc -std=c++11 -c CudaMathHelper.
gcc -std=c++11 -c CudaMathHelper.cpp 
gcc -fpermissive -std=c++11 -c GlyphsActivity.cpp 
gcc -fpermissive -std=c++11 -c main.cpp 
gcc -fpermissive -std=c++11 -c stdafx.cpp 
gcc -fpermissive -std=c++11 -c SteamlinesLineActivity.cpp 
gcc -fpermissive -std=c++11 -c Utils.cpp 
gcc -fpermissive -std=c++11 -c VectorField.cpp 
gcc -fpermissive -std=c++11 -c VectorFieldInfo.cpp 
gcc -std=c++11 -fpermissive -o main VectorFieldInfo.o VectorField.o Utils.o Activity.o CudaHelper.o CudaMathHelper.o GlyphsActivity.o SteamlinesLineActivity.o StreamSurface.o main.o -lm -lstdc++ -lcudart -lGL -lGLU -lglut -lGLEW
