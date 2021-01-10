
obj:
	nvcc  main.cu -o obj -arch=sm_75 -gencode=arch=compute_75,code=sm_75

obj2:
	nvcc  -Xptxas --dont-merge-basicblocks,--return-at-end,-v main.cu  -o obj -v -rdc=true

dobj:
	nvcc -G -v -Xptxas -v -lineinfo main.cu  -o obj 

clean:
	rm ./obj