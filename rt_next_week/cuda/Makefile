
obj:
	nvcc  motion_blur.cu  -g -o obj -v -arch=sm_75 -gencode=arch=compute_75,code=sm_75

obj2:
	nvcc  -Xptxas --dont-merge-basicblocks,--return-at-end,-v motion_blur.cu  -o obj -v -rdc=true

dobj:
	nvcc -G -v -Xptxas -v -lineinfo motion_blur.cu  -o obj 

clean:
	rm ./obj