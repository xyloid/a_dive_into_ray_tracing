obj:
	nvcc  -Xptxas --dont-merge-basicblocks,-v motion_blur.cu  -o obj -v 

dobj:
	nvcc -G -v -lineinfo motion_blur.cu  -o obj 

clean:
	rm ./obj