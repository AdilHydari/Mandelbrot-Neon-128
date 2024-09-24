# Mandelbrot set in neon 128 intrinsics using OpenCV
It is pretty much exactly what the the title describes, I wrote some extra comments for anyone who comes across this and wants to understand how I did this. Hopefully this helps any random student who likes ARM like me to understand intrinsics a bit better :)

## Example

![](mandel.png)

## Compilation
Felt too lazy to create a make file but the compilation is straight forward, just use g++ and link openmp use a -march flag and link OpenCV. As a side note, mandel.png must already be in the folder, as I didn't feel like writing the code for creating a file.


