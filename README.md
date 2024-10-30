[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/j7EgYxTu)
# A4. Matrix Multiply

Due Date: 11/5 at midnight

## Instructions

Implement the matrix multiply (matmul) A * B = C algorithm in four variations:

1. CPU Matrix Multiply
2. GPU Matrix Multiply
3. GPU Tiled Matirx Multiply
4. GPU cuBLAS Library Call

As before, you should collect timing data and analyze the results for increasing
matrix sizes. That is, at what point does the tiled approach pay off in terms
of computational time/speed?

## Reflection Questions

0. Analyze your results, when does it make sense to use the various approaches?
1. How did your speed compare with cuBLAS?
2. What went well with this assignment?
3. What was difficult?
4. How would you approach differently?
5. Anything else you want me to know?

## Submission Checklist

- [ ] Code
  - [ ] CPU (.c)
  - [ ] GPU (.cu)
  - [ ] Tiled (.cu)
  - [ ] cuBlas (.cu)
- [ ] Graph/table of runtimes/profiled code results
- [ ] Reflection (txt, md, pdf, etc)

## Peer Feedback

After the assignment is due,
you will be randomly assigned to review another student's submission.
This will be as a pull request (PR).
One is already opened by default ([PR #1](../../pull/1)) so you can leave your comments there.
You should look at the results, explanations, code, etc for
understandability, readability, style, etc and provide
any constructive or positive insights.
This peer feedback is due one week after the assignment closes.

## Updating the Assignment

To update this assignment as changes are made,
a new PR will be generated.
You can find the tab [here](../../pulls).
On that page you can merge the pull request to get the update instructions.
This may invovle rebasing or merging your contributions, reach out
if you need help with this.

## Debugging

If you encounter errors, I recommend two approahces:

1. Adding print statements both inside your kernel function
   as well as outside.
   This can include: `printf("%s\n", cudaGetErrorString(cudaGetLastError()));`
   to catch any cuda errors.

2. You can check to make sure that you code is not accessing unallocated memory
   by utilizing NVIDIA's memory sanitizer tool.
   You can run it ok `keroppi` using the following line.
   ```sh
   compute-sanitizer --tool memcheck ./your_cuda_executable_not_source
   ```

## Extras

0. Implement the conflict aware tiled matrix multiply
1. Utilize `malloc`, `cudaMalloc`, and `cudaMemcpy`
2. Handle non-square matricies
3. Compare versus other (CPU) BLAS implementations e.g. openBlas, contact me
   if they need to be installed on `keroppi`
