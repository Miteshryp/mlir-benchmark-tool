## Original Plan
1. Take input path of the model to outline
2. Run the outliner pass to generate kernels
  
For each torch-mlir kernel:
     3. Run the pass to preserve argument (and kernel name) information
              Kernel name will later be needed to understand which python
              kernel to invoke
     4. Run lowering pipeline to generate 'll' file, and compile it into
              an '.so' file
     5. Using the argument information, generate the
              required data to be passed into both C++ and python kernel
     6. Verify that the returned result from both the calls is same
     7. Use the perf-cpp library to generate cycle count of each kernel
  
8. Aggregate the data and generate the .csv for graph plotting
9. Write python program to generate the graph information
  
  
  
   -------------------------------------------------
   Alternative Path (For each kernel)
   -------------------------------------------------
   1. Generate argument metadata before lowering a kernel
   2. Invoke the fuzzer, which will
        a. Read the argument metadata
        b. Generate the required data in a file (along with any metadata
        needed)
   3. Call C++ and Python code seperately, which is going to parse and read
   this data, and then
        invoke their respective kernels and produce outputs
   4. Wrap the entire process in a bash file to automate it
  
   Benefits
    - Each run can be monitored using perf
    - Gives us ability to produce performance metrics
  
   Issue
    - Parsing cost is added in each run
    - Can also be solved in the original approach using perf-cpp

