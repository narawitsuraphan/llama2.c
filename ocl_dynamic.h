#ifndef OCL_DYNAMIC_H
#define OCL_DYNAMIC_H

#include <windows.h>
#include <stdio.h>

// --- OpenCL Types Definition (Manual) ---
typedef intptr_t cl_int;
typedef uintptr_t cl_uint;
typedef uintptr_t cl_ulong;
typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef void* cl_context;
typedef void* cl_command_queue;
typedef void* cl_mem;
typedef void* cl_program;
typedef void* cl_kernel;
typedef uintptr_t cl_device_type;
typedef uintptr_t cl_platform_info;
typedef uintptr_t cl_device_info;
typedef uintptr_t cl_context_properties;
typedef uintptr_t cl_mem_flags;
typedef uintptr_t cl_program_info;
typedef uintptr_t cl_kernel_arg_info;

#define CL_DEVICE_TYPE_GPU (1 << 2)
#define CL_PLATFORM_NAME 0x0902
#define CL_DEVICE_NAME 0x102B
#define CL_MEM_READ_ONLY (1 << 2)
#define CL_MEM_WRITE_ONLY (1 << 0)
#define CL_MEM_COPY_HOST_PTR (1 << 5)
#define CL_SUCCESS 0

// --- Function Pointer Signatures ---
typedef cl_int (WINAPI *PFN_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (WINAPI *PFN_clGetPlatformInfo)(cl_platform_id, cl_platform_info, size_t, void*, size_t*);
typedef cl_int (WINAPI *PFN_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (WINAPI *PFN_clGetDeviceInfo)(cl_device_id, cl_device_info, size_t, void*, size_t*);
typedef cl_context (WINAPI *PFN_clCreateContext)(const cl_context_properties*, cl_uint, const cl_device_id*, void (WINAPI *)(const char*, const void*, size_t, void*), void*, cl_int*);
typedef cl_command_queue (WINAPI *PFN_clCreateCommandQueue)(cl_context, cl_device_id, cl_ulong, cl_int*);
typedef cl_program (WINAPI *PFN_clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (WINAPI *PFN_clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void (WINAPI *)(cl_program, void*), void*);
typedef cl_kernel (WINAPI *PFN_clCreateKernel)(cl_program, const char*, cl_int*);
typedef cl_mem (WINAPI *PFN_clCreateBuffer)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
typedef cl_int (WINAPI *PFN_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, const void*, cl_uint, const void*, void*);
typedef cl_int (WINAPI *PFN_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int (WINAPI *PFN_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const void*, void*);
typedef cl_int (WINAPI *PFN_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, void*, cl_uint, const void*, void*);
typedef cl_int (WINAPI *PFN_clFinish)(cl_command_queue);

// Global Function Pointers
PFN_clGetPlatformIDs oclGetPlatformIDs;
PFN_clGetPlatformInfo oclGetPlatformInfo;
PFN_clGetDeviceIDs oclGetDeviceIDs;
PFN_clGetDeviceInfo oclGetDeviceInfo;
PFN_clCreateContext oclCreateContext;
PFN_clCreateCommandQueue oclCreateCommandQueue;
PFN_clCreateProgramWithSource oclCreateProgramWithSource;
PFN_clBuildProgram oclBuildProgram;
PFN_clCreateKernel oclCreateKernel;
PFN_clCreateBuffer oclCreateBuffer;
PFN_clEnqueueWriteBuffer oclEnqueueWriteBuffer;
PFN_clSetKernelArg oclSetKernelArg;
PFN_clEnqueueNDRangeKernel oclEnqueueNDRangeKernel;
PFN_clEnqueueReadBuffer oclEnqueueReadBuffer;
PFN_clFinish oclFinish;

int init_opencl_dynamic() {
    HMODULE lib = LoadLibraryA("opencl.dll");
    if (!lib) {
        printf("Error: opencl.dll not found in System32.\n");
        return 0;
    }

    oclGetPlatformIDs = (PFN_clGetPlatformIDs)GetProcAddress(lib, "clGetPlatformIDs");
    oclGetPlatformInfo = (PFN_clGetPlatformInfo)GetProcAddress(lib, "clGetPlatformInfo");
    oclGetDeviceIDs = (PFN_clGetDeviceIDs)GetProcAddress(lib, "clGetDeviceIDs");
    oclGetDeviceInfo = (PFN_clGetDeviceInfo)GetProcAddress(lib, "clGetDeviceInfo");
    oclCreateContext = (PFN_clCreateContext)GetProcAddress(lib, "clCreateContext");
    oclCreateCommandQueue = (PFN_clCreateCommandQueue)GetProcAddress(lib, "clCreateCommandQueue");
    oclCreateProgramWithSource = (PFN_clCreateProgramWithSource)GetProcAddress(lib, "clCreateProgramWithSource");
    oclBuildProgram = (PFN_clBuildProgram)GetProcAddress(lib, "clBuildProgram");
    oclCreateKernel = (PFN_clCreateKernel)GetProcAddress(lib, "clCreateKernel");
    oclCreateBuffer = (PFN_clCreateBuffer)GetProcAddress(lib, "clCreateBuffer");
    oclEnqueueWriteBuffer = (PFN_clEnqueueWriteBuffer)GetProcAddress(lib, "clEnqueueWriteBuffer");
    oclSetKernelArg = (PFN_clSetKernelArg)GetProcAddress(lib, "clSetKernelArg");
    oclEnqueueNDRangeKernel = (PFN_clEnqueueNDRangeKernel)GetProcAddress(lib, "clEnqueueNDRangeKernel");
    oclEnqueueReadBuffer = (PFN_clEnqueueReadBuffer)GetProcAddress(lib, "clEnqueueReadBuffer");
    oclFinish = (PFN_clFinish)GetProcAddress(lib, "clFinish");

    if (!oclGetPlatformIDs || !oclEnqueueNDRangeKernel) {
        printf("Error: Failed to load essential OpenCL functions.\n");
        return 0;
    }

    printf("Successfully loaded OpenCL Driver (Dynamic).\n");
    return 1;
}

#endif
