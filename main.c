// ARCHIVO: main.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <CL/cl.h>

// Definiciones de stb image
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Función para leer el archivo del kernel (.cl)
char* read_kernel_source(const char* filename, size_t* length) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "ERROR: No se pudo abrir el archivo del kernel: %s\n", filename);
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    *length = ftell(file);
    fseek(file, 0, SEEK_SET);
    char* source = (char*)malloc(*length + 1);
    if (!source) {
        fprintf(stderr, "ERROR: Falla en la asignacion de memoria.\n");
        fclose(file);
        return NULL;
    }
    fread(source, 1, *length, file);
    source[*length] = '\0';
    fclose(file);
    return source;
}

// Inversión Secuencial (CPU) para Benchmark
void invert_image_cpu(unsigned char* data, int width, int height) {
    size_t image_size = (size_t)width * (size_t)height * 4;
    for (size_t i = 0; i < image_size; i += 4) {
        data[i + 0] = 255 - data[i + 0]; // R
        data[i + 1] = 255 - data[i + 1]; // G
        data[i + 2] = 255 - data[i + 2]; // B
    }
}

// Función de comprobación de errores
void checkError(cl_int ret, const char* msg) {
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "ERROR en OpenCL durante %s: %d\n", msg, ret);
        exit(1);
    }
}

int main() {
    const char* image_path = "input.jpg";
    const char* kernel_filename = "kernel.cl";
    int width, height, num_channels;

    // Variables de tiempo
    clock_t start_time, end_time;
    double cpu_time_used, gpu_time_used;

    // Variables OpenCL
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_mem image_mem_obj = NULL;
    cl_int ret;
    char* kernel_source_ptr = NULL;
    size_t kernel_source_len;

    // --- I. BENCHMARK CPU (SECUENCIAL) ---
    unsigned char* host_image_data_cpu = stbi_load(image_path, &width, &height, &num_channels, 4);
    if (!host_image_data_cpu) {
        fprintf(stderr, "ERROR: No se pudo cargar la imagen input.jpg.\n");
        return 1;
    }

    printf("--- Iniciando Benchmark CPU (Secuencial) ---\n");
    start_time = clock();
    invert_image_cpu(host_image_data_cpu, width, height);
    end_time = clock();
    cpu_time_used = ((double)(end_time - start_time)) * 1000.0 / CLOCKS_PER_SEC;
    printf("=> Tiempo CPU: %.2f ms\n", cpu_time_used);
    stbi_write_png("output_negativo_cpu.png", width, height, 4, host_image_data_cpu, width * 4);


    // --- II. SETUP Y BENCHMARK GPU (PARALELO) ---
    // Recargar imagen original
    unsigned char* host_image_data_gpu = stbi_load(image_path, &width, &height, &num_channels, 4);
    size_t image_size = (size_t)width * (size_t)height * 4;

    // Inicialización OpenCL
    clGetPlatformIDs(1, &platform_id, NULL);
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    // Nota: clCreateCommandQueue está deprecado en versiones nuevas, se usa WithProperties, pero usaremos la básica por compatibilidad si es 1.2
    #ifdef CL_VERSION_2_0
        command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
    #else
        command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    #endif
    checkError(ret, "Contexto/Cola");

    // Cargar Kernel
    kernel_source_ptr = read_kernel_source(kernel_filename, &kernel_source_len);
    program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source_ptr, &kernel_source_len, &ret);
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Log de compilación por si falla
    if (ret != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *) malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("%s\n", log);
        free(log);
    }
    checkError(ret, "Build Program");

    kernel = clCreateKernel(program, "invert_image", &ret);
    checkError(ret, "Create Kernel");

    printf("--- Iniciando Benchmark GPU (Paralelo) ---\n");
    start_time = clock();

    // Crear buffer en GPU y copiar datos (Host -> Device)
    image_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, image_size, host_image_data_gpu, &ret);

    // Argumentos
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&image_mem_obj);
    clSetKernelArg(kernel, 1, sizeof(int), (void*)&width);
    clSetKernelArg(kernel, 2, sizeof(int), (void*)&height);

    // Ejecución
    size_t global_item_size[2] = { (size_t)width, (size_t)height };
    // Usamos NULL en local_work_size para que OpenCL decida el óptimo, o {16, 16} como dice el PDF
    size_t local_item_size[2] = {16, 16};

    // Ajuste: El tamaño global debe ser múltiplo del local
    global_item_size[0] = ((width + 15) / 16) * 16;
    global_item_size[1] = ((height + 15) / 16) * 16;

    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_item_size, local_item_size, 0, NULL, NULL);
    checkError(ret, "EnqueueNDRangeKernel");

    // Leer resultado (Device -> Host)
    ret = clEnqueueReadBuffer(command_queue, image_mem_obj, CL_TRUE, 0, image_size, host_image_data_gpu, 0, NULL, NULL);

    end_time = clock();
    gpu_time_used = ((double)(end_time - start_time)) * 1000.0 / CLOCKS_PER_SEC;
    printf("=> Tiempo GPU: %.2f ms\n", gpu_time_used);

    stbi_write_png("output_negativo_gpu.png", width, height, 4, host_image_data_gpu, width * 4);

    // Limpieza
    free(kernel_source_ptr);
    stbi_image_free(host_image_data_cpu);
    stbi_image_free(host_image_data_gpu);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(image_mem_obj);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    return 0;
}