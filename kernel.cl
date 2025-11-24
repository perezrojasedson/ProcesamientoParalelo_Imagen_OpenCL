// ARCHIVO: kernel.cl
__kernel void invert_image(__global uchar* image_data,
                           const int width,
                           const int height) {
    // 1. Identificación del Work-Item
    size_t x = get_global_id(0); // Columna
    size_t y = get_global_id(1); // Fila

    // Protección de bordes
    if (x >= width || y >= height) {
        return;
    }

    // 2. Cálculo del índice lineal
    size_t index = (y * width + x) * 4;

    // Lectura de valores originales (Necesario para todos los retos)
    uchar r = image_data[index + 0];
    uchar g = image_data[index + 1];
    uchar b = image_data[index + 2];

    // ==========================================
    // SECCIÓN 1: RETO 3 (INVERSIÓN ESTÁNDAR)
    // ESTADO: ACTIVO (Para Benchmarking)
    // ==========================================
    // Fórmula: 255 - Valor Original
    //image_data[index + 0] = 255 - r; // R
    //image_data[index + 1] = 255 - g; // G
    //image_data[index + 2] = 255 - b; // B
    
    // ==========================================
    // SECCIÓN 2: RETO 2 (FILTRO MAGENTA)
    // ESTADO: COMENTADO
    // ==========================================

    // Invertir solo Verde (G). R y B quedan intactos.
    // image_data[index + 0] = r;       // R (Intacto)
   image_data[index + 1] = 255 - g; // G (Invertido)
    // image_data[index + 2] = b;       // B (Intacto)


    // ==========================================
    // SECCIÓN 3: RETO 1 (ESCALA DE GRISES)
    // ESTADO: COMENTADO
    // ==========================================
    /*
    float gray = 0.299f * (float)r + 0.587f * (float)g + 0.114f * (float)b;
    uchar gray_final = (uchar)gray;
    
    image_data[index + 0] = gray_final;
    image_data[index + 1] = gray_final;
    image_data[index + 2] = gray_final;
    */
}