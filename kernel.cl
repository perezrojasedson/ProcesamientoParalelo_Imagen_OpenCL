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

    // 2. Mapeo 2D a 1D (RGBA = 4 bytes por pixel)
    size_t index = (y * width + x) * 4;

    // 3. Cálculo Paralelo: Inversión de Color (Negativo)
    image_data[index + 0] = 255 - image_data[index + 0];    // R
    image_data[index + 1] = 255 - image_data[index + 1];    // G
    image_data[index + 2] = 255 - image_data[index + 2];    // B
}