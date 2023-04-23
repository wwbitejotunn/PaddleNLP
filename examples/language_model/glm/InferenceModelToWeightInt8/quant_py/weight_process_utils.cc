#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

void add_bias_and_interleave_int8s_inplace(int64_t int8_tensor_ptr,
                                           int64_t num_elts)
{
    int8_t* int8_tensor = reinterpret_cast<int8_t *>(int8_tensor_ptr);
    for (int ii = 0; ii < num_elts; ++ii) {
        // int8_tensor[ii] = int8_t(int(int8_tensor[ii]) + 128);
        int8_tensor[ii] = int8_t(int(int8_tensor[ii]));
    }

    // Step 2 will transform the layout of a 32-bit register in CUDA in order to match the int4 layout. This has no
    // performance benefit and is purely so that int4 and int8 have the same layout.
    // Pictorially, this does the following:
    // bit 32                                                      0
    //      [elt_3  elt_2  elt_1  elt_0] (each elt occupies 8 bits)
    //
    // And it will rearrange the output 32 bit register to be the following:
    // bit 32                                                      0
    //      [elt_3  elt_1  elt_2  elt_0] (each elt occupies 8 bits)

    for (int64_t base = 0; base < num_elts; base += 4) {
        std::swap(int8_tensor[base + 1], int8_tensor[base + 2]);
    }
}


void interleave_column_major_tensor(int64_t                    interleaved_quantized_tensor,
                                    const int64_t              quantized_tensor,
                                    const std::vector<size_t>& shape)
{

    // We only want to run this step for weight only quant.
    std::cout<<"### in interleave_column_major_tensor"<<std::endl;
    const size_t num_rows    = shape.size() == 2 ? shape[0] : shape[1];
    const size_t num_cols    = shape.size() == 2 ? shape[1] : shape[2];

    const size_t BITS_PER_ELT  = 8;
    const size_t elts_in_int32 = 32 / BITS_PER_ELT;

    const size_t rows_per_tile = 64;
    std::cout<<"running interleave_column_major_tensor"<<std::endl;
    std::cout<<"num_rows:"<<num_rows<<","
             <<"num_cols:"<<num_cols<<","
             <<"BITS_PER_ELT:"<<BITS_PER_ELT<<","
             <<"elts_in_int32:"<<elts_in_int32<<","
             <<"rows_per_tile:"<<rows_per_tile<<std::endl;

    const uint32_t* input_byte_ptr  = reinterpret_cast<const uint32_t*>(quantized_tensor);
    uint32_t*       output_byte_ptr = reinterpret_cast<uint32_t*>(interleaved_quantized_tensor);


    const size_t num_vec_rows      = num_rows / elts_in_int32;
    const size_t vec_rows_per_tile = rows_per_tile / elts_in_int32;
    const size_t interleave        = 2;
    std::cout<<"num_vec_rows:"<<num_vec_rows<<","
             <<"vec_rows_per_tile:"<<vec_rows_per_tile<<","
             <<"interleave:"<<interleave<<std::endl;
    for (int read_col = 0; read_col < num_cols; ++read_col) {
        const size_t write_col = read_col / interleave;
        for (int base_vec_row = 0; base_vec_row < num_vec_rows; base_vec_row += vec_rows_per_tile) {
            for (int vec_read_row = base_vec_row;
                    vec_read_row < std::min(num_vec_rows, base_vec_row + vec_rows_per_tile);
                    ++vec_read_row) {
                const size_t vec_write_row = interleave * base_vec_row
                                                + vec_rows_per_tile * (read_col % interleave)
                                                + vec_read_row % vec_rows_per_tile;

                const size_t read_offset = size_t(read_col) * num_vec_rows + vec_read_row;
                const size_t write_offset = size_t(write_col) * num_vec_rows * interleave + vec_write_row;
                output_byte_ptr[write_offset] = input_byte_ptr[read_offset];
            }
        }
    }
}

PYBIND11_MODULE( weight_only_process, m ){
    m.doc() = "weight_only_process";
    m.def("interleave_column_major_tensor", &interleave_column_major_tensor, "interleave_column_major_tensor" );
    m.def("add_bias_and_interleave_int8s_inplace",&add_bias_and_interleave_int8s_inplace,"add_bias_and_interleave_int8s_inplace");
}

