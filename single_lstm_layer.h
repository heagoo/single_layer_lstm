#ifndef MULTI_LSTM_LAYER_H_
#define MULTI_LSTM_LAYER_H_

#include <new>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <mkl.h>
#include <omp.h>

namespace hpj {

template <typename T>
class Matrix {
private:
    int rows_;
    int cols_;
    int stride_;
    T* data_;
    // How many elements was allocated
    int alloc_size_;

    Matrix(const Matrix &m);
    Matrix& operator=(const Matrix &m);

public:
    Matrix() {
        this->rows_ = 0;
        this->cols_ = 0;
        this->stride_ = 0;
        this->data_ = NULL;
        this->alloc_size_ = 0;
    }
    ~Matrix() {
        this->Release();
    }

    void Resize(int rows, int cols) {
        if (rows == rows_ && cols == cols_) {
            return;
        }
        if (rows < 0 && cols < 0) {
            return;
        }
        if (rows == 0 || cols == 0) {
            this->Release();
            return;
        }
        int skip = (16 - cols % 16) % 16;
        stride_ = cols + skip;
        if (stride_ % 256 == 0) { // Need or not?
            stride_ += 4;
        }
        rows_ = rows;
        cols_ = cols;
        if (alloc_size_ >= stride_ * rows) {
            return;
        } else {
            if (data_) {
                free(data_);
            }
            alloc_size_ = stride_ * rows_;
            data_ = (T *)aligned_alloc(64, sizeof(T) * alloc_size_);
            if (data_ == NULL) {
                throw std::bad_alloc();
            }
        }
    }
    T* Data() {
        return data_;
    }
    void Release() {
        if (data_) {
            free(data_);
            data_ = NULL;
        }
        rows_ = 0;
        cols_ = 0;
        stride_ = 0;
        alloc_size_ = 0;
    }
    int Rows() {
        return rows_;
    }
    int Cols() {
        return cols_;
    }
    int Stride() {
        return stride_;
    }
    T* Row(const int idx) {
        //assert(idx < rows_ && idx >= 0);
        return data_ + stride_ * idx;
    }
    T& operator()(int r, int c) { 
        //assert(r >= 0 && r < rows_ && c >= 0 && c < cols_);
        return *(data_ + r * stride_ + c );
    }
};

template <typename T>
class Vector {
private:
    T* data_;
    int size_;
    int alloc_size_;

public:
    Vector() {
        data_ = NULL;
        size_ = 0;
        alloc_size_ = 0;
    }
    ~Vector() {
        this->Release();
    }
    void Resize(int size) {
        if (size <= 0){
            this->Release();
            return;
        }
        int skip = (16 - size % 16) % 16;
        if (alloc_size_ >= size + skip) { // space is enough
            size_ = size;
            return;
        }

        alloc_size_ = size + skip;
        size_ = size;
        if (data_) {
            free(data_);
        }
        data_ = (T *)aligned_alloc(64, sizeof(T) * alloc_size_);
        if (data_ == NULL) {
            throw std::bad_alloc();
        }
    }
    void SetZero() {
        memset(data_, 0, sizeof(T) * size_);
    }
    T* Data() {
        return data_;
    }
    void Release() {
        if (data_) {
            free(data_);
            data_ = NULL;
        }
        size_ = 0;
        alloc_size_ = 0;
    }
    int Size() {
        return size_;
    }
}; 
} // end namespace

class SingleLSTMLayer 
{
public:
    SingleLSTMLayer(const std::string& layer_name, size_t input_size, size_t output_size, float forget_bias, size_t num_threads) {
        this->layer_name = layer_name;
        this->input_size = input_size;
        this->output_size = output_size;
        this->forget_bias = forget_bias;
        this->num_threads = num_threads;

        this->p_input_data = NULL;
        this->cell_num = 4;

        omp_set_num_threads(num_threads);
    }

    virtual ~SingleLSTMLayer() {
    }

    // 网络计算函数,返回值0为正常，其他状态表示异常
    /*
     * xh = [x, h_prev]
     * [i, f, ci, o] = xh * w + b
     * f = f + forget_bias
     *
     * if not use_peephole:
     *   wci = wcf = wco = 0
     *
     *   i = sigmoid(cs_prev * wci + i)
     *   f = sigmoid(cs_prev * wcf + f)
     *   ci = tanh(ci)
     *
     *   cs = ci .* i + cs_prev .* f
     *   cs = clip(cs, cell_clip)
     *
     *   o = sigmoid(cs * wco + o)
     *   co = tanh(cs)
     *   h = co .* o
     *   ```
     */
    virtual int Forward() {
   
        Multiply(xw, p_input_data, w_ifco);
    
        for (int i = 0; i < cell_num; ++i) {
            Multiply(uh, u_ifco, ht);
      
            // it = sigmoid(Wi*xt + Ui*ht + bi)
            DoSigmoid<float, false>(it, xw, i, 0 * output_size,
                      uh, 0 * output_size, 
                      bias, 0 * output_size);
      
            // ft = sigmoid(Wf*xt + Uf*ht + bf)
            DoSigmoid<float, true>(ft, xw, i, 1 * output_size,
                      uh, 1 * output_size, 
                      bias, 1 * output_size);
      
            // ct = ft。ct + it。tanh(Wc*xt + Uc*ht + bc)
            ComputeCt(i);
      
            // ot = sigmoid(wox*xt + woy*yt + woc*ct + bo)
            DoSigmoid<float, false>(ot, xw, i, 3 * output_size, 
                      uh, 3 * output_size, 
                      bias, 3 * output_size);
      
            ComputeHt();
        } // end of for
    }

    //必要的初始化操作，包括输入输出空间分配，临时空间分配等一切可预先分配的操作，返回值0为正常，其他状态表示异常
    virtual int Init() {
        w_ifco.Resize(input_size, 4 * output_size);
        u_ifco.Resize(output_size, 4 * output_size);
  
        bias.Resize(4 * output_size);
  
        it.Resize(output_size);
        ft.Resize(output_size);
        ct.Resize(output_size);
        ot.Resize(output_size);
        ht.Resize(output_size);

        memset(ct.Data(), 0, ct.Size() * sizeof(float));
        memset(ht.Data(), 0, ht.Size() * sizeof(float));
  
        xw.Resize(cell_num, 4 * output_size);
        uh.Resize(4 * output_size);
    }
    
    // Set input
    // Note: we didn't allocate memory for input buffer
    virtual int SetInput(float *p_input_data, int seq_num = 4) {
        this->p_input_data = p_input_data;
        this->seq_num = seq_num;
        return 0;
    }

    // Get output
    // Currently only contains the last output sequence
    virtual float* GetOutput() {
        return ht.Data();
    }

    //设置kernel参数
    // kernel参数，p_weights_data is already transposed
    virtual int SetKernel(float *p_weights_data, float *p_bias_data) {
        // Transpose, [4096][2048] -> [2048][4096]
        float *pw = new float[4 * (input_size + output_size) * output_size];
        for (int i = 0; i < input_size + output_size; ++i) {
            for (int j = 0; j < 4 * output_size; ++j) {
                pw[i * (4 * output_size) + j] = p_weights_data[j * (input_size + output_size) + i];
            }
        }

        float *p = pw;
        for (int r = 0; r < input_size; ++r) {
            for (int c = 0; c < 4 * output_size; ++c) {
                w_ifco(r, c) = *p++;
            }
        }

        for (int r = 0; r < output_size; ++r) {
            for (int c = 0; c < 4 * output_size; ++c) {
                u_ifco(r, c) = *p++;
            }
        }

        // icfo -> ifco
        for (int r = 0; r < input_size; ++r) {
            for (int c = 0; c < output_size; ++c) {
                float t = w_ifco(r, c + output_size);
                w_ifco(r, c + output_size) = w_ifco(r, c + 2 * output_size);
                w_ifco(r, c + 2 * output_size) = t;
            }
        }
        for (int r = 0; r < output_size; ++r) {
            for (int c = 0; c < output_size; ++c) {
                float t = u_ifco(r, c + output_size);
                u_ifco(r, c + output_size) = u_ifco(r, c + 2 * output_size);
                u_ifco(r, c + 2 * output_size) = t;
            }
        }

        for (int i = 0; i < bias.Size(); ++i) {
            bias.Data()[i] = p_bias_data[i];
        }
        for (int i = 0; i < output_size; ++i) {
            float t = bias.Data()[i + output_size];
            bias.Data()[i + output_size] = bias.Data()[i + 2 * output_size];
            bias.Data()[i + 2 * output_size] = t;
        }

        delete[] pw;
    }

private:
    // dst(i) = sigmoid[x(i+x_row_off, x_col_off) + y(i+y_row_off) + bias(i)]
    template <typename T, bool foget_gate>
    void DoSigmoid(hpj::Vector<T> &dst, 
                   hpj::Matrix<T> &x, int x_row_off, int x_col_off,
                   hpj::Vector<T> &y, int y_off, 
                   hpj::Vector<T> &bias, int bias_off) {
        assert(x.Rows() == cell_num);
        assert(x.Cols() == 4 * output_size);

        T *px = x.Row(x_row_off) + x_col_off;
        T *py = y.Data() + y_off;
        T *pbias = bias.Data() + bias_off;
        T *pdst = dst.Data();
 
        for (int i = 0; i < dst.Size(); ++i) {
            float sum = px[i] + py[i] + pbias[i];
            if (foget_gate) {
                sum += forget_bias;
            }
            pdst[i] = 1.0f / (1.0f + exp(0.f - sum));
        }
    }

    // C = input * weights, A is the input with the dimension of [cell_num][input_size]
    // dimension of weights is input_size x (4*output_size), like  [1024][4096]
    template <typename T>
    void Multiply(hpj::Matrix<T> &C, float *input, hpj::Matrix<T> &weights) {
        assert(weights.Rows() == input_size);
        assert(weights.Cols() == 4 * output_size);
        assert(C.Rows() == cell_num);
        assert(C.Cols() == 4 * output_size);
    
        CBLAS_LAYOUT    layout = CblasRowMajor;
        CBLAS_TRANSPOSE transA = CblasNoTrans;
        CBLAS_TRANSPOSE transB = CblasNoTrans;
        T *pA = input;
        T *pB = weights.Row(0);
        T *pC = C.Row(0);
        int m = C.Rows();
        int n = C.Cols();
        int k = weights.Rows();
        MKL_INT lda = input_size;
        MKL_INT ldb = weights.Stride();
        MKL_INT ldc = C.Stride(); 
     
        float alpha = 1;
        float beta = 0;
       
        cblas_sgemm(layout, transA, transB, m, n, k, alpha,
                    pA, lda, pB, ldb, beta, pC, ldc);
    }

    template <typename T>
    void Multiply(hpj::Vector<T> &C, hpj::Matrix<T> &A, hpj::Vector<T> &B) {
        CBLAS_LAYOUT    layout = CblasRowMajor;
        CBLAS_TRANSPOSE trans = CblasTrans;
    
        int m = A.Rows();
        int n = A.Cols();
        int lda = A.Stride();
    
        float *a = A.Data();
        float *x = B.Data();
        float *y = C.Data();
        
        MKL_INT incx = 1;
        MKL_INT incy = 1;
     
        float alpha = 1;
        float beta = 0;
        cblas_sgemv(layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }

    // ct = ft。ct + it。tanh(Wc*xt + Uc*ht + bc)
    void ComputeCt(int seq) {
        float *pbias = bias.Data() + 2 * output_size;
        float *pft = ft.Data();
        float *pct = ct.Data();
        float *pit = it.Data();
        float *px = xw.Row(seq) + 2 * output_size;
        float *py = uh.Data() + 2 * output_size;

        for (int i = 0; i < output_size; ++i) {
            float v_tanh = Tanh(px[i] + py[i] + pbias[i]);
            float f_val = pft[i] * pct[i] + pit[i] * v_tanh;
            pct[i] = f_val;
        }
    }

    // ht = ot。tanh(ct)
    void ComputeHt() {
        float *pct = ct.Data();
        float *pht = ht.Data();
        float *pot = ot.Data();
        for (int i = 0; i < output_size; ++i) {
            pht[i] = pot[i] * Tanh(pct[i]);
        }
    }

    float Tanh(float x) {
        float ret;
        if (x > 10.0) {// avoid inf -nan
            ret = 1.0;
        } else if (x < -10.0) {
            ret = -1.0;
        } else {
            float tmp = exp(2 * x);
            ret = (tmp - 1) / (tmp + 1);
        }
        return ret;
    }

private:
    std::string layer_name;  //网络层名字
    size_t num_threads; //设置的线程数目，这里最好都为1

    float *p_input_data; //输入数据指针，多个LSTM Cell输入按顺序排列
    //float *p_output_data; //输出数据指针，暂时只需要一个输出

    // Number of layers
    size_t cell_num;

    // Number of sequences
    size_t seq_num;

    size_t input_size;
    size_t output_size;

    float forget_bias;

    // Combined Wi, Wf, Wc, Wo
    hpj::Matrix<float> w_ifco;
  
    // Combined Ui, Uf, Uc, Uo
    hpj::Matrix<float> u_ifco;
  
    // Combined b
    hpj::Vector<float> bias;
  
    // State and output (for batch size = 1)
    hpj::Vector<float> it;
    hpj::Vector<float> ft;
    hpj::Vector<float> ct;
    hpj::Vector<float> ot;
    hpj::Vector<float> ht;
  
    // Store temp result of x * (Wi, Wf, Wc, Wo) (include all sequences)
    hpj::Matrix<float> xw;
  
    hpj::Vector<float> uh;

    //float *p_weights_data;
    //float *p_bias_data;    //kernel参数，bias向量，大小为 4 * output_size
    //float *p_cell_data //cell data
    //float *p_hidden_data //hidden data
};

#endif // MULTI_LSTM_LAYER_H_
