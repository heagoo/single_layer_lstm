if [ ! -d "mklml_lnx_2019.0.3.20190220" ]; then 
    wget https://github.com/intel/mkl-dnn/releases/download/v0.18/mklml_lnx_2019.0.3.20190220.tgz
    tar zxf mklml_lnx_2019.0.3.20190220.tgz
fi
g++ test_single_lstm.cpp -o test -I./mklml_lnx_2019.0.3.20190220/include -L./mklml_lnx_2019.0.3.20190220/lib -O2 -mavx2 -mfma -lmklml_gnu -fopenmp -Wl,-rpath ./mklml_lnx_2019.0.3.20190220/lib
