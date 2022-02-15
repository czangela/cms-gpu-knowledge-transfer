#include <iostream>

__global__ void mykernel(){
    printf("%s %d %d\n", "Hello world ", threadIdx.x, blockIdx.x); 
}
int main() {
mykernel<<<2,8>>>();
cudaDeviceSynchronize();
std::cout << "Hello World!" << std::endl;
return 0;
}