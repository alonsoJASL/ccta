#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "UNet3dModel.h"

int main(int argc, char const *argv[]) {    
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;

    int in_channels = 1, out_channels = 11;
    std::unique_ptr<UNet3dModel> unet = std::make_unique<UNet3dModel>(in_channels, out_channels);
    unet->Print();

    std::string current_exec_name = argv[0]; // Name of the current exec program
    std::string filepath = argv[1];

    // load data

    // ROI 1

    // ROI 2

    // SEG CCTA

    // POST PROCESS

    // SAVE

    return 0;
}