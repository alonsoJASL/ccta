#ifndef UNET3DMODEL_H
#define UNET3DMODEL_H

#include <torch/torch.h>
#include <iostream>

class UNet3dModel : public torch::nn::Module {
public:
    UNet3dModel(int in_channel, int out_channel);
    torch::Tensor forward(torch::Tensor input);

    torch::nn::Sequential ContractingBlock(int in_channels, int mid_channel, int out_channels, int kernel_size = 3);
    torch::nn::Sequential ExpansiveBlock(int in_channels, int mid_channel, int out_channels, int kernel_size = 3);
    torch::nn::Sequential FinalBlock(int in_channels, int mid_channel, int out_channels, int kernel_size = 3);
    torch::Tensor CropAndConcat(torch::Tensor upsampled, torch::Tensor bypass, bool crop = false);

    inline void Print() { std::cout << "Object created, in_channels:" << _inch << ", out_channels:" << _outch << std::endl; }; 

private:
    torch::nn::Sequential _encode1, _encode2, _encode3;
    torch::nn::MaxPool3d _maxpool1, _maxpool2, _maxpool3;
    torch::nn::Sequential _bottleneck;
    torch::nn::Sequential _decode3, _decode2, _final_layer;

    int _inch, _outch;
};

#endif /* CCTATORCHMODEL_H */
