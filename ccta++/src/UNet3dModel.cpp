#include "../include/CommonUtils.h"
#include "../include/UNet3dModel.h"

UNet3dModel::UNet3dModel(int in_channel, int out_channel)
    : _encode1(ContractingBlock(in_channel, 16, 32)),
      _maxpool1(torch::nn::MaxPool3dOptions(2)),
      _encode2(ContractingBlock(32, 32, 64)),
      _maxpool2(torch::nn::MaxPool3dOptions(2)),
      _encode3(ContractingBlock(64, 64, 128)),
      _maxpool3(torch::nn::MaxPool3dOptions(2)),
      _bottleneck(torch::nn::Sequential(
          torch::nn::Conv3d(torch::nn::Conv3dOptions(128, 128, 3).padding(1)),
          torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)),
          torch::nn::BatchNorm3d(128),
          torch::nn::Conv3d(torch::nn::Conv3dOptions(128, 128, 3).padding(1)),
          torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)),
          torch::nn::BatchNorm3d(256),
          torch::nn::ConvTranspose3d(torch::nn::ConvTranspose3dOptions(256,256,3).stride(2).padding(1).output_padding(1))
          )), 
      _decode3(ExpansiveBlock(128 + 256, 128, 128)),
      _decode2(ExpansiveBlock( 64 + 128,  64, 64)), 
      _final_layer(FinalBlock( 32 +  64,  32, out_channel)) {

    register_module("_encode1", _encode1);
    register_module("_maxpool1", _maxpool1);
    register_module("_encode2", _encode2);
    register_module("_maxpool2", _maxpool2);
    register_module("_encode3", _encode3);
    register_module("_maxpool3", _maxpool3);
    register_module("_bottleneck", _bottleneck);
    register_module("_decode3", _decode3);
    register_module("_decode2", _decode2);
    register_module("_final_layer", _final_layer);

    _inch = in_channel;
    _outch = out_channel;
}

torch::nn::Sequential UNet3dModel::ContractingBlock(int in_channels, int mid_channel, int out_channels, int kernel_size) {
    torch::nn::Sequential block{
        torch::nn::Conv3d(torch::nn::Conv3dOptions(in_channels, mid_channel, kernel_size).padding(1)),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)),
        torch::nn::BatchNorm3d(mid_channel),
        torch::nn::Conv3d(torch::nn::Conv3dOptions(mid_channel, out_channels, kernel_size).padding(1)),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)),
        torch::nn::BatchNorm3d(out_channels)};

    return block;
}

torch::nn::Sequential UNet3dModel::ExpansiveBlock(int in_channels, int mid_channel, int out_channels, int kernel_size){
    torch::nn::Sequential block {
        torch::nn::Conv3d(torch::nn::Conv3dOptions(in_channels, mid_channel, kernel_size).padding(1)),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)),
        torch::nn::BatchNorm3d(mid_channel),
        torch::nn::Conv3d(torch::nn::Conv3dOptions(mid_channel, mid_channel, kernel_size).padding(1)),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)),
        torch::nn::BatchNorm3d(mid_channel),
        torch::nn::ConvTranspose3d(torch::nn::ConvTranspose3dOptions(mid_channel, out_channels, kernel_size).stride(2).padding(1).output_padding(1))
    };

    return block;
}

torch::nn::Sequential UNet3dModel::FinalBlock(int in_channels, int mid_channel, int out_channels, int kernel_size) {
    torch::nn::Sequential block{
        torch::nn::Conv3d(torch::nn::Conv3dOptions(in_channels, mid_channel, kernel_size).padding(1)),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)),
        torch::nn::BatchNorm3d(mid_channel),
        torch::nn::Conv3d(torch::nn::Conv3dOptions(mid_channel, mid_channel, kernel_size).padding(1)),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)),
        torch::nn::BatchNorm3d(mid_channel),
        torch::nn::Conv3d(torch::nn::Conv3dOptions(mid_channel, out_channels, kernel_size).padding(1)),
        torch::nn::Sigmoid()
    };
    return block;
}

torch::Tensor UNet3dModel::CropAndConcat(torch::Tensor upsampled, torch::Tensor bypass, bool crop) {
    if (crop) {
        int c = (bypass.size(2) - upsampled.size(2)) / 2;
        bypass = torch::nn::functional::pad(bypass, torch::nn::functional::PadFuncOptions({-c, -c, -c, -c}));
    }

    return torch::cat({upsampled, bypass}, /*dim=*/1); 
}

void UNet3dModel::Print(bool verbose) {
    std::string msg = "Info: \n\tObject created, in_channels:" + std::to_string(_inch) + ", out_channels:" + std::to_string(_outch);
    msg += "\n\tModels path: " + _models_path;
    CommonUtils::log(msg, "UNet3dModel::Print", true);
}

void UNet3dModel::LoadXModel(std::string x, bool verbose) {
    std::string msg = "Info: Loading <" + x + "> model from: " + _models_path;
    CommonUtils::log(msg, "UNet3dModel::LoadXModel", verbose);

    torch::serialize::InputArchive input_archive;
    input_archive.load_from(_models_path + x);

    this->load(input_archive);
}

