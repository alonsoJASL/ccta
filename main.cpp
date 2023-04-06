#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "ccta++/include/ImageHandler.h"
#include "ccta++/include/UNet3dModel.h"
#include "ccta++/include/CommonUtils.h"

void Usage(std::string name);

int main(int argc, char const *argv[]) {    

    if (argc < 4) {
        CommonUtils::log("Not enough arguments\n", "CCTA", true); 
        Usage(argv[0]);
        return 1;
    }

    std::string data_path = argv[1];
    std::string image_name = argv[2];
    std::string models_path = argv[3];

    bool verbose = false;
    bool debug = false;
    std::string verbose_str, debug_str;
    switch (argc) {
    case 4:
        verbose_str = argv[3];
        if (verbose_str == "-v" || verbose_str == "--verbose") {
            verbose = true;
        }
        else if (verbose_str == "-d" || verbose_str == "--debug") {
            debug = true;
        }
        break;

    case 5:
        verbose_str = argv[3];
        debug_str = argv[4];
        if (verbose_str == "-v" || verbose_str == "--verbose") {
            verbose = true;
        }
        else if (verbose_str == "-d" || verbose_str == "--debug") {
            debug = true;
        }
        else if (debug_str == "-v" || debug_str == "--verbose") {
            verbose = true;
        }
        else if (debug_str == "-d" || debug_str == "--debug") {
            debug = true;
        }
    default:
        break;
    }

    if (argc >= 5) {
        std::string verbose_str = argv[4];
        if (verbose_str == "-v" || verbose_str == "--verbose") {
            verbose = true;

        }
        else if (verbose_str == "-d" || verbose_str == "--debug") {
            debug = true;
        }
    }
    CommonUtils::log("Arguments parsed", "CCTA");

    bool is_dicom = (image_name.find(".nii") == std::string::npos);
    
    std::string image_type = (is_dicom) ? "DICOM" : "NIFTI";
    CommonUtils::log("Loading " + image_type + " image: " + image_name, "CCTA", verbose);

    std::unique_ptr<ImageHandler> image_handler = std::make_unique<ImageHandler>(data_path, image_name, is_dicom, verbose);
    CommonUtils::log("Image loaded", "CCTA", verbose);

    std::string save_name = (is_dicom) ? image_handler->GetDicomName() + "_input.nii" : "input_" + image_handler->GetNiftiName();

    image_handler->SetImageMaxAndMin(1024, -1024, 1024); 
    CommonUtils::log("Image max and min set", "CCTA", verbose);

    image_handler->TransposeImage(2, 1, 0);
    CommonUtils::log("Image transposed", "CCTA", verbose);

    if (debug){
        image_handler->SaveImage(data_path + "/transposed.nii", verbose);
    }

    std::vector<double> x, y, z, vs3(3);
    image_handler->GenerateGrid(x, y, z, vs3, verbose);
    CommonUtils::log("Grid generated", "CCTA", verbose);

    // ROI 1
    double vs = 2;
    ImageType::Pointer im_2mm = image_handler->GenerateGridData(x, y, z, vs, verbose);
    
    if (im_2mm.IsNull()) {
        CommonUtils::log("Grid image is null", "CCTA", verbose);
        return 1;
    }

    if (debug) {
       image_handler->SaveNifti<ImageType::PixelType>(im_2mm, (data_path + std::string("/") + "grid.nii"), verbose);
    }

    int size_c;
    torch::Tensor img_tst = image_handler->GenerateRoiImage(im_2mm, size_c, verbose);

    int in_channels = 1,
        out_channels = 11;
    std::unique_ptr<UNet3dModel> unet = std::make_unique<UNet3dModel>(in_channels, out_channels);
    unet->SetModelsPath(models_path);
    unet->Print(verbose);

    unet->LoadSegInitModel();
    unet->to(torch::Device("cpu"), torch::kFloat); 

    // ROI 2

    // SEG CCTA

    // POST PROCESS

    // SAVE

    return 0;
}

void Usage(std::string name) {
    std::cerr << "Usage: " << name;
    std::cerr << "  DATA_PATH IMAGE_NAME[.nii|dcm_folder] MODELS_PATH [-v | --verbose] [-d | --debug]" << std::endl;
}
