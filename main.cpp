#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "ImageHandler.h"
#include "UNet3dModel.h"
#include "CommonUtils.h"

static void Usage(std::string name) {
    std::cerr << "Usage: " << name;
    std::cerr << "  DATA_PATH IMAGE_NAME[.nii|dcm_folder] MODELS_PATH [-v | --verbose] [-d | --debug]" << std::endl;
}

int main(int argc, char const *argv[]) {    

    if (argc < 3) {
        Usage(argv[0]);
        return 1;
    }

    std::string data_path = argv[1];
    std::string image_name = argv[2];
    std::string models_path = argv[3];

    bool verbose = false;
    bool debug = false;
    switch (argc) {
    case 4:
        std::string verbose_str = argv[4];
        if (verbose_str == "-v" || verbose_str == "--verbose") {
            verbose = true;
        }
        else if (verbose_str == "-d" || verbose_str == "--debug") {
            debug = true;
        }
        break;

    case 5:
        std::string verbose_str = argv[4];
        std::string debug_str = argv[5];
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

    bool verbose = (argc >= 5 && (std::string(argv[4]) == "-v" || std::string(argv[4]) == "--verbose"));
    bool is_dicom = (image_name.find(".nii") == std::string::npos);
    
    std::string image_type = (is_dicom) ? "DICOM" : "NIFTI";
    CommonUtils::log("Loading " + image_type + " image: " + image_name, "CCTA", verbose);

    std::unique_ptr<ImageHandler> image_handler = std::make_unique<ImageHandler>(data_path, image_name, is_dicom, verbose);
    CommonUtils::log("Image loaded", "CCTA", verbose);

    image_handler->SetImageMaxAndMin(1024, -1024, 1024); 
    CommonUtils::log("Image max and min set", "CCTA", verbose);

    image_handler->TransposeImage(2, 1, 0);
    CommonUtils::log("Image transposed", "CCTA", verbose);

    std::vector<double> x, y, z, vs3(3);
    image_handler->GenerateGrid(x, y, z, vs3, verbose);
    CommonUtils::log("Grid generated", "CCTA", verbose);

    int in_channels = 1, out_channels = 11;
    std::unique_ptr<UNet3dModel> unet = std::make_unique<UNet3dModel>(in_channels, out_channels);
    unet->Print();

    // ROI 1

    // ROI 2

    // SEG CCTA

    // POST PROCESS

    // SAVE

    return 0;
}