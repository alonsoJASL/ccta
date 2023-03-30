#include <iostream>
#include <string>
#include <vector>

#include <torch/torch.h>

#include <itkPermuteAxesImageFilter.h>
#include <itkPasteImageFilter.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkResampleImageFilter.h>

#include "../include/CommonUtils.h"
#include "../include/ImageHandler.h"

ImageHandler::ImageHandler() {
}

ImageHandler::ImageHandler(std::string dir, std::string name, bool is_dicom, bool verbose) {
    im = ImageType::New();
    seg = SegmentationType::New();
    ipp = {-1, -1};
    ps = {-1, -1};

    if (is_dicom) {
        LoadDicom(dir, name, verbose);
        dicom_name = name;
    }
    else {
        LoadNifti(dir, name, verbose);
        nifti_name = name;
    }
}

void ImageHandler::LoadNifti(std::string dir, std::string nii_name, bool verbose){
    if (nii_name.find(".nii") == std::string::npos) {
        nii_name += ".nii";
    }

    nii_name = dir + std::string("/") + nii_name;

    try {
        im = itk::ReadImage<ImageType>(nii_name);
    }
    catch (const itk::ExceptionObject & ex) {
        std::cout << ex << std::endl;
        return;
    }
}

void ImageHandler::LoadDicom(std::string dir, std::string dcm_folder, bool verbose){
    using NamesGeneratorType = itk::GDCMSeriesFileNames;
    NamesGeneratorType::Pointer name_gen = NamesGeneratorType::New();

    std::string path_to_dcm = dir + std::string("/") + dcm_folder;

    name_gen->SetUseSeriesDetails(true);
    name_gen->AddSeriesRestriction("0008|0021");
    name_gen->SetGlobalWarningDisplay(false);
    name_gen->SetDirectory(path_to_dcm);

    try {
        using SeriesIdContainer = std::vector<std::string>;
        const SeriesIdContainer & seriesUID = name_gen->GetSeriesUIDs();
        auto                      seriesItr = seriesUID.begin();
        auto                      seriesEnd = seriesUID.end();

        if ((seriesItr != seriesEnd) && verbose) {
            std::cout << "The directory: ";
            std::cout << path_to_dcm << std::endl;
            std::cout << "Contains the following DICOM Series: ";
            std::cout << std::endl;
        }
        else {
            std::cout << "No DICOMs in: " << path_to_dcm << std::endl;
            return;
        }

        while ((seriesItr != seriesEnd) && verbose) {
            std::cout << seriesItr->c_str() << std::endl;
            ++seriesItr;
        }

        std::cout << "\nReading: " << std::endl;
        seriesItr = seriesUID.begin();
        while (seriesItr != seriesUID.end()) {
            std::string seriesIdentifier;
            seriesIdentifier = seriesItr->c_str();
            seriesItr++;

            CommonUtils::log("Reading series: " + seriesIdentifier, "ImageHandler::LoadDicom", verbose);

            using FileNamesContainer = std::vector<std::string>;
            using ReaderType = itk::ImageSeriesReader<ImageType>;
            using ImageIOType = itk::GDCMImageIO;

            FileNamesContainer fileNames = name_gen->GetFileNames(seriesIdentifier);
            auto reader = ReaderType::New();
            auto dicomIO = ImageIOType::New();
            reader->SetImageIO(dicomIO);
            reader->SetFileNames(fileNames);
            reader->ForceOrthogonalDirectionOff(); // properly read CTs with gantry tilt
            reader->Update();

            CommonUtils::log("Reading complete.", "ImageHandler::LoadDicom", verbose);
            
            // add slice to variable im
            auto slice = reader->GetOutput();
            auto sliceRegion = slice->GetLargestPossibleRegion();
            itk::Index<3> sliceIndex = {0, 0, 0};
            itk::Size<3> sliceSize = sliceRegion.GetSize();
            itk::ImageRegion<3> targetRegion(sliceIndex, sliceSize);

            CommonUtils::log("Image properties", "ImageHandler::LoadDicom", verbose);

            itk::PasteImageFilter<ImageType>::Pointer pasteFilter = itk::PasteImageFilter<ImageType>::New();
            pasteFilter->SetSourceImage(slice);
            pasteFilter->SetDestinationImage(im);
            pasteFilter->SetSourceRegion(sliceRegion);
            pasteFilter->SetDestinationIndex(sliceIndex);
            pasteFilter->Update();
            im = pasteFilter->GetOutput();
            CommonUtils::log("Paste filter", "ImageHandler::LoadDicom", verbose);
        }
        CommonUtils::log("Reading complete.", "ImageHandler::LoadDicom", verbose);
    }
    catch (const itk::ExceptionObject & ex) {
        std::cout << ex << std::endl;

    }
}

void ImageHandler::SetImageMaxAndMin(float max, float min, float value){
    im_max = max;
    im_min = min;
    // iterate over image and set all values greater than max to + value
    // and all values less than min to -value

    itk::ImageRegionIterator<ImageType> imIter(im, im->GetLargestPossibleRegion());
    while (!imIter.IsAtEnd()) {
        if (imIter.Get() > im_max) {
            imIter.Set(value);
        }
        else if (imIter.Get() < im_min) {
            imIter.Set(-value);
        }
        ++imIter;
    }
}

void ImageHandler::TransposeImage(int axis1, int axis2, int axis3) {

    using TransposeFilterType = itk::PermuteAxesImageFilter<ImageType>;
    auto transposer = TransposeFilterType::New();
    transposer->SetInput(im);

    TransposeFilterType::PermuteOrderArrayType order;
    order[0] = axis1;
    order[1] = axis2;
    order[2] = axis3;

    transposer->SetOrder(order);
    transposer->Update();

    im = transposer->GetOutput();
}

void ImageHandler::GenerateGrid(std::vector<double>& x, std::vector<double>& y, std::vector<double>& z,
              std::vector<double>& vs3, bool verbose) {

    if (im.IsNull()) {
        std::cout << "No image loaded" << std::endl;
        return;
    }

    // Get the image position and pixel spacing
    ImageType::SpacingType spacing = im->GetSpacing();
    ImageType::PointType origin = im->GetOrigin();
    ImageType::DirectionType direction = im->GetDirection();

    std::string msg = "Info:\n\tSpacing: " + std::to_string(spacing[0]) + ", " + std::to_string(spacing[1]) + ", " + std::to_string(spacing[2]);
    msg += "\n\tOrigin: " + std::to_string(origin[0]) + ", " + std::to_string(origin[1]) + ", " + std::to_string(origin[2]);
    CommonUtils::log(msg, "ImageHandler::GenerateGrid", verbose);

    ipp.at(0) = 0;
    ipp.at(1) = spacing[0];
    // ipp.at(1) = spacing[0] + (im->GetBufferedRegion().GetSize()[2] - 1) * spacing[2];

    ps.at(0) = spacing[2];
    ps.at(1) = spacing[1];

    msg = "Info:\n\tIPP: " + std::to_string(ipp.at(0)) + ", " + std::to_string(ipp.at(1));
    msg += "\n\tPS: " + std::to_string(ps.at(0)) + ", " + std::to_string(ps.at(1));
    CommonUtils::log(msg, "ImageHandler::GenerateGrid", verbose);
    
    double st = std::sqrt(std::pow(ipp.at(0) - ipp.at(1), 2.0)); 
    unsigned int nx = im->GetLargestPossibleRegion().GetSize()[0];
    unsigned int ny = im->GetLargestPossibleRegion().GetSize()[1];
    unsigned int nz = im->GetLargestPossibleRegion().GetSize()[2];
    msg = "Info:\n\tNX: " + std::to_string(nx) + ", NY: " + std::to_string(ny) + ", NZ: " + std::to_string(nz);
    CommonUtils::log(msg, "ImageHandler::GenerateGrid", verbose);

    x.resize(nx);
    y.resize(ny);
    z.resize(nz);

    for (unsigned int ix = 0; ix < nx; ++ix) {
        x.at(ix) = (double) ix * st;
    }
    for (unsigned int jx = 0; jx < ny; ++jx) {
        y.at(jx) = (double) jx * ps.at(0);
    }
    for (unsigned int kx = 0; kx < nz; ++kx) {
        z.at(kx) = (double) kx * ps.at(1);
    }

    vs3.at(0) = st;
    vs3.at(1) = ps.at(0);
    vs3.at(2) = ps.at(1);

    msg = "Info:\n\tVS3: " + std::to_string(vs3[0]) + ", " + std::to_string(vs3[1]) + ", " + std::to_string(vs3[2]);
    CommonUtils::log(msg, "ImageHandler::GenerateGrid", verbose);
}

ImageType::Pointer ImageHandler::GenerateGridData(std::vector<double> x, std::vector<double> y, std::vector<double> z, double vs, bool verbose){
    double half_vs = vs / 2.0;
    unsigned long x_size = std::ceil((x.back() - half_vs) / vs);
    unsigned long y_size = std::ceil((y.back() - half_vs) / vs);
    unsigned long z_size = std::ceil((z.back() - half_vs) / vs);

    std::string msg = "X size: " + std::to_string(x_size) + ", Y size: " + std::to_string(y_size) + ", Z size: " + std::to_string(z_size);
    CommonUtils::log(msg, "ImageHandler::GenerateGridData", verbose);

    std::vector<double> x_grid(x_size), y_grid(y_size), z_grid(z_size);
    for (int i = 0; i < x_size; ++i) {
        x_grid[i] = (i + 0.5) * vs - half_vs;
    }
    for (int j = 0; j < y_size; ++j) {
        y_grid[j] = (j + 0.5) * vs - half_vs;
    }
    for (int k = 0; k < z_size; ++k) {
        z_grid[k] = (k + 0.5) * vs - half_vs;
    }
    CommonUtils::log("Grid generated", "ImageHandler::GenerateGridData", verbose);

    // Create the image, by interpolating it with class member ImageType::Pointer im
    using InterpolatorType = itk::LinearInterpolateImageFunction<ImageType, double>;
    InterpolatorType::Pointer interpolator = InterpolatorType::New();
    // interpolator->SetInputImage(im);

    using ResamplerType = itk::ResampleImageFilter<ImageType, ImageType>;
    const ImageType::SizeType outputSize{x_size, y_size, z_size};
    double outputSpacing[3] = {vs, vs, vs};

    ResamplerType::Pointer resampler = ResamplerType::New();
    resampler->SetInput(im);
    resampler->SetOutputSpacing(outputSpacing);
    resampler->SetSize(outputSize);
    resampler->SetInterpolator(interpolator);
    resampler->SetOutputOrigin(im->GetOrigin()); 
    resampler->Update();

    CommonUtils::log("Image resampled", "ImageHandler::GenerateGridData", verbose);

    return resampler->GetOutput();
}

torch::Tensor ImageHandler::GenerateRoiImage(ImageType::Pointer im_2mm, int &size_c, bool verbose) {
    std::vector<int> sz(3);
    sz[0] = im_2mm->GetLargestPossibleRegion().GetSize()[0];
    sz[1] = im_2mm->GetLargestPossibleRegion().GetSize()[1];
    sz[2] = im_2mm->GetLargestPossibleRegion().GetSize()[2];

    double mean_sz = (sz[0] + sz[1] + sz[2]) / 3;
    double min_sz = std::min_element(sz.begin(), sz.end()) - sz.begin();

    std::cout << "Mean size: " << mean_sz << std::endl;
    std::cout << "Min size: " << min_sz << std::endl;
    
    auto arg_min_sz = std::distance(sz.begin(), std::min_element(sz.begin(), sz.end()));

    std::vector<std::array<double, 3>> c;
    if (mean_sz < 128) {

        c = {{{sz[0] / 2.0, sz[1] / 2.0, sz[2] / 2.0}}};

    } else if (min_sz < 128) {

        if (arg_min_sz == 0) {
            c = {{{sz[0] / 2.0, sz[1] / 2.0, sz[2] / 2.0}},
                 {{sz[0] / 2.0, 64.0, 64.0}},
                 {{sz[0] / 2.0, 64.0, sz[2] - 64.0}},
                 {{sz[0] / 2.0, sz[1] - 64.0, 64.0}},
                 {{sz[0] / 2.0, sz[1] - 64.0, sz[2] - 64.0}}};
        } else if (arg_min_sz == 1) {
            c = {{{sz[0] / 2.0, sz[1] / 2.0, sz[2] / 2.0}},
                 {{64.0, sz[1] / 2.0, 64.0}},
                 {{64.0, sz[1] / 2.0, sz[2] - 64.0}},
                 {{sz[0] - 64.0, sz[1] / 2.0, 64.0}},
                 {{sz[0] - 64.0, sz[1] / 2.0, sz[2] - 64.0}}};
        } else {
            c = {{{sz[0] / 2.0, sz[1] / 2.0, sz[2] / 2.0}},
                 {{64.0, 64.0, sz[2] / 2.0}},
                 {{64.0, sz[1] / 2.0, sz[2] - 64.0}},
                 {{sz[0] - 64.0, 64.0, sz[2] / 2.0}},
                 {{sz[0] - 64.0, sz[1] / 2.0, sz[2] - 64.0}}};
        }
    } else {
        c = {{{sz[0] / 2.0, sz[1] / 2.0, sz[2] / 2.0}},
             {{64.0, 64.0, 64.0}},
             {{64.0, 64.0, sz[2] - 64.0}},
             {{64.0, sz[1] - 64.0, 64.0}},
             {{64.0, sz[1] - 64.0, sz[2] - 64.0}},
             {{sz[0] - 64.0, 64.0, 64.0}},
             {{sz[0] - 64.0, 64.0, sz[2] - 64.0}}, 
             {{sz[0] - 64.0, sz[1] - 64.0, 64.0}},
             {{sz[0] - 64.0, sz[1] - 64.0, sz[2] - 64.0}}};
    }

    const int b = 256; // background images size 2b * 2b, i.e. around 0.5 * 0.5 * 0.5 m^3

    // create background image
    ImageType::Pointer img_b = ImageType::New();
    ImageType::SizeType size;
    size[0] = 2 * b;
    size[1] = 2 * b;
    size[2] = 2 * b;
    ImageType::RegionType region;
    region.SetSize(size);
    img_b->SetRegions(region);
    img_b->Allocate();
    img_b->FillBuffer(-1024);

    // copy 2mm image into background image
    typedef itk::ImageRegionIterator<ImageType> IteratorType;
    ImageType::RegionType inputRegion = im_2mm->GetLargestPossibleRegion();
    inputRegion.SetIndex(0, b - sz[0] / 2);
    inputRegion.SetIndex(1, b - sz[1] / 2);
    inputRegion.SetIndex(2, b - sz[2] / 2);
    
    IteratorType outputIt(img_b, img_b->GetLargestPossibleRegion());
    outputIt.SetRegion(inputRegion);
    IteratorType inputIt(im_2mm, inputRegion);
    outputIt.GoToBegin();
    inputIt.GoToBegin();
    while (!outputIt.IsAtEnd()) {
        outputIt.Set(inputIt.Get());
        ++outputIt;
        ++inputIt;
    }

    size_c = static_cast<int> (c.size());
    torch::Tensor img_tst = torch::zeros({size_c, 1, 128, 128, 128}, torch::kFloat);
    for (int k = 0; k < size_c; k++) {
        const int x_min = b - static_cast<int>(sz[0] / 2.0) + static_cast<int>(c[k][0]) - 64;
        const int x_max = x_min + 128;
        const int y_min = b - static_cast<int>(sz[1] / 2.0) + static_cast<int>(c[k][1]) - 64;
        const int y_max = y_min + 128;
        const int z_min = b - static_cast<int>(sz[2] / 2.0) + static_cast<int>(c[k][2]) - 64;
        const int z_max = z_min + 128;
        img_tst[k][0] = torch::from_blob(img_b->GetPixelContainer()->GetBufferPointer() + x_min + y_min * 512 + z_min * 512 * 512, {128, 128, 128}, torch::kFloat32).clone();
    }

    return img_tst;
}

torch::Tensor ImageHandler::itkImageToTensor(ImageType::Pointer img, bool verbose) {
    // Get img size
    const auto size = img->GetLargestPossibleRegion().GetSize();

    // Create a new tensor with the same size as the img
    torch::Tensor tensor = torch::zeros({1, 1, static_cast<int>(size[2]), static_cast<int>(size[1]), static_cast<int> (size[0])});

    // Copy the img data to the tensor
    float *tensor_data = tensor.data_ptr<float>();
    itk::ImageRegionConstIteratorWithIndex<itk::Image<float, 3>> iterator(img, img->GetLargestPossibleRegion());
    while (!iterator.IsAtEnd()) {
        const auto index = iterator.GetIndex();
        const auto value = iterator.Get();
        const auto tensor_index = index[2] + index[1] * size[2] + index[0] * size[2] * size[1];
        tensor_data[tensor_index] = value;
        ++iterator;
    }

    // Return the tensor
    return tensor;
}