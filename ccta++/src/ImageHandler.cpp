#include <iostream>
#include <string>
#include <vector>

#include <itkPermuteAxesImageFilter.h>
#include <itkPasteImageFilter.h>

#include "../include/CommonUtils.h"
#include "../include/ImageHandler.h"

ImageHandler::ImageHandler() {
}

ImageHandler::ImageHandler(std::string dir, std::string name, bool is_dicom, bool verbose) {
    im = ImageType::New();
    seg = SegmentationType::New();
    ipp = std::vector<float>(2);
    ps = std::vector<float>(2);

    if (is_dicom) {
        LoadDicom(dir, name, verbose);
    }
    else {
        LoadNifti(dir, name, verbose);
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

        std::cout << "... complete." << std::endl; 
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

    ipp[0] = origin[2];
    ipp[1] = origin[2] + (im->GetBufferedRegion().GetSize()[2] - 1) * spacing[2];

    ps[0] = spacing[1];
    ps[1] = spacing[0];

    msg = "Info:\nIPP: " + std::to_string(ipp[0]) + ", " + std::to_string(ipp[1]);
    msg += "\nPS: " + std::to_string(ps[0]) + ", " + std::to_string(ps[1]);
    CommonUtils::log(msg, "ImageHandler::GenerateGrid", verbose);
    
    double st = std::sqrt(std::pow(ipp[0] - ipp[1], 2.0) +
                          std::pow(ipp[2] - ipp[1], 2.0));
    unsigned int nx = im->GetLargestPossibleRegion().GetSize()[0];
    unsigned int ny = im->GetLargestPossibleRegion().GetSize()[1];
    unsigned int nz = im->GetLargestPossibleRegion().GetSize()[2];
    x.resize(nx);
    y.resize(ny);
    z.resize(nz);

    for (unsigned int ix = 0; ix < nx; ++ix) {
        x[ix] = ix * st;
    }
    for (unsigned int jx = 0; jx < ny; ++jx) {
        y[jx] = jx * (double) ps[0];
    }
    for (unsigned int kx = 0; kx < nz; ++kx) {
        z[kx] = kx * (double) ps[1];
    }

    vs3[0] = st;
    vs3[1] = ps[0];
    vs3[2] = ps[1];

    msg = "Info:\n\tVS3: " + std::to_string(vs3[0]) + ", " + std::to_string(vs3[1]) + ", " + std::to_string(vs3[2]);
    CommonUtils::log(msg, "ImageHandler::GenerateGrid", verbose);
}

ImageType::Pointer ImageHandler::GenerateGridData(std::vector<double> x, std::vector<double> y, std::vector<double> z, double vs, bool verbose = false){
    int x_size = std::ceil((x.back() - vs / 2) / vs);
    int y_size = std::ceil((y.back() - vs / 2) / vs);
    int z_size = std::ceil((z.back() - vs / 2) / vs);

    std::vector<double> x_grid(x_size), y_grid(y_size), z_grid(z_size);
    for (int i = 0; i < x_size; ++i){
        x_grid[i] = (i + 0.5) * vs;
    }
    for (int j = 0; j < y_size; ++j){
        y_grid[j] = (j + 0.5) * vs;
    }
    for (int k = 0; k < z_size; ++k){
        z_grid[k] = (k + 0.5) * vs;
    }

    // Create the image, by interpolating it with im
    
}

    /// @brief Save image to file. Call with either ImageType or SegmentationType
    /// @tparam T can be ImageType or SegmentationType
    /// @param image Image to save
    /// @param filename Path to save image to
    template <typename T>
    void ImageHandler::SaveNifti(typename itk::Image<T, 3>::Pointer image, std::string filename)
{
    using WriterType = itk::ImageFileWriter< itk::Image<T, 3> >;
    auto writer = WriterType::New();
    writer->SetFileName(filename);
    writer->SetInput(image);
    try {
        writer->Update();
    }
    catch (const itk::ExceptionObject & ex) {
        std::cout << ex << std::endl;
    }
}
