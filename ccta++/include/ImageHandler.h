#ifndef IMAGEHANDLER_H
#define IMAGEHANDLER_H

#include "itkImage.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkImageFileWriter.h"

#include <torch/torch.h>

#include "CommonUtils.h"

typedef itk::Image<float, 3> ImageType;
typedef itk::Image<short, 3> SegmentationType;

static const std::string kImageHandler = "ImageHandler";

class ImageHandler {
    public:
        ImageHandler();
        ImageHandler(std::string dir, std::string name, bool is_dicom, bool verbose = false);
        void LoadNifti(std::string dir, std::string nii_name, bool verbose = false);
        void LoadDicom(std::string dir, std::string dcm_folder, bool verbose = false);

        void SetImageMaxAndMin(float max=1024, float min=-1024, float value=1024);
        void TransposeImage(int axis1, int axis2, int axis3);
        void GenerateGrid(std::vector<double> &x, std::vector<double> &y, std::vector<double> &z, std::vector<double> &vs3, bool verbose = false);
        ImageType::Pointer GenerateGridData(std::vector<double> x, std::vector<double> y, std::vector<double> z, double vs, bool verbose = false);

        torch::Tensor GenerateRoiImage(ImageType::Pointer im_2mm, int& size_c, bool verbose = false);
        
        torch::Tensor itkImageToTensor(ImageType::Pointer img, bool verbose = false);

        template<typename T>
        void SaveNifti(typename itk::Image<T, 3>::Pointer image, std::string filename, bool verbose = false);

        inline ImageType::Pointer image(){return im;};
        inline SegmentationType::Pointer segmentation(){return seg;};

        inline void SaveImage(std::string filename, bool verbose = false ){SaveNifti<ImageType::PixelType>(im, filename, verbose );};
        inline void SaveSegmentation(std::string filename, bool verbose = false ){SaveNifti<SegmentationType::PixelType>(seg, filename, verbose );};
        inline void SetVerbose(bool v){verbose = v;};

        inline std::string GetNiftiName(){return nifti_name;};
        inline std::string GetDicomName(){return dicom_name;};

    private:
        ImageType::Pointer im;
        SegmentationType::Pointer seg;
        std::vector<double> ipp, ps;
        float im_max, im_min;
        bool verbose;
        std::string nifti_name, dicom_name;
};

/// @brief Save image to file. Call with either ImageType or SegmentationType
/// @tparam T can be ImageType or SegmentationType
/// @param image Image to save
/// @param filename Path to save image to
template <typename T>
void ImageHandler::SaveNifti(typename itk::Image<T, 3>::Pointer image, std::string filename, bool verbose) {
    using WriterType = itk::ImageFileWriter< itk::Image<T, 3> >;
    auto writer = WriterType::New();
    writer->SetFileName(filename);
    writer->SetInput(image);
    CommonUtils::log("Saving image to " + filename, "ImageHandler::SaveNifti", verbose);
    try {
        writer->Update();
        CommonUtils::log("Success", "ImageHandler::SaveNifti", verbose);
    }
    catch (const itk::ExceptionObject & ex) {
        std::cout << ex << std::endl;
    }
}

#endif /* IMAGEHANDLER_H */
