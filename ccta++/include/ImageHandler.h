#ifndef IMAGEHANDLER_H
#define IMAGEHANDLER_H

#include "itkImage.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkImageFileWriter.h"
// #include "itkTransposeImageFilter.h"

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

        template<typename T>
        void SaveNifti(typename itk::Image<T, 3>::Pointer image, std::string filename);

        inline ImageType::Pointer image(){return im;};
        inline SegmentationType::Pointer segmentation(){return seg;};

        inline void SaveImage(std::string filename){SaveNifti<ImageType::PixelType>(im, filename);};
        inline void SaveSegmentation(std::string filename){SaveNifti<SegmentationType::PixelType>(seg, filename);};
        inline void SetVerbose(bool v){verbose = v;};

    private:
        ImageType::Pointer im;
        SegmentationType::Pointer seg;
        std::vector<long> ipp, ps;
        float im_max, im_min;
        bool verbose;
};

#endif /* IMAGEHANDLER_H */
