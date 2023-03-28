set(CCTA_INCLUDES 
    "../include/UNet3dModel.h"
    "../include/ImageHandler.h"
    "../include/CommonUtils.h"
)

set(CCTA_SRCS 
    UNet3dModel.cpp 
    ImageHandler.cpp 
    CommonUtils.cpp
)

add_library(ccta++ ${CCTA_SRCS} ${CCTA_INCLUDES})
target_link_libraries(ccta++ ${TORCH_LIBRARIES} ${ITK_LIBRARIES})