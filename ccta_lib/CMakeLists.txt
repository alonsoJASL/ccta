set(CCTA_INCLUDES 
    "include/UNet3dModel.h"
)

set(CCTA_SRCS 
    src/UNet3dModel.cpp 
)

add_library(ccta_lib ${CCTA_SRCS} ${CCTA_INCLUDES})