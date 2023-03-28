#include <iostream>
#include "../include/CommonUtils.h"

void CommonUtils::log(std::string msg, std::string file, bool verbose) {
    if (verbose) {
        std::cout << "[" << file << "] " << msg << std::endl;
    }
}
