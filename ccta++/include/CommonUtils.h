#ifndef COMMONUTILS_H 
#define COMMONUTILS_H

#include <iostream>

class CommonUtils
{
private:
    /* data */
public:
    static void log(std::string msg, std::string file, bool verbose = false);
};

#endif /* COMMONUTILS_H */