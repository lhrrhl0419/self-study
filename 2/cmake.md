# cmake
1. 教程 https://cmake.org/cmake/help/v3.22/guide/tutorial/index.html#guide:CMake%20Tutorial
2. 常用指令
    1. Step 1
        * cmake (目录) # configure the project and generate a native build system
        * cmake --build (目录) # build system to actually compile/link the project
        * CMakeLists.txt
            * cmake_minimum_required(VERSION 3.10) # cmake 版本
            * project((项目名) (可加 VERSION 版本号)) # 确定项目名称与版本号
            * configure_file((头文件文件名) (头文件名)) #引入头文件
            * add_executable((项目名) (代码文件名)) #添加可执行文件
            * target_include_directories((项目名) "(位置，可填${PROJECT_BINARY_DIR})") #加在最后，提供获取include文件位置
        * (头文件)
            * @Tutorial_VERSION_MAJOR@ / @Tutorial_VERSION_MINOR@ #版本号