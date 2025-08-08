# Online Spike Sorter
This is an online spike sorter, which can be used to decode neuronal activity in real time. The sorter uses the method described in this [paper](https://ieeexplore.ieee.org/document/8717072).
The sorter uses templates, which have to be delivered by the user. It will match these templates to the measured data which is streamed in from [SpikeGLX](https://billkarsh.github.io/SpikeGLX/).
A quickly made GUI is delivered with the sorter, which can be turned off by switching the macro. The program has networking capabilities, thus allowing one machine for sorting the neuronal data
and another for running the GUI. The events can easily be read out on the receiving computer, thus easily allowing for custom neurofeedback experiments.

Our advice is to use [KiloSort](https://github.com/MouseLand/Kilosort) to generate the templates. 


If you want more information regarding the parameters used, visit the [wiki](https://gitlab.tudelft.nl/mars-lab/online-spike-sorting/onlinespikesorter/-/wikis/home)


## Installing

### Windows

Build by first configuring the Makefile by modifying the following:

* CUDA_PATH: the location of your CUDA installation
* CUDA_VERSION: version number of your CUDA installation
* PYTHON_PATH: path to your Python installation (used for our C++ scikit-learn wrapper for decoding)
* NUMPY_PATH: path to your NumPy package installation in Python
* WINSDK_INCLUDE_DIR, WINSDK_LIB_DIR: paths to the /Include/<version>/um and /Lib/<version>/um/<architecture> paths of your Windows SDK installation
* SGLX_API_DIR, SGLX_LIB_DIR: we have it as absolute paths right now (TODO: change to relative)

Ensure the following .dll's are in the same directory as the application executable:
* cudnn_cnn_infer64_8.dll
* cudnn_ops_infer64_8.dll
* libgcc_s_seh-1.dll
* libstdc++-6.dll
* libwinpthread-1.dll

Additionally, ensure all necessary .dll's are in $(CUDA_PATH)/bin.

Then, simply build with
```console
foo@bar:~$ make
```

We have only tested the installation with the MSVC compiler and linker (cl.exe). We have not tested compilation of this project under any standard other than C++17.

We recommend using MSVC 2017 and CUDA 11.3. On this it has been heavily tested.


### Linux

This program did once work on Linux, however it has not been tested in a while. There is a small chanche it will still work, as many things have been changed since. If you are going to build a machine especially to use this tool, we recommend installing windows.



### Mac

As SpikeGLX is unavailable for Mac this cannot be used as a spike sorter. However, it can be used for the GUI.


## Running spike sorting pipeline

TODO: Talk about modified Kilosort.

