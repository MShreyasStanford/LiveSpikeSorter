# Online Spike Sorter
This is an online spike sorter, which can be used to decode neuronal activity in real time. The sorter uses the method described in this [paper](https://ieeexplore.ieee.org/document/8717072).
The sorter uses templates, which have to be delivered by the user. It will match these templates to the measured data which is streamed in from [SpikeGLX](https://billkarsh.github.io/SpikeGLX/).
A quickly made GUI is delivered with the sorter, which can be turned off by switching the macro. The program has networking capabilities, thus allowing one machine for sorting the neuronal data
and another for running the GUI. The events can easily be read out on the receiving computer, thus easily allowing for custom neurofeedback experiments.

Our advice is to use [KiloSort](https://github.com/MouseLand/Kilosort) to generate the templates. 


If you want more information regarding the parameters used, visit the [wiki](https://gitlab.tudelft.nl/mars-lab/online-spike-sorting/onlinespikesorter/-/wikis/home)


## Installing

### Windows

We recommend using Visual Studio 2017 and CUDA 11.3. On this it has been heavily tested, but upgrading should be no problem as long as there are no syntax changes.
We also recommend taking the Visual Studio solution and changing paths to the correct position. You can also use Cmake.

Download CUDA 11.3 (or higher) and set the paths in your environment.
Add the cudnn.h package/extension [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#prerequisites-windows), follow the instructions.

And build the Solution

##### Common Installation Errors

A common error that can occur when opening the visual studio file in the git is that it cannot find the CUDA props file and thus cannot open the project. This can be fixed by doing [this](https://forums.developer.nvidia.com/t/cannot-run-samples-on-ms-visual-studi-2019/72472).

If you get an error which is along the line of 'can't find 'rc.exe'' you have to use the 8.1 SDK, or copy the 'rc.exe' and
its corresponding '.dll' file to your Microsoft Kits bin file.


### Linux

This program did once work on Linux, however it has not been tested in a while. There is a small chanche it will still work, as many things have been changed since. If you are going to build a machine especially to use this tool, we recommend installing windows.



### Mac

As SpikeGLX is unavailable for Mac this cannot be used as a spike sorter. However, it can be used for the GUI.




