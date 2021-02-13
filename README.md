# PixelShiftSR

This repository is a prototype test that leans on the 
[ESPCN implementation](https://github.com/leftthomas/ESPCN) of 
[Real-Time Single Image and Video Super-Resolution Using an Efficient
Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158).

The project shows that simple sub-pixel row translations in the
camera sensor layout itself will not hinder the results of a
super-resolution network too much. The quality of the output images
only goes slightly down when using random shifted rows, while using
regular shifted rows the results are nearly identical to the
unshifted version.

This was a proof of concept version for a thesis topic.
Educational topics hit with this project:
- Python and Pycharm IDE
- Anaconda and dependency management
- Pytorch and Torchvision
- Linux SSH server connection for training the neural networks
- Image manipulation and computer graphics
- Neural networks and specifically super-resolution networks
