% script for analyse the fused images

fileName_source_l = ['your infrared iamge path'];
fileName_source_r = ['your visible iamge path'];
fileName_fused = ['your fused image path'];

% fused results
fused_image = imread(fileName_fused);
% input
sourceTestImage1 = imread(fileName_source_l);
sourceTestImage2 = imread(fileName_source_r);
tic;
metrics = analysis_Reference(fused_image,sourceTestImage1,sourceTestImage2);
toc;
temp = [metrics.EN, metrics.SD, metrics.MI, metrics.Nabf, metrics.SCD, metrics.MS_SSIM];


