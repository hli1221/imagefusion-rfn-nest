function metrics = analysis_Reference(image_f,image_1,image_2)

[s1,s2] = size(image_1);
imgSeq = zeros(s1, s2, 2);
imgSeq(:, :, 1) = image_1;
imgSeq(:, :, 2) = image_2;

image1 = im2double(image_1);
image2 = im2double(image_2);
image_fused = im2double(image_f);

metrics.EN = entropy(image_fused);
metrics.SD = analysis_sd(image_fused);
metrics.MI = analysis_MI(image_1,image_2,image_f);
metrics.Nabf = analysis_nabf(image_fused,image1,image2);
metrics.SCD = analysis_SCD(image1,image2,image_fused);
[MS_SSIM,t1,t2]= analysis_ms_ssim(imgSeq, image_f);
metrics.MS_SSIM = MS_SSIM;

end







