%ComputeBERonSet.m
% This code is used for computing Balanced Error Rate (BER), evaluation of shadow detection for standard comparison to our published works
% Feel free to use the code. Please cite the following papers when you use it
%    [1] Tomas F. Yago Vicente, Minh Hoai, Dimitris Samaras, Noisy Label Recovery for Shadow Detection in Unfamiliar Domains. CVPR 2016
%    [2] Tomas F. Yago Vicente, Le Hou, Chen-Ping Yu, Minh Hoai, and Dimitris Samaras, Large-scale training of shadow detectors with noisily-annotated shadow examples. ECCV 2016
%    [3] Vu Nguyen, Tomas F. Yago Vicente, Maozheng Zhao, Minh Hoai, Dimitris Samaras, Shadow Detection with Conditional Generative Adversarial Networks. ICCV 2017
%
% SBU Dataset is available at
%    http://www3.cs.stonybrook.edu/~cvl/content/datasets/shadow_db/SBU-shadow.zip
%
% Contact information
%    Vu Nguyen - vhnguyen@cs.stonybrook.edu
%
% Usage:
%    GTFolder   : Folder with ground truth masks
%    PredFolder : Folder of predicted masks

function evaluate(GTFolder, PredFolder)
	dImg = dir([GTFolder,'*.png']);
	stats = zeros(numel(dImg),4);
	tsFor1 = 125; % set based on predicted map range of values 0-1 or 0-255
	disp(numel(dImg));

	%loop images
	for i=1:numel(dImg)
		gtImg = imread([GTFolder,dImg(i).name]);
		disp([PredFolder,dImg(i).name(1:end-3),'png']);
		predImg = imread([PredFolder,dImg(i).name(1:end-3),'png']);
		%Take one channel in case of rgb images
		gtImg = gtImg(:,:,1);
		predImg = predImg(:,:,1);
		%GT positives and negatives
		posPoints = gtImg>tsFor1;
		negPoints = gtImg<=tsFor1;
		countPos = sum(posPoints(:));
		countNeg = sum(negPoints(:));

		%Compute true positives
		tp = posPoints & predImg>tsFor1;
		countTP = sum(tp(:));
		%Compute true negatives
		tn = negPoints & predImg<=tsFor1;
		countTN = sum(tn(:));
		%stats: tp count, tn count, gt positive count, gt negative count
		stats(i,:) = [countTP, countTN, countPos, countNeg];
	end
	%Compute stats for set of images
	posAcc = sum(stats(:,1)) / sum(stats(:,3));
	negAcc = sum(stats(:,2)) / sum(stats(:,4));
	BER = 0.5 * (2 - posAcc - negAcc);
    acc_final = (sum(stats(:,1)) + sum(stats(:,2))) / (sum(stats(:,3)) + sum(stats(:,4)));
    fprintf('d/b: %.2f, b/d: %.2f\n',sum(stats(:,4))/sum(stats(:,3)), sum(stats(:,3))/sum(stats(:,4)));
	fprintf('BER: %.2f, pErr: %.2f, nErr: %.2f, acc: %.4f\n',100*BER,100*(1-posAcc),100*(1-negAcc), acc_final);
end
