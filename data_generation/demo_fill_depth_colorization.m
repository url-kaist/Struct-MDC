%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% original Matlab code: 
% https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

% modified by Jinwoo Jeon, zinuok@kaist.ac.kr
% last update: 2022/05/09
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clc; clear;  echo off
alpha = 1.0;

%% Main
path = '/home/zinuok/Dataset/PLAD_v3_pose/data/';
isWindow = false;


if isWindow
    slash = '\';
else
    slash = '/';
end

sequences = dir(fullfile(path, '*'));
sequences = sequences(3:end,:);

for i=1:size(sequences)
    seq = sequences(i).name;
    seq_path = strcat(path,seq);
    
    rgb_path = strcat(seq_path,slash,'image',slash);
    gt_path  = strcat(seq_path,slash,'ground_truth',slash);
    
    % get RGB images & GT images paths in .png
    RGBs = dir(strcat(rgb_path, '*.png'));
    GTs  = dir(strcat(gt_path, '*.png'));

    total = size(RGBs);

    
    [s, ss] = size(RGBs);
    
    parfor j=1:s
        tic;
        rgb_path = strcat(RGBs(j).folder, slash, RGBs(j).name);
        gt_path  = strcat(GTs(j).folder, slash, GTs(j).name);
        
        % get file
        rgb = imread(rgb_path);
        gt  = imread(gt_path);
        [row_, col_] = size(gt);


        % pre-process GT
        gt(gt<=2) = 0;
        gt = double(gt) / 256.0;


        % skip if this image has been processed already.
        if row_*col_ == nnz(gt)
            continue
        end

        % filling holes
        gt_filled = fill_depth_colorization_my(rgb, gt, alpha);
        gt_filled = uint16(gt_filled*256.0);
        toc
        
        % save result (, replacing the original incomplete Depth img.)
%         imwrite(gt_filled, strcat(test, GTs(j).name), 'PNG');
        save_depth(gt_filled, strcat(GTs(j).folder, slash, GTs(j).name));

        printStr=sprintf('[%s] [%d/%d] processed [%s]. (^_^)/', seq, j,total, GTs(j).name);
        disp(printStr);
        
        % Option: visualize
        subplot(1,2,1)
        imagesc(gt);
        
        subplot(1,2,2)
        imagesc(gt_filled);


                
    end

 
end











%% Function Def.
function save_depth(file, path)
    imwrite(file, path, 'PNG');
end


% Preprocesses the kinect depth image using a gray scale version of the
% RGB image as a weighting for the smoothing. This code is a slight
% adaptation of Anat Levin's colorization code:
%
% See: www.cs.huji.ac.il/~yweiss/Colorization/
%
% Args:
%   imgRgb - HxWx3 matrix, the rgb image for the current frame. This must
%            be between 0 and 1.
%   imgDepth - HxW matrix, the depth image for the current frame in
%              absolute (meters) space.
%   alpha - a penalty value between 0 and 1 for the current depth values.
function denoisedDepthImg = fill_depth_colorization_my(imgRgb, imgDepth, alpha)
  error(nargchk(2, 3, nargin));
  if nargin < 3
    alpha = 1;
  end
  
  imgIsNoise = (imgDepth == 0 | imgDepth == 10);

  maxImgAbsDepth = max(imgDepth(~imgIsNoise));
  imgDepth = imgDepth ./ maxImgAbsDepth;
  imgDepth(imgDepth > 1) = 1;
  
  assert(ndims(imgDepth) == 2);
  [H, W] = size(imgDepth);
  numPix = H * W;
  
  indsM = reshape(1:numPix, H, W);
  
  knownValMask = ~imgIsNoise;
  
  grayImg = rgb2gray(imgRgb);

  winRad = 1;
  
  len = 0;
  absImgNdx = 0;
  cols = zeros(numPix * (2*winRad+1)^2,1);
  rows = zeros(numPix * (2*winRad+1)^2,1);
  vals = zeros(numPix * (2*winRad+1)^2,1);
  gvals = zeros(1, (2*winRad+1)^2);

  for j = 1 : W
    for i = 1 : H
      absImgNdx = absImgNdx + 1;
      
      nWin = 0; % Counts the number of points in the current window.
      for ii = max(1, i-winRad) : min(i+winRad, H)
        for jj = max(1, j-winRad) : min(j+winRad, W)
          if ii == i && jj == j
            continue;
          end

          len = len+1;
          nWin = nWin+1;
          rows(len) = absImgNdx;
          cols(len) = indsM(ii,jj);
          gvals(nWin) = grayImg(ii, jj);
        end
      end

      curVal = grayImg(i, j);
      gvals(nWin+1) = curVal;
      c_var = mean((gvals(1:nWin+1)-mean(gvals(1:nWin+1))).^2);

      csig = c_var*0.6;

      
 
      mgv = min((double(gvals(1:nWin))-double(curVal)).^2);
      if csig < (-mgv/log(0.01))
        csig=-mgv/log(0.01);
      end
      
      if csig < 0.000002
        csig = 0.000002;
      end

      gvals(1:nWin) = exp(-(double(gvals(1:nWin))-double(curVal)).^2/csig);
      gvals(1:nWin) = gvals(1:nWin) / sum(gvals(1:nWin));
      vals(len-nWin+1 : len) = -gvals(1:nWin);

      % Now the self-reference (along the diagonal).
      len = len + 1;
      rows(len) = absImgNdx;
      cols(len) = absImgNdx;
      vals(len) = 1; %sum(gvals(1:nWin));
    end
  end

  vals = vals(1:len);
  cols = cols(1:len);
  rows = rows(1:len);
  A = sparse(rows, cols, vals, numPix, numPix);
   
  rows = 1:numel(knownValMask);
  cols = 1:numel(knownValMask);
  vals = knownValMask(:) * alpha;
  G = sparse(rows, cols, vals, numPix, numPix);
  
  new_vals = (A + G) \ (vals .* imgDepth(:));
  new_vals = reshape(new_vals, [H, W]);
  
  denoisedDepthImg = new_vals * maxImgAbsDepth;
end
