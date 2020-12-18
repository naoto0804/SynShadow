function dist = evaluate_recovery(gt_set, recovered_set,mask_set)
  pic_num = numel(gt_set);
  total_dist = 0;
  total_pixel = 0;  

  for i = 1 : pic_num
    gt = gt_set{i};
    recovered = recovered_set{i};
      
    gt = double(gt)/255;
    recovered = double(recovered)/255;

    if ~isempty(mask_set)
      mask = mask_set{i};
    
      if numel(mask) ~= numel(recovered)/3
        mask = imresize(mask, [size(recovered,1) size(recovered,2)],'nearest');
      end
    else
      mask = ones([size(recovered,1) size(recovered,2)]);
      %mask = uint8(mask);
    end

    if numel(gt) ~= numel(recovered)
      gt = imresize(gt, [size(recovered,1) size(recovered,2)]);
    end
    
    cform = makecform('srgb2lab');
    gt = applycform(gt,cform);
    recovered = applycform(recovered,cform);

    dist = abs((gt - recovered).* repmat(mask,[1 1 3]));
    total_dist = total_dist + sum(dist(:));
    total_pixel = total_pixel + sum(mask(:));  
    
    fprintf('processed %d/%d\n',i,pic_num);
    
    
  end

  dist = total_dist/total_pixel;

end
