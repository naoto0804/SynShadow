%% This is the code used for evulating the shadow removal results.
% It takes two arguments; data_type (e.g., ISTD) and result_dir
% $ matlab -nodisplay
% $ evaluate('ISTD', '../../../data/ISTD/test/target/')

function dist = evaluate(data_type, result_dir)
  fd = fopen([data_type, '.txt']);
  a=textscan(fd, '%s');
  fclose(fd);
  testfnlist = a{1};

  %%evaluate darker
  dark_num = length(testfnlist);

  gt_dark = cell(dark_num,1);
  shadow_dark = cell(dark_num,1);
  recovered_dark = cell(dark_num,1);
  mask_dark = cell(dark_num,1);
  mask2_dark = cell(dark_num,1);

  recovery_num = dark_num; %sum(indicator == 1);

  gt_recovery = cell(recovery_num,1);
  shadow_recovery = cell(recovery_num,1);
  recovered_recovery = cell(recovery_num,1);
  mask_recovery = cell(recovery_num,1);
  mask2_recovery = cell(recovery_num,1);

  dark_count = 1;
  recovery_count = 1;

  disp(numel(testfnlist));

  for i = 1 : numel(testfnlist)

    switch data_type
      case 'ISTD+'
        root = '../../../datasets/ISTD+/test';
        recovered_dir = result_dir;
        gt_dir = [root '/target/'];
        mask_dir = [root '/mask/'];
      case 'SRD+'
        root = '../../../datasets/SRD+/test';
        gt_dir = [root '/target/'];
        recovered_dir = result_dir;
        mask_dir = [root '/mask/'];
      otherwise
        exit
    end

    gt_recovery{recovery_count} = imread([gt_dir testfnlist{i}]);
    recovered_recovery{recovery_count} = imread([recovered_dir testfnlist{i}]);
    m = imread([mask_dir testfnlist{i}]);

    if numel(size(m)) == 3
        m = rgb2gray(m);
    end

    m(m~=0)=1;

    m = double(m);

    mask_recovery{recovery_count} = m;

    mask2_recovery{recovery_count} = 1-m;

    recovery_count = recovery_count + 1;
  end



  %% for the overall regions
  dist_12 = evaluate_recovery(gt_recovery, recovered_recovery, cell(0));


  %% for the shadow evaluation
  dist_14 = evaluate_recovery(gt_recovery, recovered_recovery, mask_recovery);

  %% for the non_shadow evalutation
  dist_16 = evaluate_recovery(gt_recovery, recovered_recovery, mask2_recovery);

  dist = [dist_14 dist_16 dist_12];
  disp(dist);
