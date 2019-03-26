% clear all;
% 
% addpath('/home/mahesh/Downloads/libsvm-3.21/matlab/');
% mex HoGfeatures.cc
% 
% %myFolder = 'img-elec/governor';
% myFolder = 'img-elec/senator';
% 
% filePattern = fullfile(myFolder, '*.jpg');
% datFiles = dir(filePattern);
% 
% hogLinFeat = zeros(length(datFiles),61*61*32);
% for nn = 1:length(datFiles)      
%     
%     if mod(nn,10) == 0
%         fprintf(' %d\n', nn);
%     end
%         
%     baseFileName = datFiles(nn).name;
%     fullFileName = fullfile(myFolder, baseFileName);
%     
%     img = imread(fullFileName);
%     im = double(img);
%     hogfeat = HoGfeatures(im);
%     LinFeat = zeros(1,size(hogfeat,1)*size(hogfeat,2)*size(hogfeat,3));
%     cnt = 1;
%     for i = 1:size(hogfeat,1)
%         for j = 1:size(hogfeat,2)
%             for k = 1:size(hogfeat,3)
%                 LinFeat(cnt) = hogfeat(i,j,k);
%                 cnt = cnt + 1;
%             end
%         end
%     end
%     
%     hogLinFeat(nn,:) = LinFeat;
% end
% 
% %load('stat-gov.mat');
% load('stat-sen.mat');
% 
% face_landmark = [face_landmark,hogLinFeat];
% 
% sample_labels = ones(length(datFiles),1);
% 
% load('hogmodel14.mat');
% [predict_label, accuracy, dec_values] = svmpredict(sample_labels, face_landmark, model, '-b 1');
% 
%  save(['2dec_values',num2str(14),'.mat'],'dec_values','-mat','-v7.3');
% 
final_features = zeros(112,1);
%for i=1:14
    load(['dec_values',num2str(14),'.mat']);
    final_features(:,1) = dec_values(:,1);
%end

load('stat-gov.mat');

  class = [];
    for j = 1:size(vote_diff,1)
        if(vote_diff(j) >= 0)
            class(j) = int16(1);
        else
            class(j) = int16(2);
        end
    end
  
    
    acc_val = 0;
    m_id = 0;
    max_val = 0;
    cv_size = 12;
    for cv_index = 2:cv_size:(size(vote_diff,1)-cv_size)
        m_id = m_id+1;
        m_id
        cross_validation_samples =  final_features(cv_index:cv_index+cv_size,:);
        cross_validation_labels = class(cv_index:cv_index+cv_size);
        
        train_samples = [final_features(1:cv_index,:); final_features(cv_index+cv_size:size(vote_diff,1),:)];
        train_label = [class(1:cv_index), class(cv_index+cv_size:size(vote_diff,1))];
        
        model = svmtrain2(train_label', train_samples, '-s 0 -c  1.5  -t 0 -b 1');
        [predict_label, accuracy, dec_values] = svmpredict(cross_validation_labels', cross_validation_samples, model, '-b 1');
        
        acc_val = acc_val+ accuracy(1);
        
        if(accuracy(1) > max_val)
            max_val = accuracy(1);
            save(['2B2hogmodel',num2str(i),'.mat'],'model','-mat','-v7.3');
        end
        
    end
    acc = (acc_val*1.0)/m_id
%   