clear all;
addpath('/home/mahesh/Downloads/libsvm-3.21/matlab/');
mex HoGfeatures.cc

myFolder = 'img/';
filePattern = fullfile(myFolder, '*.jpg');
datFiles = dir(filePattern);

hogLinFeat = zeros(length(datFiles),61*61*32);
for nn = 1:length(datFiles)      
    
    if mod(nn,10) == 0
        fprintf(' %d\n', nn);
    end
        
    baseFileName = datFiles(nn).name;
    fullFileName = fullfile(myFolder, baseFileName);
    
    img = imread(fullFileName);
    im = double(img);
    hogfeat = HoGfeatures(im);
    LinFeat = zeros(1,size(hogfeat,1)*size(hogfeat,2)*size(hogfeat,3));
    cnt = 1;
    for i = 1:size(hogfeat,1)
        for j = 1:size(hogfeat,2)
            for k = 1:size(hogfeat,3)
                LinFeat(cnt) = hogfeat(i,j,k);
                cnt = cnt + 1;
            end
        end
    end
    
    hogLinFeat(nn,:) = LinFeat;
end

load('train-anno.mat');

face_landmark = [face_landmark,hogLinFeat];

threshold = mean(trait_annotation);
acc = zeros(1,14);
for i = 1:14
    i
    class = [];
    for j = 1:491
        if(trait_annotation(j,i) >= threshold(i))
            class(j) = int16(1);
        else
            class(j) = int16(2);
        end
    end
  
    acc_val = 0;
    m_id = 0;
    max_val = 0;
    for cv_index = 2:45:441
        m_id
        m_id = m_id+1;
        cross_validation_samples =  face_landmark(cv_index:cv_index+40,:);
        cross_validation_labels = class(cv_index:cv_index+40);
        
        train_samples = [face_landmark(1:cv_index,:); face_landmark(cv_index+40:491,:)];
        train_label = [class(1:cv_index), class(cv_index+40:491)];
        
        model = svmtrain2(train_label', train_samples, '-s 0 -c  2  -t 0 -b 1');
        [predict_label, accuracy, dec_values] = svmpredict(cross_validation_labels', cross_validation_samples, model);
        
        acc_val = acc_val+ accuracy(1);
        
        if(accuracy(1) > max_val)
            max_val = accuracy(1);
            save(['hogmodel',num2str(i),'.mat'],'model','-mat','-v7.3');
        end
        
    end
    acc(i) = (acc_val*1.0)/m_id;
end