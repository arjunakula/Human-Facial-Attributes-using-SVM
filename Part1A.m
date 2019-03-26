clear all;
addpath('/home/mahesh/Downloads/libsvm-3.21/matlab/');

load('train-anno.mat');

threshold = mean(trait_annotation);
acc = zeros(1,14);
for i = 1:14
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
    %for cv_index = 2:45:441
    for cv_index = 2:2
        m_id = m_id+1;
        cross_validation_samples =  face_landmark(cv_index:cv_index+40,:);
        cross_validation_labels = class(cv_index:cv_index+40);
        
        train_samples = [face_landmark(1:cv_index,:); face_landmark(cv_index+40:491,:)];
        train_label = [class(1:cv_index), class(cv_index+40:491)];
        
        model = svmtrain2(train_label', train_samples, '-s 0 -c  0.025  -t 0');
        [predict_label, accuracy, dec_values] = svmpredict(cross_validation_labels', cross_validation_samples, model);
        
        acc_val = acc_val+ accuracy(1);
        
        if(accuracy(1) > max_val)
            max_val = accuracy(1);
            save(['model',num2str(i),'.mat'],'model','-mat','-v7.3');
        end
        
    end
    acc(i) = (acc_val*1.0)/m_id;
end