%% Load tsne library
addpath tSNE_matlab/

%% Load files
load('./inf_vecs.csv');
load('./labels.csv');

%% Process labels into text labels

txt_labels = cell(size(labels,1),1);

% Re-label them
for l = 1:size(labels,1)
   label = labels(l);
   
   if (label == 0)
       txt_labels{l} = 'arts';
   elseif (label == 1)
       txt_labels{l} = 'food';
   elseif (label == 2)
       txt_labels{l} = 'politics';
   elseif (label == 3)
       txt_labels{l} = 'tech';
   elseif (label == 4)
       txt_labels{l} = 'ancient law';
   elseif (label == 5)
       txt_labels{l} = 'book of genesis';
   elseif (label == 6)
       txt_labels{l} = 'astronomy';
   elseif (label == 7)
       txt_labels{l} = 'computer science';
   end
end

%% Run tsne
mappedX = tsne(inf_vecs,labels, 2, 30, 30);
% Visualize
gscatter(mappedX(:,1), mappedX(:,2), txt_labels);

% 3-class
ind = (labels == 2) | (labels == 4) | (labels == 5);
X = inf_vecs(ind,:);
mappedX = tsne(X, labels(ind), 2, 30, 30);
gscatter(mappedX(:,1), mappedX(:,2), txt_labels(ind));

% 5-class
% 3-class
ind = (labels == 2) | (labels == 4) | (labels == 5) | (labels == 0) | (labels == 6);
X = inf_vecs(ind,:);
mappedX = tsne(X, labels(ind), 2, 30, 30);
gscatter(mappedX(:,1), mappedX(:,2), txt_labels(ind));
