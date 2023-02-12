% Description: compute the weighted GDI score (a measure of impurity) for a
% particular set of labels
%
% Inputs:
% these_labels: a set of labels we want to calculate a weighted GDI score for
% train_labels: all of the training labels in our overall problem
%
% Outputs:
% wGDI: the weighted GDI score for these_labels
% 
% Notes:
% Depending on your overall approach, this function could be integrated
% into another class as a method, and made more efficient.
%
function wGDI = weightedGDI(these_labels, train_labels)

    % find the number of training labels in the overall problem:
    total_training_labels = size(train_labels,1);
    % find the unique class labels in the overall problem:
    unique_classes = unique(train_labels);

    % find the weighting for this particular set of labels:
    weighting = length(these_labels) / total_training_labels;

    % find the GDI score for this particular set of labels:
    
    % a summation, starting at zero:
    summ = 0;
    % find the total number of labels we're calculating GDI for:
    total_labels = length(these_labels);
    % compute each term in the summation:
    for i=1:length(unique_classes)
        
        % compute, and add to our summation, the contribution for this
        % class (the fraction of labels that match this class, squared):
        
        % the fraction:
        pc = length(these_labels(these_labels==unique_classes(i))) / total_labels;
        
        % the squaring and adding:
        summ = summ + (pc*pc);

    end
    % final part of the GDI calculation:
    g = 1 - summ;

    % apply the weighting to the resulting GDI score:
    wGDI = weighting * g;

end