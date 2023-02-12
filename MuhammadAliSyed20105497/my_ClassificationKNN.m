classdef my_ClassificationKNN < handle

    
    properties
        
        % Note: we stick with the Matlab naming conventions from fitcknn
        
        X % training examples
        Y % training labels
        NumNeighbors % number of nearest neighbours to consider

        Verbose % are we printing out debug as we go?
    end
    
    methods
        
        % constructor: implementing the fitting phase
        
        function obj = my_ClassificationKNN(X, Y, NumNeighbors, Verbose)
            
            % set up our training data:
            obj.X = X;
            obj.Y = Y;
            % store the number of nearest neighbours we're using:
            obj.NumNeighbors = NumNeighbors;

            
            % are we printing out debug as we go?:
            obj.Verbose = Verbose;     
        end
        
        % the prediction phase:
        
        function [predictions, scores_knn] = predict(obj, test_examples)
            
            % get ready to store our predicted class labels:
            predictions = categorical;
            scores_knn = [];
             
                
            for i=1:size(test_examples,1) %Run through all the testing examples
                test_point = test_examples(i, 1:end);
                distances = [];
            
                for j=1:size(obj.X,1) %Run through all the training examples
                    train_point = obj.X(j, 1:end);
            
                    d = sqrt(sum((test_point - train_point) .^ 2)); %find euclidean distance between two points
                    distances(end+1, 1) = d; 
                end

                 
                [~, ind] = sort(distances); %Find index of all distances min to max
                neighbors = obj.Y(ind(1:obj.NumNeighbors));
                predictions(end+1, 1) = mode(neighbors); %Add the label into predictions array

                
                %SCORES_KNN
                uniqueLabels = unique(obj.Y); %Unique classes in training data

                result = zeros(1, length(uniqueLabels)); %Empty array with same number of columns as uniquelabels
                
                for m=1:length(neighbors) %go through all the neighbors
                    this_neighbor = neighbors(m);
                    for n=1:length(uniqueLabels) %go through all the unique labels
                        this_unique_label = uniqueLabels(n);
                        if this_neighbor == this_unique_label %check if they match
                               result(1,n) = result(1,n) + 1; % add 1 to the particular column
                        end
                    end
                end
                result = result / sum(result); %turn into probabilities
                scores_knn(end+1,:) = result;
            end
        end
        
    end
    
end





























%%%%%%% PART 1 OF REIMPLEMENTATION
%              for i=1:size(test_examples,1)
% 
%                  this_test_example = test_examples(i,1:end);
%  
%                 ind = knnsearch(obj.X, this_test_example, 'K', obj.NumNeighbors);
%              
%                  % each one gets added on to the end of predictions
%                  predictions(end+1, 1) = mode(obj.Y(ind));
%              end       
%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%% PART 2 OF REIMPLEMENTATION FOR K=1
 
           
                
%             for i=1:size(test_examples,1) %Run through all the testing examples
%                 test_point = test_examples(i, 1:end);
%                 distances = [];
% 
%                 for j=1:size(obj.X,1) %Run through all the training examples
%                     train_point = obj.X(j, 1:end);
% 
%                     d = sqrt(sum((test_point - train_point) .^ 2)); %find euclidean distance between two points
%                     distances(end+1, 1) = d;
% 
%                     [~, ind] = min(distances); %Find index of smallest distance
%                 end
%                 predictions(end+1, 1) = mode(obj.Y(ind)); %Add the label into predictions array
%             end