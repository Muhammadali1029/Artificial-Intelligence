classdef my_ClassificationEnsemble2 < handle

    
    properties
        
        % Note: we stick with the Matlab naming conventions from fitcknn
        
        train_examples % training examples
        train_labels % training labels
        ClassNames
        
        Verbose

        my_nb
        my_knn
        my_dt
    end
    
     methods
        
         function obj = my_ClassificationEnsemble2(train_examples , train_labels, Verbose)
               obj.train_examples = train_examples;
               obj.train_labels = train_labels;
               obj.ClassNames = unique(train_labels);
                
               obj.Verbose = Verbose; 

               obj.my_nb = my_fitcnb(train_examples, train_labels);
               obj.my_knn = my_fitcknn(train_examples, train_labels, 'NumNeighbors', 5);
               obj.my_dt = my_fitctree(train_examples, train_labels, 'MinParentSize', 10);
         end

 

         function [predictions_en, scores_en] = predict(obj, test_examples, Verbose)
           
                predictions_en = categorical;

                %Prediction nb
                [pred_nb, scores_nb] = obj.my_nb.predict(test_examples);

                %Prediction knn
                [pred_knn, scores_knn] = obj.my_knn.predict(test_examples);

                %predictions dt
                [pred_dt, scores_dt] = obj.my_dt.predict(test_examples);

                %Ensemble score
                scores_en = (scores_nb + scores_knn + scores_dt) ./ 3;

                 if obj.Verbose
                  nb = pred_nb
                  nb = scores_nb
                  knn = pred_knn
                  knn = scores_knn
                  dt = pred_dt
                  dt = scores_dt
                  en = scores_en
                end

                % use the indices of the maximum values to read out the resulting predicted labels:
                [~, ind] = max(scores_en');
                predictions_en = obj.ClassNames(ind,1);

         end

     end
 end