
classdef my_ClassificationNaiveBayes < handle
   
    properties
        
        % Note: we stick with the Matlab naming conventions from fitcnb
        
        X % training examples
        Y % training labels
        
        NumObservations % the total number of training examples
        ClassNames % each of the class labels in our problem
        Prior % the prior probabilities of each class, based on the training data
        DistributionParameters % the parameters of each Normal distribution (means and standard deviations)
                    
        Verbose % are we printing out debug as we go?
    end
    
    methods
        
        % constructor: implementing the fitting phase
        
        function obj = my_ClassificationNaiveBayes(X, Y, Verbose)
            
            % set up our training data:
            obj.X = X;
            obj.Y = Y;
            % how many training examples do we have altogether:
            obj.NumObservations = size(obj.Y, 1);

            % are we printing out debug as we go?:
            obj.Verbose = Verbose;
            
            % what class labels are there in this problem:
            obj.ClassNames = unique(obj.Y);

            % Get ready to store the parameters of our Normal distributions
            obj.DistributionParameters = {};
            % and the prior probabilities:
            obj.Prior = [];
            
            % for each class in the problem:
            for i = 1:length(obj.ClassNames)
                
                % grab the current class name:
				this_class = obj.ClassNames(i);
                % get all the examples belonging to this class:
                examples_from_this_class = obj.X(obj.Y==this_class,:);
                
                % count them, and divide by the total number of examples to 
                % estimate the prior probability of observing this class:
                obj.Prior(end+1) = size(examples_from_this_class,1) / obj.NumObservations;
                
                if Verbose
                    % informative fprintf example:
                    fprintf('\nClass %s prior probability = %.2f\n\n', this_class, obj.Prior(end));
                end
                
                % and estimate the parameters of a Normal distribution from
                % the values seen for each feature (within this class):
                % (for loop over the features for clarity):
                for j = 1:size(obj.X, 2)
                    
                    % mean and standard deviation:
                    obj.DistributionParameters{i,j} = [mean(examples_from_this_class(:,j)); std(examples_from_this_class(:,j))];
                    
                    if Verbose
                        % informative fprintf example:
                        fprintf('Class %s, feature %d: mean=%.2f, standard deviation=%.2f\n', this_class, j, obj.DistributionParameters{i,j}(1), obj.DistributionParameters{i,j}(2));
                    end
                end
                                
            end
            
            if Verbose
                % quick dropped semi-colons examples:
                obj.Prior
                obj.DistributionParameters
            end
            
            
        end
        
        % the prediction phase:
        
        function [predictions, scores_nb] = predict(obj, test_examples)
 
            % get ready to store our predicted class labels:
            predictions = categorical;
            scores_nb = [];
            
            % set to 1 to pre-allocate the arrays; slight performance gain
            % here, but good practice for future:
            if 0
                % either...
                % write something to the last element you'll use:
                predictions(size(test_examples,1), 1) = obj.ClassNames(1);
                % or call zeros() (if it's a numerical array):
                posterior_ = zeros(1, length(obj.ClassNames));
            end
            
            % for all the testing examples we've been passed:
            for i=1:size(test_examples,1)

                if obj.Verbose
                    % informative fprintf example:
                    fprintf('classifying example example %i/%i: ', i, size(test_examples,1));
                end
                
                % grab the next testing example:
                this_test_example = test_examples(i,:);

                % for each class, calcuate a value proportional to the
                % posterior probability by multiplying the likelihood by
                % the prior (Bayes theorem):
                for j=1:length(obj.ClassNames)

                    % (we need to multiply lots of individual likelihoods
                    % (per feature value) together; starting off with 1
                    % lets us write a loop that just does multiplications)
                    this_likelihood = 1;
            
                    % get the overall likelihood of the current example by
                    % multiplying together individual likelihoods for each
                    % feature value in it (treating them as independent
                    % events, as per the class conditional independence
                    % assumption):
                    for k=1:length(this_test_example)
                        % individual likelihoods of each feature value,
                        % given this class, come from the Normal
                        % distributions we estimated for this class during
                        % fitting:
                        this_likelihood = this_likelihood * obj.calculate_pd(this_test_example(k), obj.DistributionParameters{j,k}(1), obj.DistributionParameters{j,k}(2));
                    end
                                        
                    % get the prior probability for this class:
                    this_prior = obj.Prior(j);
                    
                    % multiply the likelihood and the prior for a value
                    % proportional (not equal) to the posterior (hence the
                    % underscore):
                    posterior_(j) = this_likelihood * this_prior;
                end

                % which class had the highest posterior probability (and is
                % therefore the most likely class label):
                [~, winning_index] = max(posterior_);
                % set this as the prediction for this example:
                this_prediction = obj.ClassNames(winning_index);

                if obj.Verbose
                    % informative fprintf example:
                    fprintf('%s\n', this_prediction);
                end
                
                % add it to our array of predictions, and move on to the
                % next example (next iteration of this for loop):
                predictions(i,1) = this_prediction;
                
                
                %%%SCORES NB
                result = zeros(1,length(obj.ClassNames)); %empty array same size as number of classes
                
                for m=1:length(obj.ClassNames)
                    result(1,m) = posterior_(m) ./ sum(posterior_); %get full posterior for each class for a test example
                end
                scores_nb(end+1,:) = result; %store the scores

            end
            
            if obj.Verbose
                % quick dropped semi-colon example:
                predictions
            end
            
        end
        
        % calculate the value of a Normal probability density function
        % (described by mu, sigma) for a feature value x
        function pd = calculate_pd(obj, x, mu, sigma)
        
			first_bit = 1 / sqrt(2*pi*sigma^2);
            second_bit = - ( ((x-mu)^2) / (2*sigma^2) );
            pd = first_bit * exp(second_bit);
        
		end
        
    end
    
end