
classdef my_ClassificationTree < handle

    properties

        % Note: we stick with the Matlab naming conventions from fitctree

        X % training examples
        Y % training labels
        MinParentSize % minimum parent node size
        MaxNumSplits % maximum number of splits

        Verbose % are we printing out debug as we go?

        % add any other properties you want on the lines below...

        root %root node


    end

    methods

        % constructor: implementing the fitting phase

        function obj = my_ClassificationTree(X, Y, MinParentSize, MaxNumSplits, Verbose)

            % set up our training data:
            obj.X = X;
            obj.Y = Y;
            % store the minimum parent node size we're using:
            obj.MinParentSize = MinParentSize;
            % store the maximum number of splits we're using:
            obj.MaxNumSplits = MaxNumSplits;

            % are we printing out debug as we go?:
            obj.Verbose = Verbose;

            % over to you for the rest...

            % add your code on the lines below...

            % (note: a function has also been provided on Moodle to
            % calculate weighted impurity given any set of labels)

            %initialise the Root node
            obj.root = struct('colIndex', 0, 'splitValue', 0, 'leftChild', [], 'rightChild', [], 'class', mode(Y), 'classProbability', obj.classProb(Y));


            %Build tree recursively
            obj.root = obj.buildTree(obj.root, obj.X, obj.Y, 1);


        end


        %build tree
        function node = buildTree(obj, node, examples, labels, depth)
            %Stopping conditions
            if size(examples,1) < obj.MinParentSize || depth > obj.MaxNumSplits || size(unique(labels),1) == 1
                if obj.Verbose
                    fprintf('stopping conditions\n');
                end
                return;
            end

            % get the best splits
            [splitValue, colIndex] = obj.best_split(examples, labels); 
            
            if obj.Verbose
                splitValue
                colIndex
            end
            
            %split using the returned splitting value and column index
            [leftChild, rightChild, leftLabels, rightLabels] = obj.split(examples, labels, colIndex, splitValue);
            
            if obj.Verbose
                depth
                sizeLeft = size(leftChild)
                sizeRight = size(rightChild)
                summary(leftLabels);
                summary(rightLabels);
            end

            % recursively build left and rigth sub trees
            node.colIndex = colIndex;
            node.splitValue = splitValue;

            node.leftChild = obj.buildTree(struct('colIndex', 0, 'splitValue', 0, 'leftChild', [], 'rightChild', [], 'class', mode(leftLabels), 'classProbability', obj.classProb(leftLabels)), leftChild, leftLabels, depth+1);

            node.rightChild = obj.buildTree(struct('colIndex', 0, 'splitValue', 0, 'leftChild', [], 'rightChild', [], 'class', mode(rightLabels), 'classProbability', obj.classProb(rightLabels)), rightChild, rightLabels, depth+1);
        end


        % add any other methods you want on the lines below...

        function [value, index] = best_split(obj, examples, labels)
            bestCol = 0;
            bestValue = 0;
            prevValue = 0;
            bestWGDI = inf;

            %go through all columns
            for i=1:size(examples,2)
                uniqueValues = unique(examples(:,i)); %get Unique Vlaues in that columns
                %go through all rows in uniqueValues
                for j=2:size(uniqueValues,1)
                    col = i;
                    thisValue = uniqueValues(j, 1);
                    %Get index of all values smaller and larger
                    leftInd = examples(:,i) < thisValue;
                    rightInd = examples(:,i) >= thisValue;
                    wgdi = weightedGDI(labels(leftInd), labels) + weightedGDI(labels(rightInd), labels); %Impurity
                    if wgdi < bestWGDI
                        bestCol = col;
                        bestValue = thisValue;
                        prevValue = uniqueValues(j-1,1);
                        bestWGDI = wgdi;
                    end
                end
            end
            value = (bestValue + prevValue) / 2;
            index = bestCol; 
        end

        %% split the node
        function [leftChild, rightChild, leftLabels, rightLabels] = split(obj, examples, labels, colIndex, splitValue)

            leftChildIndex = examples(:,colIndex) < splitValue;
            rightChildIndex = examples(:,colIndex) >= splitValue;

            leftChild = examples(leftChildIndex,:);
            leftLabels = labels(leftChildIndex);

            rightChild = examples(rightChildIndex,:);
            rightLabels = labels(rightChildIndex);
        end


        % the prediction phase:

        function [predictions, scores_dt] = predict(obj, test_examples)

            % get ready to store our predicted class labels:
            predictions = categorical;
            scores_dt = [];

            % over to you for the rest...

            % add your code on the lines below...
            for i=1:size(test_examples,1)
                node = obj.root;
                while isstruct(node) && node.colIndex ~= 0 %make sure node is not leaf node
                    if test_examples(i,node.colIndex) <= node.splitValue
                        node = node.leftChild;
                    else
                        node = node.rightChild;
                    end
                end
                predictions(i,1) = node.class;
                scores_dt(i,:) = node.classProbability;
            end
        end

        %Get class probability
        function classProbability = classProb(obj, labels)
            uniqueLabels = unique(obj.Y); %Unique classes in training data

            result = zeros(1, length(uniqueLabels)); %Empty array with same number of columns as uniquelabels

            for m=1:size(labels,1) %go through all the labels
                this_label = labels(m);
                for n=1:size(uniqueLabels,1) %go through all the unique labels
                    this_unique_label = uniqueLabels(n);
                    if this_label == this_unique_label %check if they match
                        result(1,n) = result(1,n) + 1; % add 1 to the particular column
                    end
                end
            end
            result = result / sum(result); %turn into probabilities
            classProbability = result;
        end
    
        % display the decision tree
        function view(obj, node)
            if isempty(node)
                return;
            end
            feature = node.colIndex;
            value = node.splitValue;
            class = node.class;

            if feature ~= 0
                fprintf('if x%d<%.4f then left elseif x%d>=%.4f then right else %s\n', feature, value, feature, value, class);
            else
                fprintf('%s\n', class);
            end
            %recursively go down left and right sides
            obj.view(node.leftChild);
            obj.view(node.rightChild);
        end
    end
end
   
