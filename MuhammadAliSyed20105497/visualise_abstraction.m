% Description: generate a 2D visualisation of the abstraction produced by a
% classifier
%
% Inputs: 
% m: a classifier
% 
% Outputs:
% None
% 
% Notes: 
% You can just assume for now that the classifier has been trained on only
% two predictive features. We'll return to relax this assumption later on.
%
function visualise_abstraction(m, train_examples)

    figure; % open a new figure window, ready for plotting
    
    % add your code on the lines below...
    
    hold('on');
    
  

    % add your code on the lines below:

    minX = min(train_examples(:,1));
    maxX = max(train_examples(:,1));
    minY = min(train_examples(:,2));
    maxY = max(train_examples(:,2));
    pointDistX = (maxX - minX)/100;
    pointDistY = (maxY - minY)/100;

    
%%%%%%  WITHOUT MESHGRID  %%%%%%%%%%
    X = [];  
    Y = [];
    mesh = zeros(0,2);
    X = minX:pointDistX:maxX;
    Y = minY:pointDistY:maxY;
    
    for i=1:size(X')
        pointX = X(i);
        for j=1:size(Y')
            pointY = Y(j);
            mesh(end+1,1) = pointX;
            mesh(end,2) = pointY;
        end
    end
    [mypredictions,~] = m.predict(mesh); %make predictions
    gscatter(mesh(:,1), mesh(:,2), mypredictions);
end






