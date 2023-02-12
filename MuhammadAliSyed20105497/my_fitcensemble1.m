function m = my_fitcensemble1(train_examples, train_labels, varargin)

    % take an extra name-value pair allowing us to turn debug on:
    p = inputParser;

    addParameter(p, 'Verbose', false);
    p.parse(varargin{:});
    % use the supplied parameters to create a new my_ClassificationEncemble
    % object:
    
    m = my_ClassificationEnsemble1(train_examples, train_labels, p.Results.Verbose);
            
end

