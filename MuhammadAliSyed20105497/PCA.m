function [reduced_train, mean_train, train_PCA]  = PCA(train_examples)
    reduced_train = [];
    variance = [];

    %Store mean of training data
    mean_train = mean(train_examples);
    
    %center training data and do pca of this centered data
    centered_train_examples = (train_examples - (repmat(mean_train, [size(train_examples,1) 1])));
    [train_principal_components, ~, ~, ~, ~] = pca(centered_train_examples, 'Centered', false);
    
    %get projected points
    my_train_data_transformed = centered_train_examples * train_principal_components;

    %Variance
    variance = 100 * var(my_train_data_transformed)' ./ sum(var(my_train_data_transformed));

    %reduce dimensions of these projected points
    my_train_data_transformed = my_train_data_transformed(:,1:2);
    reduced_train = my_train_data_transformed;
   
    %Plotting variance
    figure;
    hold("on");
    plot(cumsum(variance));

    train_PCA = train_principal_components;
end