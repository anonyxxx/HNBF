
%% Global parameter declaration

global K                 % number of topics
global M                 % number of users
global N                 % number of items

global Best_matTheta
global Best_matBeta
 
% ----- HNBF -----
global G_matTheta        % dim(M, K): latent document-topic intensities
global G_matTheta_Shp    % dim(M, K): varational param of matTheta (shape)
global G_matTheta_Rte    % dim(M, K): varational param of matTheta (rate)

global G_matBeta         % dim(N, K): latent word-topic intensities
global G_matBeta_Shp     % dim(N, K): varational param of matBeta (shape)
global G_matBeta_Rte     % dim(N, K): varational param of matBeta (rate)

global G_matEpsilon      % dim(M, 1): latent word-topic intensities
global G_matEpsilon_Shp  % dim(M, 1): varational param of matEpsilon (shape)
global G_matEpsilon_Rte  % dim(M, 1): varational param of matEpsilon (rate)

global G_matEta          % dim(N, 1): latent word-topic intensities
global G_matEta_Shp      % dim(N, 1): varational param of matEta (shape)
global G_matEta_Rte      % dim(N, 1): varational param of matEta (rate)

global G_vecMu           % dim(M, 1): approximate matD
global G_vecMu_Shp       % dim(M, 1): approximate matD
global G_vecMu_Rte       % dim(M, 1): approximate matD
global G_matGamma        % dim(M, K): approximate matD
global G_matGamma_Shp    % dim(M, K): approximate matD
global G_matGamma_Rte    % dim(M, K): approximate matD

global G_vecPi           % dim(N, 1): approximate matD
global G_vecPi_Shp       % dim(N, 1): approximate matD
global G_vecPi_Rte       % dim(N, 1): approximate matD
global G_matDelta        % dim(N, K): approximate matD
global G_matDelta_Shp    % dim(N, K): approximate matD
global G_matDelta_Rte    % dim(N, K): approximate matD

global vec_matR_ui_shp
global vec_matR_ui_rte
global vec_matR_ui
global vec_matD_ui_shp
global vec_matD_ui_rte
global vec_matD_ui

global G_prior

global matX_train        % dim(M, N): consuming records for training
global matX_test         % dim(M, N): consuming records for testing
global matX_valid        % dim(M, N): consuming records for validation

global matPrecNRecall



%% Experimental Settings
%
% ---------- Statistics of Datasets ---------- 
% LastFm2K      =>  M = 1892    , N = 17632  , NNX = 92,834


NUM_RUNS = 10;
likelihood_step = 10;
check_step = 50;

ini_scale = 0.001;
Ks = [20];
topK = [5, 10, 15, 20];
MaxItr = 400;
G_prior = [3, 1, 0.1, ...
           3, 1, 0.1, ...
           1e2, 1e7, 1e2, 1e7, ...
           200, 1e5];
    
matPrecNRecall = zeros(NUM_RUNS*length(Ks), length(topK)*6);


%% Load Data
env_path = 'dataset/';
[ M, N ] = LoadUtilities(strcat(env_path, 'LastFm2K_train.csv'), strcat(env_path, 'LastFm2K_test.csv'), strcat(env_path, 'LastFm2K_valid.csv'));

usr_zeros = sum(matX_train, 2)==0;
itm_zeros = sum(matX_train, 1)==0;


%% Experiments
for kk = 1:length(Ks)

    %% Paramter settings
    %
    K = Ks(kk);
    usr_batch_size = M;     

    valid_precision = zeros(ceil(MaxItr/check_step), length(topK));
    valid_recall = zeros(ceil(MaxItr/check_step), length(topK));
    valid_nDCG = zeros(ceil(MaxItr/check_step), length(topK));
    valid_MRR = zeros(ceil(MaxItr/check_step), length(topK));
    
    test_precision = zeros(ceil(MaxItr/check_step), length(topK));
    test_recall = zeros(ceil(MaxItr/check_step), length(topK));
    test_nDCG = zeros(ceil(MaxItr/check_step), length(topK));
    test_MRR = zeros(ceil(MaxItr/check_step), length(topK));
    
    train_poisson = zeros(ceil(MaxItr/likelihood_step), 2);
    test_poisson = zeros(ceil(MaxItr/likelihood_step), 2);
    valid_poisson = zeros(ceil(MaxItr/likelihood_step), 2);

    vecD_tmpX = zeros(ceil(MaxItr/likelihood_step), 3);

    
    %% Model initialization
    %
    G_prior(1:6) = [30, 1*K, 0.1*sqrt(K), ...
                        30, 1*K, 0.1*sqrt(K)];
    newFastHNBF(ini_scale, usr_zeros, itm_zeros);
    
    [is_X_train, js_X_train, vs_X_train] = find(matX_train);
    
    itr = 0;
    IsConverge = false;
    while IsConverge == false
        itr = itr + 1;
        lr = 1.0;

        % Sample usr_idx, itm_idx
        [usr_idx, itm_idx, usr_idx_len, itm_idx_len] = sampleData_userwise(usr_batch_size);

        fprintf('Itr: %d  K = %d  ==> ', itr, K);
        fprintf('subPredict_X: ( %d , %d ) , nnz = %d , G_lr = %f \n', usr_idx_len, itm_idx_len, nnz(matX_train(usr_idx, itm_idx)), lr);


        %% Train generator G
        %
        % Train generator G given samples and their scores evluated by D
        Learn_FastHNBF(lr);
        
        
        %% Calculate precision, recall, MRR, and nDCG
        %
        if check_step > 0 && mod(itr, check_step) == 0
            
            % Calculate the metrics on validation set
            fprintf('Validation ... \n');
            indx = itr / check_step;
            if usr_idx_len > 5000 && itm_idx_len > 20000
                user_probe = datasample(usr_idx, min(usr_idx_len, 5000), 'Replace', false);
            else
                user_probe = usr_idx;
            end
            if strcmp(DATA, 'MovieLens100K') || strcmp(DATA, 'MovieLens1M') || strcmp(DATA, 'ML100KPos')
                [valid_precision(indx,:), valid_recall(indx,:), valid_nDCG(indx,:), valid_MRR(indx,:)] = Evaluate_ALL(matX_valid(user_probe,:), matX_train(user_probe,:), ...
                                                                                                                      G_matTheta(user_probe,:), G_matBeta, topK);
                fprintf('validation nDCG: %f\n', valid_nDCG(indx,1));
            else
                [valid_precision(indx,:), valid_recall(indx,:), valid_MRR(indx,:)] = Evaluate_PrecNRec(matX_valid(user_probe,:), matX_train(user_probe,:), ...
                                                                                                       G_matTheta(user_probe,:), G_matBeta, topK);
            end
            fprintf('validation precision: %f\n', valid_precision(indx,1));
            fprintf('validation recall: %f\n', valid_recall(indx,1));
            
            % Calculate the metrics on testing set
            fprintf('Testing ... \n');
            indx = itr / check_step;
            if usr_idx_len > 5000 && itm_idx_len > 20000
                user_probe = datasample(usr_idx, min(usr_idx_len, 5000), 'Replace', false);
            else
                user_probe = 1:length(usr_idx);
            end
            if strcmp(DATA, 'MovieLens100K') || strcmp(DATA, 'MovieLens1M') || strcmp(DATA, 'ML100KPos')
                [test_precision(indx,:), test_recall(indx,:), test_nDCG(indx,:), test_MRR(indx,:)] = Evaluate_ALL(matX_test(user_probe,:)+matX_valid(user_probe,:), matX_train(user_probe,:), ...
                                                                                                                  G_matTheta(user_probe,:), G_matBeta, topK);
                fprintf('testing nDCG: %f\n', test_nDCG(indx,1));
            else
                [test_precision(indx,:), test_recall(indx,:), test_MRR(indx,:)] = Evaluate_PrecNRec(matX_test(user_probe,:)+matX_valid(user_probe,:), matX_train(user_probe,:), ...
                                                                                                    G_matTheta(user_probe,:), G_matBeta, topK);
            end
            fprintf('testing precision: %f\n', test_precision(indx,1));
            fprintf('testing recall: %f\n', test_recall(indx,1));
            
            % Draw a consumption sample 
            range = min(N, 100);
            index = 30;
            tmp1 = G_matBeta(1:range,:) * G_matTheta(index,:)';
            tmp2 = G_matDelta(1:range,:) * G_matGamma(index,:)' / K;
            js_sparse = js_X_train(is_X_train == index);
            D_sparse = vec_matD_ui(is_X_train == index);
            js_sparse_range = js_sparse(js_sparse<=range);
            D_sparse_range = D_sparse(js_sparse<=range);
            tmp2(js_sparse_range) = D_sparse_range;
            plot(full([tmp1 tmp1.*tmp2 matX_train(index,1:range)' matX_test(index,1:range)']));
        end
    
        if itr >= MaxItr
            IsConverge = true;
        end               
    end
    
    if valid_poisson(end,1) > valid_poisson(end-1,1)
        Best_matTheta = G_matTheta;
        Best_matBeta = G_matBeta;
    end
    Best_matTheta = G_matTheta;
    Best_matBeta = G_matBeta;
    
    if strcmp(DATA, 'MovieLens100K') || strcmp(DATA, 'MovieLens1M') || strcmp(DATA, 'ML100KPos')
        [total_test_precision, total_test_recall, total_test_nDCG, total_test_MRR] = Evaluate_ALL(matX_test(usr_idx,:)+matX_valid(usr_idx,:), matX_train(usr_idx,:), ...
                                                                                                  Best_matTheta(usr_idx,:), Best_matBeta, topK);
        fprintf('testing nDCG: %f\n', total_test_nDCG);
    else
        [total_test_precision, total_test_recall, total_test_MRR] = Evaluate_PrecNRec(matX_test(usr_idx,:)+matX_valid(usr_idx,:), matX_train(usr_idx,:), ...
                                                                                      Best_matTheta(usr_idx,:), Best_matBeta, topK);
    end
end


save matPrecNRecall











