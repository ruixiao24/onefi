% COMPUTE VELOCITY DISTRIBUTION
% INPUT:
% - spfx_ges: prefix of gesture file

% OUTPUT:
% - velocity distribution: computed velocity distribution
function velocity_distribution = compute_velocity_distribution(spfx_ges)
    if nargin == 0
        spfx_ges = '1-1-1-1';
    end
    
    % obtain meta data
    meta_data = obtain_meta_data();
    n_rx = meta_data('n_rx');
    
    n_bins = meta_data('n_bin');
    % Segment Settings          
    l_seg = meta_data('segment_length');
    MaxFunctionEvaluations = meta_data('MaxFunctionEvaluations');
    disp(spfx_ges);
    
    % Generate Doppler Spectrum
    [doppler_spectrum, freq_bin] = compute_doppler_spectrum(spfx_ges, true);

    for nr = 1:n_rx
        if ~any(doppler_spectrum(nr,:,:), 'all')
	    fprintf("One RX is missing. Go to next file\n");
            return;
        end
    end
    
    % For Each Segment Do Mapping
    doppler_spectrum_max = max(max(max(doppler_spectrum,[],2),[],3));
    U_bound = repmat(3*doppler_spectrum_max, n_bins, n_bins);
    
    % SHAPE:[n_bin, n_bin, n_rx, 121]
    [mapping_temp, bezero] = obtain_mapping_matrix(freq_bin, meta_data);
    mapping = permute(mapping_temp, [2,3,1,4]);    
    
    % stores velocity spectrum
    n_timestamps = size(doppler_spectrum, 3);
    n_seg = floor(n_timestamps/l_seg);
    velocity_distribution = zeros(n_bins, n_bins, n_seg);
    
    % transform the doppler spectrum to execute for loop in parallel
    doppler_spectrum_seg = zeros(n_rx, size(doppler_spectrum, 2), l_seg, n_seg);
    for ns = 1:n_seg
        doppler_spectrum_seg(:, :, :, ns) = doppler_spectrum(:, :, (ns-1)*l_seg+1 : ns*l_seg);
    end

    % execute for loop in parallel
    parfor ns = 1:n_seg
    %for ns = 1:n_seg
        % Set-up fmincon Input
        doppler_specturm_seg_tgt_temp = squeeze(doppler_spectrum_seg(:, :, :, ns));
        doppler_spectrum_seg_tgt = mean(doppler_specturm_seg_tgt_temp, 3);
        
        % normalization Between Receivers(Compensate Path-Loss)
        for nr = 2:n_rx 
            if any(doppler_spectrum_seg_tgt(nr,:))
                doppler_spectrum_seg_tgt(nr,:) = doppler_spectrum_seg_tgt(nr,:)...
                    * sum(doppler_spectrum_seg_tgt(1,:))/sum(doppler_spectrum_seg_tgt(nr,:));
            end
        end
        
        % apply fmincon Solver (original)
        [P,floss,exitFlag,~] = fmincon(...
            @(P)DVM_target_func(P, mapping, bezero, doppler_spectrum_seg_tgt, meta_data),...
            zeros(n_bins,n_bins),...  % Initial Value
            [],[],...       % Linear Inequality Constraints
            [],[],...       % Linear Equality Constraints
            zeros(n_bins,n_bins),...  % Lower Bound
            U_bound,...     % Upper Bound
            [],...          % Non-linear Constraints
            optimoptions('fmincon',...
            'MaxFunctionEvaluations', MaxFunctionEvaluations,...
            'FiniteDifferenceStepSize', 1e-4));	% Options
        P(bezero == 1) = 0;
        % store result in the matrix 
        velocity_distribution(:,:,ns) = P;
        fprintf("segment: %d exitFlag: %d loss: %.2f\n", ns, exitFlag, floss);
    end
    
    
end

% Input:
% P: velocity distribution
% mapping: map velocity distribution to doppler frequency component
function loss = DVM_target_func(P, mapping, bezero, ds_gt, meta_data)
   
    % obtain meta data
    n_rx = meta_data('n_rx');
    
    % Initialize Variable
    n_freq = size(ds_gt,2);
    
    P(bezero == 1) = 0;
    % Construct Approximation Doppler Spectrum
    ds_pred = estimate_doppler(P, mapping);
    
    % EMD Distance
    loss = 0;
    for nr = 1:n_rx
        if(any(ds_gt(nr,:)))
            loss = loss + distance_metric(ds_pred(nr,:), ds_gt(nr,:),'EMD', n_freq);
        end
    end
    
end

% Construct Approximation Doppler Spectrum
function doppler_spectrum_approximate = estimate_doppler(P, mapping, n_rx, n_freq)
    if nargin == 2
        n_rx = size(mapping, 3);
        n_freq = size(mapping, 4);
    end
    
    P_extent = repmat(P, 1, 1, n_rx, n_freq);
    doppler_spectrum_approximate = squeeze(sum(P_extent .* mapping, [1,2]));
end

function distance = distance_metric(a, b, type, dim)
    if nargin == 2
        type = 'EMD';
        dim = 121;
    end
    
    if strcmp(type, 'EMD')
        distance = sum(abs((a-b) * triu(ones(dim,dim),0)));
    end
end


