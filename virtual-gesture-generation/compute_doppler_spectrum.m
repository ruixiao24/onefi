% Compute doppler spectrum
% INPUT:
% spfx_ges - prefix of filename
% shift - shift frequency order

% OUTPUT:
% doppler_spectrum - computed doppler spectrum
% freq_bin - frequency bin

function [doppler_spectrum, freq_bin] = compute_doppler_spectrum(spfx_ges, shift)
    if nargin == 0
        spfx_ges = '1-1-1-1';
        shift = true;
    end
    
    if nargin == 2
        shift = false;
    end
    
    % spfx_ges must be char type
    assert(isa(spfx_ges, 'char'));
    
    % obtain meta data
    meta_data = obtain_meta_data();
    % no. of receiver
    n_rx = meta_data('n_rx');
    % no. of antenna on each receiver
    n_an = meta_data('n_an');
    % sampling rate
    samp_rate = meta_data('samp_rate');
    half_rate = meta_data('half_rate');
    uppe_orde = meta_data('uppe_orde');
    uppe_stop = meta_data('uppe_stop');
    lowe_orde = meta_data('lowe_orde');
    lowe_stop = meta_data('lowe_stop');
    
    % compute params
    [lu,ld] = butter(uppe_orde,uppe_stop/half_rate,'low');
    [hu,hd] = butter(lowe_orde,lowe_stop/half_rate,'high');
    freq_bins_unwrap = [0:samp_rate/2-1 -samp_rate/2:-1]'/samp_rate;
    freq_lpf_sele = freq_bins_unwrap <= uppe_stop / samp_rate & freq_bins_unwrap >= -uppe_stop / samp_rate;
    freq_lpf_positive_max = sum(freq_lpf_sele(2:length(freq_lpf_sele)/2));
    freq_lpf_negative_min = sum(freq_lpf_sele(length(freq_lpf_sele)/2:end));
    
    % Doppler Spectrum For Each Antenna
    [csi_data, ~] = truncate_csi_data([spfx_ges, '-r', num2str(1), '.dat']);
    doppler_spectrum = zeros(n_rx, 1+freq_lpf_positive_max + freq_lpf_negative_min,...
        floor(size(csi_data, 1)));
    
    for nr = 1:n_rx
        spth = [spfx_ges, '-r', num2str(nr), '.dat'];
        try
            [csi_data, ~] = truncate_csi_data(spth);
        catch err
            disp(err)
            continue
        end
        csi_data = csi_data(round(1:1:size(csi_data,1)),:);
        
        % Select Antenna Pair[WiDance]
        csi_mean = mean(abs(csi_data));
        csi_var = sqrt(var(abs(csi_data)));
        csi_mean_var_ratio = csi_mean./csi_var;
        [~,idx] = max(mean(reshape(csi_mean_var_ratio,[30 n_an]),1));
        %     idx = 1;
        csi_data_ref = repmat(csi_data(:,(idx-1)*30+1:idx*30), 1, n_an);
        
        % Amp Adjust[IndoTrack]
        csi_data_adj = zeros(size(csi_data));
        csi_data_ref_adj = zeros(size(csi_data_ref));
        alpha_sum = 0;
        for jj = 1:30*n_an
            amp = abs(csi_data(:,jj));
            alpha = min(amp(amp~=0));
            if isempty(alpha)
                alpha = 0;
            end
            alpha_sum = alpha_sum + alpha;
            csi_data_adj(:,jj) = abs(abs(csi_data(:,jj))-alpha).*exp(1j*angle(csi_data(:,jj)));
        end
        beta = 1000*alpha_sum/(30*n_an);
        for jj = 1:30*n_an
            csi_data_ref_adj(:,jj) = (abs(csi_data_ref(:,jj))+beta).*exp(1j*angle(csi_data_ref(:,jj)));
        end
        
        % Conj Mult
        conj_mult = csi_data_adj .* conj(csi_data_ref_adj);
        conj_mult = [conj_mult(:,1:30*(idx - 1)) conj_mult(:,30*idx+1:90)];
        
        % Filter Out Static Component & High Frequency Component
        for jj = 1:size(conj_mult, 2)
            conj_mult(:,jj) = filter(lu, ld, conj_mult(:,jj));
            conj_mult(:,jj) = filter(hu, hd, conj_mult(:,jj));
        end
        
        % PCA analysis.
        pca_coef = pca(conj_mult);
        conj_mult_pca = conj_mult * pca_coef(:,1);
        
        % STFT
        time_instance = 1:length(conj_mult_pca);
        window_size = round(samp_rate/4+1);
        if(~mod(window_size,2))
            window_size = window_size + 1;
        end
        freq_time_prof_allfreq = tfrsp(conj_mult_pca, time_instance,...
            samp_rate, tftb_window(window_size, 'gauss'));
        
        % Select Concerned Freq
        freq_time_prof = freq_time_prof_allfreq(freq_lpf_sele, :);
        % Spectrum Normalization By Sum For Each Snapshot
        freq_time_prof = abs(freq_time_prof) ./ repmat(sum(abs(freq_time_prof),1),...
            size(freq_time_prof,1), 1);
        
        % Frequency Bin(Corresponding to FFT Results)
        freq_bin = [0:freq_lpf_positive_max -1*freq_lpf_negative_min:-1];
        
        % Store Doppler Velocity Spectrum
        if(size(freq_time_prof,2) >= size(doppler_spectrum,3))
            doppler_spectrum(nr,:,:) = freq_time_prof(:,1:size(doppler_spectrum,3));
        else
            doppler_spectrum(nr,:,:) = [freq_time_prof zeros(size(doppler_spectrum,2),...
                size(doppler_spectrum,3) - size(freq_time_prof,2))];
        end
    end
    
    % shift doppler frequency specturm
    if shift
        % Cyclic Doppler Spectrum According To frequency bin
        [~,idx] = max(freq_bin);
        circ_len = length(freq_bin) - idx;
        doppler_spectrum = circshift(doppler_spectrum, [0 circ_len 0]);
    end
end