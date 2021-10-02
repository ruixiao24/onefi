% Construct Approximation Doppler Spectrum
function virtual_gesture = create_virtual_gesture(spfx_ges, mapping, direction, meta_data)
    if nargin == 0
        spfx_ges = '1-1-1-1';

        meta_data = obtain_meta_data();
        new_hpos = (meta_data('rotate_mat') * [0, 0]')';
        meta_data('TX_pos') = meta_data('TX_pos')-new_hpos;
        meta_data('RX_pos') = meta_data('RX_pos')-new_hpos;
        freq_bin = -60:60;
        % SHAPE:[n_bin, n_bin, n_rx, 121]
        mapping = permute(obtain_mapping_matrix(freq_bin, meta_data), [2,3,1,4]);
        
        % rotation angle in degree
        direction = -45;
    end
    
    % read computed velocity distribution
    [~, vd_path] = config();
    mat = load(fullfile(vd_path, [spfx_ges '.mat']));
    vd = mat.vd;
    vd = imrotate(vd, direction, 'nearest', 'crop');
    % obtain meta data
    n_rx = meta_data('n_rx');
    n_freq = meta_data('n_freq');
    n_timestamps = meta_data('n_packet_thresh');
    l_seg = meta_data('segment_length');
    n_seg = floor(n_timestamps/l_seg);
    
    % stores reconstructed doppler frequency spectrum
    virtual_gesture = zeros(n_rx, n_freq, n_seg);
    parfor ns = 1:n_seg
        P = squeeze(vd(:,:,ns));
        P_extent = repmat(P, 1, 1, n_rx, n_freq);
        doppler_spectrum = squeeze(sum(P_extent .* mapping, [1,2]));
        % lowpass function filters each column independently
        virtual_gesture(:,:,ns) = lowpass(doppler_spectrum', 0.18)';
    end
    virtual_gesture(virtual_gesture<0) = 0;
    
end