function [trct_data, timestamp] = truncate_csi_data(filename)
    if nargin == 0
        filename = '1-1-1-1-r1.dat';
    end
    
    src_path = config();
    meta_data = obtain_meta_data();
    
    % no. of antennas
    n_an = meta_data('n_an');
    % no. of subcarriers
    n_sc = meta_data('n_sc');
    % threshold of no. of packages
    thresh = meta_data('n_packet_thresh');  
    
    % read csi data
    data_path = fullfile(src_path, filename);
    csi_trace = read_bf_file(data_path);
    
    % stores trucated csi data
    % SHAPE: [truncation threshold, no. of receiver * no. of subcarriers]
    trct_data = zeros(thresh, n_an * n_sc);
    timestamp = zeros(thresh, 1);

    % ignore trials with less than 2990 packages
    if numel(csi_trace) < thresh
        disp('ERROR');
        return
    end
    
    for k = 1:thresh
        csi_entry = csi_trace{k}; % for the k_{th} packet
        
        % obtain scaled csi
        csi_temp = get_scaled_csi(csi_entry);
        assert(all(size(csi_temp) == [1 3 30]));
        % obtain data only from the tx-th antenna on tx1
        csi_all = squeeze(csi_temp).'; % estimate channel matrix Hexp-figu

        csi = [csi_all(:,1); csi_all(:,2); csi_all(:, 3)].'; % select CSI for one antenna pair

        timestamp(k) = csi_entry.timestamp_low;
        trct_data(k,:) = csi;
    end
end