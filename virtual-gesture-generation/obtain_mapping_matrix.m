function [mapping, bezero] = obtain_mapping_matrix(freq_bin, meta_data)

    wave_length = meta_data('wave_length');
    n_rx = meta_data('n_rx');
    v_max = meta_data('v_max');
    % v_min = meta_data('v_min');
    n_bin = meta_data('n_bin');
    M = n_bin;
    velocity_bin = ((1:M) - M/2) / (M/2) * v_max;

    F = size(freq_bin, 2);
    M = size(velocity_bin, 2);
    freq_min = min(freq_bin);
    freq_max = max(freq_bin);
    
    A = get_A_matrix(meta_data)/wave_length;
    
    mapping = zeros(n_rx, M, M, F);
    bezero = zeros(M, M);
    % For Each Link
    for nr = 1:n_rx
        for i = 1:M
            for j = 1:M
                plcr_hz = round(A(nr,:) * [velocity_bin(i) velocity_bin(j)]');
                if plcr_hz > freq_max || plcr_hz < freq_min
                    bezero(i, j) = 1;
                    continue;
                end
                idx = plcr_hz + 1 - freq_min;
                mapping(nr,i,j,idx) = 1;
            end
        end
    end
end

function A = get_A_matrix(meta_data)
    
    tx_pos = meta_data('TX_pos');
    rx_pos = meta_data('RX_pos');
    n_rx = meta_data('n_rx');
    t_pos = meta_data('t_pos');
    
    if n_rx > size(rx_pos,1)
        error('Error Rx Count!')
    end
    A = zeros(n_rx,2);
    
    for nr = 1:n_rx
        dis_torso_tx = norm(t_pos-tx_pos);
        dis_torso_rx = norm(t_pos-rx_pos(nr, :));
        A(nr,:) = (t_pos-tx_pos)/dis_torso_tx + (t_pos-rx_pos(nr, :))/dis_torso_rx;
    end
end