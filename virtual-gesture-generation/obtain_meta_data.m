% this file contains all the parameters required for BVP extraction
function meta_data = obtain_meta_data()
    % create a data structure for storing all meta data
    meta_data = containers.Map();

    %% Experiment Setup
    % no. of receiver
    meta_data('n_rx') = 4;
    % no. of antenna on each receiver
    meta_data('n_an') = 3;
    % no. of subcarriers
    meta_data('n_sc') = 30;
    % rotation matrix
    an = 7/180 * pi;
    assert(an >= 0 && an <= pi/4);
    rotation_matrix = [cos(an), sin(an); -sin(an), cos(an)];
    meta_data('rotate_mat') = rotation_matrix;
    meta_data('angle') = an;
    % position of TX
    TX_pos = [0, sqrt(2)];
    meta_data('TX_pos') = (rotation_matrix * TX_pos')';%[0 0];
    % position of RXs
    RX_pos = [sqrt(2), 0; ...
              sqrt(2)/2, sqrt(2)/2; ...
             -sqrt(2)/2, sqrt(2)/2; ...
             -sqrt(2), 0];
    meta_data('RX_pos') = (rotation_matrix * RX_pos')';
    
    % torso position
    meta_data('t_pos') = [0, 0];%[1 1];

    % minimum number of packages
    meta_data('n_packet_thresh') = 2000;
    
    %% Params for Doppler Frequency Spectrum
    meta_data('samp_rate') = 1000;
    meta_data('half_rate') = meta_data('samp_rate')/2;
    meta_data('uppe_orde') = 6;
    meta_data('uppe_stop') = 60;
    meta_data('lowe_orde') = 3;
    meta_data('lowe_stop') = 2;
    meta_data('n_freq') = 121;

    %% Params for Velocity Spectrum
    meta_data('v_max') = 2;
    meta_data('v_min') = -2;
    meta_data('n_bin') = 25;
    meta_data('v_resolution') = (meta_data('v_max') - meta_data('v_min'))/meta_data('n_bin');
    
    meta_data('wave_length') = 299792458 / 5.540e9;
    meta_data('segment_length') = 100;
    
    %% params for fmincon solver
    meta_data('MaxFunctionEvaluations') = 1e5;
    
end

