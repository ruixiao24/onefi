 function [data_path, vd_path] = config(is_save_path)
    if nargin == 0
        is_save_path = false;
    end
    src_path = '';
    data_path = fullfile(src_path, 'data');
    vd_path = fullfile(src_path, 'data');
    
    if is_save_path
        % Determine where your m-file's folder is.
        curr_folder = fileparts(which('config.m'));
        % Add that folder plus all subfolders to the path.
        addpath(genpath(curr_folder));
    end
end
