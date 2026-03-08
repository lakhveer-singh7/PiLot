%% plot_multi_accuracy_vs_time.m
%  Multi-device comparison: Accuracy vs Time for 7, 9, 12 devices.
%  One figure per dataset, 3 curves per figure.
%
%  Reads: results/csv_multi/accuracy_vs_time_<dataset>.csv
%  Format: num_devices,epoch,timespan_s,train_acc,test_acc
%
%  Compatible with GNU Octave (headless gnuplot).

clear; clc; close all;
graphics_toolkit('gnuplot');
setenv('GNUTERM', 'dumb');
warning('off', 'all');

datasets = {'Cricket_X', 'ECG5000', 'FaceAll'};
ndevs    = [7, 9, 12];
colors   = {[0.2 0.4 0.8], [1.0 0.6 0.2], [0.2 0.67 0.33]};   % blue, orange, green
markers  = {'o', 's', 'd'};
labels   = {'7 Devices', '9 Devices', '12 Devices'};

csv_dir = fullfile('..', 'results', 'csv_multi');
out_dir = fullfile('..', 'results', 'multi_device');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

for d = 1:length(datasets)
    ds = datasets{d};
    csv_file = fullfile(csv_dir, sprintf('accuracy_vs_time_%s.csv', ds));
    if ~exist(csv_file, 'file')
        fprintf('SKIP %s — CSV not found: %s\n', ds, csv_file);
        continue;
    end

    %% Read CSV — manual parsing for mixed int/float columns
    fid = fopen(csv_file, 'r');
    hdr = fgetl(fid);   % skip header
    data = textscan(fid, '%d%d%f%f%f', 'Delimiter', ',');
    fclose(fid);
    col_ndev = double(data{1});
    col_epoch = double(data{2});
    col_time  = data{3};
    col_train = data{4};
    col_test  = data{5};

    hf = figure();
    set(hf, 'PaperPositionMode', 'auto', 'Position', [100 100 900 600]);
    hold on;
    leg_handles = [];
    leg_labels  = {};

    for k = 1:length(ndevs)
        nd = ndevs(k);
        idx = (col_ndev == nd);
        if ~any(idx); continue; end
        t   = col_time(idx);
        acc = col_test(idx);
        h = plot(t, acc, ['-' markers{k}], 'Color', colors{k}, ...
                 'LineWidth', 1.8, 'MarkerSize', 4, 'MarkerFaceColor', colors{k});
        leg_handles(end+1) = h;
        leg_labels{end+1}  = labels{k};
    end
    hold off;

    xlabel('Training Time (s)', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Test Accuracy (%)', 'FontSize', 13, 'FontWeight', 'bold');
    title(sprintf('Accuracy vs Time — %s  (Distributed)', ...
          strrep(ds, '_', '\_')), 'FontSize', 15, 'FontWeight', 'bold');
    legend(leg_handles, leg_labels, 'Location', 'southeast', 'FontSize', 11);
    grid on;
    set(gca, 'FontSize', 12);

    out_png = fullfile(out_dir, sprintf('accuracy_vs_time_%s.png', ds));
    print(hf, out_png, '-dpng', '-r200');
    close(hf);
    fprintf('Saved: %s\n', out_png);
end
