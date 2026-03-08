%% plot_multi_inference_latency.m
%  Multi-device comparison: Inference Latency (computation + communication)
%  One figure per dataset with 6 bars (2 per config × 3 configs).
%
%  Reads: results/csv_multi/inference_latency.csv
%  Format: dataset,num_devices,pipeline_latency_ms,computation_ms,communication_ms
%
%  Compatible with GNU Octave (headless gnuplot).

clear; clc; close all;
graphics_toolkit('gnuplot');
setenv('GNUTERM', 'dumb');
warning('off', 'all');

datasets = {'Cricket_X', 'ECG5000', 'FaceAll'};
ndevs    = [7, 9, 12];
labels   = {'7 Devices', '9 Devices', '12 Devices'};

comp_color = [1.0 0.6 0.2];     % orange
comm_color = [0.75 0.15 0.05];  % dark red

csv_file = fullfile('..', 'results', 'csv_multi', 'inference_latency.csv');
out_dir  = fullfile('..', 'results', 'multi_device');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

if ~exist(csv_file, 'file')
    error('CSV not found: %s\nRun parse_multi_device_results.py first.', csv_file);
end

%% Read CSV
fid = fopen(csv_file, 'r');
hdr = fgetl(fid);
raw = textscan(fid, '%s%d%f%f%f', 'Delimiter', ',');
fclose(fid);
r_ds   = raw{1};
r_ndev = double(raw{2});
r_pipe = raw{3};
r_comp = raw{4};
r_comm = raw{5};

for d = 1:length(datasets)
    ds = datasets{d};

    hf = figure();
    set(hf, 'PaperPositionMode', 'auto', 'Position', [100 100 900 600]);
    hold on;

    bar_w = 0.35;
    x_ticks = [];
    x_labels_plot = {};

    for k = 1:length(ndevs)
        nd = ndevs(k);
        idx = find(strcmp(r_ds, ds) & (r_ndev == nd));
        if isempty(idx); continue; end
        comp = r_comp(idx(1));
        comm = r_comm(idx(1));

        base_x = (k-1) * 2.5 + 1;
        x1 = base_x;
        x2 = base_x + bar_w + 0.05;

        % Computation bar
        fill([x1-bar_w/2 x1+bar_w/2 x1+bar_w/2 x1-bar_w/2], ...
             [0 0 comp comp], comp_color, 'EdgeColor', 'k', 'LineWidth', 0.5);
        text(x1, comp + 0.15, sprintf('%.2f', comp), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
             'FontSize', 9, 'FontWeight', 'bold');

        % Communication bar
        fill([x2-bar_w/2 x2+bar_w/2 x2+bar_w/2 x2-bar_w/2], ...
             [0 0 comm comm], comm_color, 'EdgeColor', 'k', 'LineWidth', 0.5);
        text(x2, comm + 0.15, sprintf('%.2f', comm), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
             'FontSize', 9, 'FontWeight', 'bold');

        group_center = (x1 + x2) / 2;
        x_ticks(end+1) = group_center;
        x_labels_plot{end+1} = labels{k};
    end

    hold off;

    % Legend
    hold on;
    h1 = fill([0 0 0 0], [0 0 0 0], comp_color, 'EdgeColor', 'k');
    h2 = fill([0 0 0 0], [0 0 0 0], comm_color, 'EdgeColor', 'k');
    legend([h1 h2], {'Computation', 'Communication (IPC)'}, ...
           'Location', 'northeast', 'FontSize', 10);
    hold off;

    set(gca, 'XTick', x_ticks, 'XTickLabel', x_labels_plot, 'FontSize', 12);
    xlabel('Configuration', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Inference Latency (ms)', 'FontSize', 13, 'FontWeight', 'bold');
    title(sprintf('Inference Latency — %s  (Distributed)', ...
          strrep(ds, '_', '\_')), 'FontSize', 15, 'FontWeight', 'bold');
    grid on;

    out_png = fullfile(out_dir, sprintf('inference_latency_%s.png', ds));
    print(hf, out_png, '-dpng', '-r200');
    close(hf);
    fprintf('Saved: %s\n', out_png);
end
