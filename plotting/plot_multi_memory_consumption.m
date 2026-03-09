%% plot_multi_memory_consumption.m
%  Multi-device comparison: Memory Consumption
%  One figure per dataset with 9 bars (weight/optimizer/buffer × 3 configs).
%
%  Reads: results/csv_multi/memory_consumption.csv
%  Format: dataset,num_devices,weight_avg_kb,weight_std_kb,
%          optimizer_avg_kb,optimizer_std_kb,buffer_avg_kb,buffer_std_kb
%
%  Compatible with GNU Octave (headless gnuplot).

clear; clc; close all;
graphics_toolkit('gnuplot');
setenv('GNUTERM', 'dumb');
warning('off', 'all');

datasets = {'Cricket_X', 'ECG5000', 'FaceAll'};
ndevs    = [7, 9, 12];
labels   = {'7 Devices', '9 Devices', '12 Devices'};

w_color = [0.2 0.4 0.8];    % blue
o_color = [1.0 0.6 0.2];    % orange
b_color = [0.2 0.67 0.33];  % green

csv_file = fullfile('..', 'results', 'csv_multi', 'memory_consumption.csv');
out_dir  = fullfile('..', 'results', 'multi_device');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

if ~exist(csv_file, 'file')
    error('CSV not found: %s\nRun parse_multi_device_results.py first.', csv_file);
end

%% Read CSV  (9 columns — skip the last total_avg_kb with %*f)
fid = fopen(csv_file, 'r');
hdr = fgetl(fid);
raw = textscan(fid, '%s%d%f%f%f%f%f%f%*f', 'Delimiter', ',');
fclose(fid);
r_ds   = raw{1};
r_ndev = double(raw{2});
r_wavg = raw{3};  r_wstd = raw{4};
r_oavg = raw{5};  r_ostd = raw{6};
r_bavg = raw{7};  r_bstd = raw{8};

for d = 1:length(datasets)
    ds = datasets{d};

    hf = figure();
    set(hf, 'PaperPositionMode', 'auto', 'Position', [100 100 900 600]);
    hold on;

    bar_w = 0.22;
    x_ticks = [];
    x_labels_plot = {};

    for k = 1:length(ndevs)
        nd = ndevs(k);
        idx = find(strcmp(r_ds, ds) & (r_ndev == nd));
        if isempty(idx); continue; end

        w_avg = r_wavg(idx(1));  w_std = r_wstd(idx(1));
        o_avg = r_oavg(idx(1));  o_std = r_ostd(idx(1));
        b_avg = r_bavg(idx(1));  b_std = r_bstd(idx(1));

        base_x = (k-1) * 3.0 + 1.5;
        x_w = base_x - bar_w;
        x_o = base_x;
        x_b = base_x + bar_w;

        % Weight bar
        fill([x_w-bar_w/2 x_w+bar_w/2 x_w+bar_w/2 x_w-bar_w/2], ...
             [0 0 w_avg w_avg], w_color, 'EdgeColor', 'k', 'LineWidth', 0.5);
        % Error bar
        plot([x_w x_w], [w_avg-w_std w_avg+w_std], 'k-', 'LineWidth', 1.5);
        plot([x_w-0.06 x_w+0.06], [w_avg+w_std w_avg+w_std], 'k-', 'LineWidth', 1.5);
        plot([x_w-0.06 x_w+0.06], [w_avg-w_std w_avg-w_std], 'k-', 'LineWidth', 1.5);
        text(x_w, w_avg + w_std + 0.3, sprintf('%.1f', w_avg), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
             'FontSize', 8.5, 'FontWeight', 'bold');

        % Optimizer bar
        fill([x_o-bar_w/2 x_o+bar_w/2 x_o+bar_w/2 x_o-bar_w/2], ...
             [0 0 o_avg o_avg], o_color, 'EdgeColor', 'k', 'LineWidth', 0.5);
        plot([x_o x_o], [o_avg-o_std o_avg+o_std], 'k-', 'LineWidth', 1.5);
        plot([x_o-0.06 x_o+0.06], [o_avg+o_std o_avg+o_std], 'k-', 'LineWidth', 1.5);
        plot([x_o-0.06 x_o+0.06], [o_avg-o_std o_avg-o_std], 'k-', 'LineWidth', 1.5);
        text(x_o, o_avg + o_std + 0.3, sprintf('%.1f', o_avg), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
             'FontSize', 8.5, 'FontWeight', 'bold');

        % Buffer bar
        fill([x_b-bar_w/2 x_b+bar_w/2 x_b+bar_w/2 x_b-bar_w/2], ...
             [0 0 b_avg b_avg], b_color, 'EdgeColor', 'k', 'LineWidth', 0.5);
        plot([x_b x_b], [b_avg-b_std b_avg+b_std], 'k-', 'LineWidth', 1.5);
        plot([x_b-0.06 x_b+0.06], [b_avg+b_std b_avg+b_std], 'k-', 'LineWidth', 1.5);
        plot([x_b-0.06 x_b+0.06], [b_avg-b_std b_avg-b_std], 'k-', 'LineWidth', 1.5);
        text(x_b, b_avg + b_std + 0.3, sprintf('%.1f', b_avg), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
             'FontSize', 8.5, 'FontWeight', 'bold');

        x_ticks(end+1) = base_x;
        x_labels_plot{end+1} = labels{k};
    end

    hold off;

    % Legend
    hold on;
    h1 = fill([0 0 0 0], [0 0 0 0], w_color, 'EdgeColor', 'k');
    h2 = fill([0 0 0 0], [0 0 0 0], o_color, 'EdgeColor', 'k');
    h3 = fill([0 0 0 0], [0 0 0 0], b_color, 'EdgeColor', 'k');
    legend([h1 h2 h3], {'Weights (avg)', 'Optimizer (avg)', 'Buffers (avg)'}, ...
           'Location', 'northeast', 'FontSize', 10);
    hold off;

    set(gca, 'XTick', x_ticks, 'XTickLabel', x_labels_plot, 'FontSize', 12);
    xlim([0, max(x_ticks)+2]);
    ylim([0, inf]);
    xlabel('Configuration', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Avg Memory per Device (KB)', 'FontSize', 13, 'FontWeight', 'bold');
    title(sprintf('Memory Consumption — %s  (Distributed)', ...
          strrep(ds, '_', '\_')), 'FontSize', 15, 'FontWeight', 'bold');
    grid on;

    out_png = fullfile(out_dir, sprintf('memory_consumption_%s.png', ds));
    print(hf, out_png, '-dpng', '-r200');
    close(hf);
    fprintf('Saved: %s\n', out_png);
end
