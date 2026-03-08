%% plot_inference_latency.m
%  Plot (b): Inference Latency — Grouped bar chart
%  Centralized: single solid bar (computation only)
%  Distributed: stacked bar (computation + communication overhead)
%    Computation portion  = centralized latency (same CNN FLOPs)
%    Communication portion = distributed_total - centralized
%
%  Reads: results/csv/inference_latency.csv
%  Format: dataset, centralized_ms, distributed_ms
%
%  Compatible with both MATLAB and GNU Octave.

clear; clc; close all;
graphics_toolkit('gnuplot');
setenv('GNUTERM', 'dumb');
warning('off', 'all');

csv_file = fullfile('..', 'results', 'csv', 'inference_latency.csv');

if ~exist(csv_file, 'file')
    error('CSV file not found: %s\nRun parse_results.py first.', csv_file);
end

fid = fopen(csv_file, 'r');
header = fgetl(fid);
raw = textscan(fid, '%s%f%f', 'Delimiter', ',');
fclose(fid);

ds_labels = raw{1};
cent_ms   = raw{2};
dist_ms   = raw{3};
n = numel(ds_labels);

% Communication = pipeline latency - computation
comp_ms = cent_ms;                % computation same as centralized
comm_ms = dist_ms - cent_ms;      % IPC / shared-memory overhead

for i = 1:n
    ds_labels{i} = strrep(ds_labels{i}, '_', '\_');
end

hf = figure();
set(hf, 'PaperPositionMode', 'auto', 'Position', [150, 150, 900, 600]);

% --- Centralized bars (left group) ---
bar_w = 0.3;
x = (1:n)';
hold on;

% Centralized: solid blue bar
for i = 1:n
    fill([x(i)-bar_w x(i) x(i) x(i)-bar_w], [0 0 cent_ms(i) cent_ms(i)], ...
         [0.2 0.4 0.8], 'EdgeColor', 'k', 'LineWidth', 0.5);
end

% Distributed: stacked bar (computation bottom, communication top)
for i = 1:n
    % Computation portion (orange)
    fill([x(i) x(i)+bar_w x(i)+bar_w x(i)], [0 0 comp_ms(i) comp_ms(i)], ...
         [1.0 0.6 0.2], 'EdgeColor', 'k', 'LineWidth', 0.5);
    % Communication portion (dark red) stacked on top
    fill([x(i) x(i)+bar_w x(i)+bar_w x(i)], ...
         [comp_ms(i) comp_ms(i) comp_ms(i)+comm_ms(i) comp_ms(i)+comm_ms(i)], ...
         [0.75 0.15 0.05], 'EdgeColor', 'k', 'LineWidth', 0.5);
    % Dashed line at the cut between computation and communication
    plot([x(i) x(i)+bar_w], [comp_ms(i) comp_ms(i)], 'w--', 'LineWidth', 1.5);
end

hold off;

set(gca, 'XTick', x, 'XTickLabel', ds_labels, 'FontSize', 12);
xlim([0.5, n+0.5+bar_w]);
xlabel('Dataset', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Inference Latency (ms/sample)', 'FontSize', 13, 'FontWeight', 'bold');
title('Inference Latency: Centralized vs Distributed', ...
      'FontSize', 15, 'FontWeight', 'bold');

% Dummy invisible plots for legend entries
hold on;
h1 = fill([0 0 0 0], [0 0 0 0], [0.2 0.4 0.8]);
h2 = fill([0 0 0 0], [0 0 0 0], [1.0 0.6 0.2]);
h3 = fill([0 0 0 0], [0 0 0 0], [0.75 0.15 0.05]);
legend([h1 h2 h3], {'Centralized (Computation)', ...
                    'Distributed - Computation', ...
                    'Distributed - Communication (IPC)'}, ...
       'Location', 'northwest', 'FontSize', 10);
hold off;
grid on;

% Value labels
for i = 1:n
    % Centralized total
    text(x(i)-bar_w/2, cent_ms(i)+0.2, sprintf('%.2f', cent_ms(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', 9, 'FontWeight', 'bold');
    % Distributed: computation portion
    text(x(i)+bar_w/2, comp_ms(i)/2, sprintf('%.2f', comp_ms(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
         'FontSize', 8, 'FontWeight', 'bold', 'Color', 'w');
    % Distributed: communication portion
    text(x(i)+bar_w/2, comp_ms(i)+comm_ms(i)/2, sprintf('%.2f', comm_ms(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
         'FontSize', 8, 'FontWeight', 'bold', 'Color', 'w');
    % Distributed: total on top
    text(x(i)+bar_w/2, dist_ms(i)+0.2, sprintf('%.2f', dist_ms(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', 9, 'FontWeight', 'bold');
end

out_png = fullfile('..', 'results', 'inference_latency.png');
print(hf, out_png, '-dpng', '-r200');
close(hf);
fprintf('Saved: %s\n', out_png);
