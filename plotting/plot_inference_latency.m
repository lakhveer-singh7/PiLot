%% plot_inference_latency.m
%  Plot (b): Inference Latency — Grouped bar chart
%  For Centralized: computation latency only
%  For Distributed: computation + communication (full pipeline latency)
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

% Read CSV (skip header): col1=dataset(text), col2=centralized_ms, col3=distributed_ms
fid = fopen(csv_file, 'r');
header = fgetl(fid);  % skip header
raw = textscan(fid, '%s%f%f', 'Delimiter', ',');
fclose(fid);

ds_labels = raw{1};
cent_ms   = raw{2};
dist_ms   = raw{3};
n = numel(ds_labels);

for i = 1:n
    ds_labels{i} = strrep(ds_labels{i}, '_', '\_');
end

data = [cent_ms, dist_ms];

hf = figure();
set(hf, 'PaperPositionMode', 'auto', 'Position', [150, 150, 800, 550]);

b = bar(data, 'grouped');
set(b(1), 'FaceColor', [0.2, 0.4, 0.8]);   % Blue for centralized
set(b(2), 'FaceColor', [0.85, 0.25, 0.1]);  % Red for distributed

set(gca, 'XTickLabel', ds_labels, 'FontSize', 12);
xlabel('Dataset', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Inference Latency (ms/sample)', 'FontSize', 13, 'FontWeight', 'bold');
title('Inference Latency: Centralized vs Distributed', ...
      'FontSize', 15, 'FontWeight', 'bold');
legend({'Centralized (Computation only)', ...
        'Distributed (Computation + Communication)'}, ...
       'Location', 'northwest', 'FontSize', 11);
grid on;

% Value labels on bars (Octave-compatible positioning)
for i = 1:n
    % Centralized bar (left of center)
    text(i - 0.14, cent_ms(i) + 0.15, sprintf('%.2f', cent_ms(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', 10, 'FontWeight', 'bold');
    % Distributed bar (right of center)
    text(i + 0.14, dist_ms(i) + 0.15, sprintf('%.2f', dist_ms(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', 10, 'FontWeight', 'bold');
end

out_png = fullfile('..', 'results', 'inference_latency.png');
print(hf, out_png, '-dpng', '-r200');
close(hf);
fprintf('Saved: %s\n', out_png);
