%% plot_memory_consumption.m
%  Plot (c): Memory Consumption — Grouped bar chart
%  For Centralized: total memory (RSS of single process)
%  For Distributed: average memory per device (across all 7 devices)
%
%  Reads: results/csv/memory_consumption.csv
%  Format: dataset, centralized_kb, distributed_avg_kb, distributed_std_kb
%
%  Compatible with both MATLAB and GNU Octave.

clear; clc; close all;
graphics_toolkit('gnuplot');
setenv('GNUTERM', 'dumb');
warning('off', 'all');

csv_file = fullfile('..', 'results', 'csv', 'memory_consumption.csv');

if ~exist(csv_file, 'file')
    error('CSV file not found: %s\nRun parse_results.py first.', csv_file);
end

% Read CSV (skip header): dataset(text), centralized_kb, dist_avg_kb, dist_std_kb
fid = fopen(csv_file, 'r');
header = fgetl(fid);
raw = textscan(fid, '%s%f%f%f', 'Delimiter', ',');
fclose(fid);

ds_labels  = raw{1};
cent_kb    = raw{2};
dist_avg   = raw{3};
dist_std   = raw{4};
n = numel(ds_labels);

for i = 1:n
    ds_labels{i} = strrep(ds_labels{i}, '_', '\_');
end

data = [cent_kb, dist_avg];

hf = figure();
set(hf, 'PaperPositionMode', 'auto', 'Position', [200, 200, 800, 550]);

b = bar(data, 'grouped');
set(b(1), 'FaceColor', [0.2, 0.4, 0.8]);
set(b(2), 'FaceColor', [0.85, 0.25, 0.1]);

hold on;

% Error bars for distributed bars (Octave-compatible syntax)
x_err = (1:n)' + 0.14;
errorbar(x_err, dist_avg, dist_std, 'k');

% Value labels
for i = 1:n
    text(i - 0.14, cent_kb(i) + 30, sprintf('%.0f', cent_kb(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', 10, 'FontWeight', 'bold');
    text(i + 0.14, dist_avg(i) + dist_std(i) + 8, sprintf('%.1f', dist_avg(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', 10, 'FontWeight', 'bold');
end

hold off;

set(gca, 'XTickLabel', ds_labels, 'FontSize', 12);
xlabel('Dataset', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Memory Consumption (KB)', 'FontSize', 13, 'FontWeight', 'bold');
title('Memory Consumption: Centralized vs Distributed', ...
      'FontSize', 15, 'FontWeight', 'bold');
legend({'Centralized (Single Device)', ...
        'Distributed (Avg per Device +/- Std)'}, ...
       'Location', 'northwest', 'FontSize', 11);
grid on;

out_png = fullfile('..', 'results', 'memory_consumption.png');
print(hf, out_png, '-dpng', '-r200');
close(hf);
fprintf('Saved: %s\n', out_png);
