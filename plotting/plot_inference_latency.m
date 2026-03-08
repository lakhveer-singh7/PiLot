%% plot_inference_latency.m
%  Plot (b): Inference Latency — Grouped bar chart
%  For Centralized: computation latency only
%  For Distributed: computation + communication (full pipeline latency)
%
%  Reads: results/csv/inference_latency.csv
%  Format: dataset, centralized_ms, distributed_ms

clear; clc; close all;

csv_file = fullfile('..', 'results', 'csv', 'inference_latency.csv');

if ~isfile(csv_file)
    error('CSV file not found: %s\nRun parse_results.py first.', csv_file);
end

T = readtable(csv_file);
n = height(T);

% Prepare data matrix: rows = datasets, cols = [centralized, distributed]
data = [T.centralized_ms, T.distributed_ms];
ds_labels = T.dataset;

% Clean up dataset names for display
for i = 1:n
    ds_labels{i} = strrep(ds_labels{i}, '_', '\_');
end

figure('Name', 'Inference Latency Comparison', 'Position', [150, 150, 800, 550]);

b = bar(data, 'grouped');
b(1).FaceColor = [0.2, 0.4, 0.8];   % Blue for centralized
b(2).FaceColor = [0.85, 0.25, 0.1];  % Red for distributed

set(gca, 'XTickLabel', ds_labels, 'FontSize', 12);
xlabel('Dataset', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Inference Latency (ms/sample)', 'FontSize', 13, 'FontWeight', 'bold');
title('Inference Latency: Centralized vs Distributed', ...
      'FontSize', 15, 'FontWeight', 'bold');
legend({'Centralized (Computation only)', ...
        'Distributed (Computation + Communication)'}, ...
       'Location', 'northwest', 'FontSize', 11);
grid on;

% Add value labels on top of bars
for i = 1:length(b)
    xtips = b(i).XEndPoints;
    ytips = b(i).YEndPoints;
    labels = arrayfun(@(v) sprintf('%.2f', v), ytips, 'UniformOutput', false);
    text(xtips, ytips, labels, 'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');
end

% Save
saveas(gcf, fullfile('..', 'results', 'inference_latency.png'));
saveas(gcf, fullfile('..', 'results', 'inference_latency.fig'));
fprintf('Saved: inference_latency.png\n');
