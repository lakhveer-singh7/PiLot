%% plot_memory_consumption.m
%  Plot (c): Memory Consumption — Grouped bar chart
%  For Centralized: total memory (RSS of single process)
%  For Distributed: average memory per device (across all 7 devices)
%
%  Reads: results/csv/memory_consumption.csv
%  Format: dataset, centralized_kb, distributed_avg_kb, distributed_std_kb

clear; clc; close all;

csv_file = fullfile('..', 'results', 'csv', 'memory_consumption.csv');

if ~isfile(csv_file)
    error('CSV file not found: %s\nRun parse_results.py first.', csv_file);
end

T = readtable(csv_file);
n = height(T);

% Prepare data matrix: rows = datasets, cols = [centralized, distributed_avg]
data = [T.centralized_kb, T.distributed_avg_kb];
err_data = [zeros(n, 1), T.distributed_std_kb];  % Error bars (std) for distributed only

ds_labels = T.dataset;
for i = 1:n
    ds_labels{i} = strrep(ds_labels{i}, '_', '\_');
end

figure('Name', 'Memory Consumption Comparison', 'Position', [200, 200, 800, 550]);

b = bar(data, 'grouped');
b(1).FaceColor = [0.2, 0.4, 0.8];   % Blue for centralized
b(2).FaceColor = [0.85, 0.25, 0.1];  % Red for distributed

hold on;

% Add error bars for distributed (std across devices)
for i = 1:n
    x_dist = b(2).XEndPoints(i);
    y_dist = b(2).YEndPoints(i);
    errorbar(x_dist, y_dist, err_data(i, 2), 'k', 'LineWidth', 1.5, ...
             'CapSize', 10);
end

hold off;

set(gca, 'XTickLabel', ds_labels, 'FontSize', 12);
xlabel('Dataset', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Memory Consumption (KB)', 'FontSize', 13, 'FontWeight', 'bold');
title('Memory Consumption: Centralized vs Distributed', ...
      'FontSize', 15, 'FontWeight', 'bold');
legend({'Centralized (Single Device)', ...
        'Distributed (Avg over 7 Devices ± Std)'}, ...
       'Location', 'northwest', 'FontSize', 11);
grid on;

% Add value labels
for i = 1:length(b)
    xtips = b(i).XEndPoints;
    ytips = b(i).YEndPoints;
    labels = arrayfun(@(v) sprintf('%.1f', v), ytips, 'UniformOutput', false);
    text(xtips, ytips, labels, 'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');
end

% Save
saveas(gcf, fullfile('..', 'results', 'memory_consumption.png'));
saveas(gcf, fullfile('..', 'results', 'memory_consumption.fig'));
fprintf('Saved: memory_consumption.png\n');
