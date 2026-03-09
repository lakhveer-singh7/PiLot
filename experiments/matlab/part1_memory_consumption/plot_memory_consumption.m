%% Part 1 - Plot (c): Memory Consumption Bar Plot
%  Bar plot with 10 bars (2 per dataset):
%    - Centralized PiLot: total memory
%    - Distributed PiLot (N=7): average memory per device
%  Plus RockNet distributed memory
%
%  CSV: ../csv_results/part1/memory_consumption_summary.csv

clear; close all; clc;

csv_file = fullfile('..', '..', 'csv_results', 'part1', 'memory_consumption_summary.csv');
fig_dir = fullfile('..', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

T = readtable(csv_file);
n_datasets = height(T);

datasets = T.dataset;

% Convert bytes to KB for readability
cent_avg_kb = T.centralized_avg_bytes / 1024;
dist_avg_kb = T.distributed_avg_bytes / 1024;
rock_runtime_kb = T.rocknet_runtime_kb;

% For Part 1 bar plot: 3 bars per dataset (centralized, distributed PiLot, RockNet)
fig = figure('Position', [100 100 1000 550], 'Visible', 'off');

bar_data = [cent_avg_kb, dist_avg_kb, rock_runtime_kb];
b = bar(bar_data, 'grouped');
b(1).FaceColor = [0.0 0.45 0.74];
b(2).FaceColor = [0.85 0.33 0.10];
b(3).FaceColor = [0.47 0.67 0.19];

grid on;
set(gca, 'XTick', 1:n_datasets);
ds_labels = cellfun(@(s) strrep(s, '_', '\_'), datasets, 'UniformOutput', false);
set(gca, 'XTickLabel', ds_labels);
xlabel('Dataset', 'FontSize', 12);
ylabel('Memory Consumption (KB)', 'FontSize', 12);
title('Memory Consumption — Centralized vs Distributed (N=7)', 'FontSize', 14);
legend('Centralized PiLot (Total)', ...
       'Distributed PiLot (Avg/Device)', ...
       'Distributed RockNet (Runtime)', ...
       'Location', 'northwest', 'FontSize', 10);
set(gca, 'FontSize', 11);

% Add value labels on bars
for k = 1:3
    xtips = b(k).XEndPoints;
    ytips = b(k).YEndPoints;
    labels = arrayfun(@(v) sprintf('%.1f', v), ytips, 'UniformOutput', false);
    text(xtips, ytips, labels, 'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'bottom', 'FontSize', 8);
end

saveas(fig, fullfile(fig_dir, 'part1_memory_consumption.png'));
saveas(fig, fullfile(fig_dir, 'part1_memory_consumption.fig'));
fprintf('Saved: part1_memory_consumption.png\n');
close(fig);

fprintf('Done: Part 1 Memory Consumption plot.\n');
