%% Part 1 - Plot (b): Inference Latency Bar Plot
%  Single bar plot with 10 bars (2 per dataset):
%    - Centralized PiLot: computation latency only
%    - Distributed PiLot: computation + communication latency (stacked)
%  Plus RockNet distributed latency
%
%  Actually 15 bars: 3 per dataset (Centralized, Distributed PiLot, Distributed RockNet)
%
%  CSV: ../csv_results/part1/inference_latency_summary.csv

clear; close all; clc;

csv_file = fullfile('..', '..', 'csv_results', 'part1', 'inference_latency_summary.csv');
fig_dir = fullfile('..', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

T = readtable(csv_file);
n_datasets = height(T);

% Extract data
datasets = T.dataset;
cent_comp = T.centralized_computation_ms;
dist_comp = T.distributed_computation_ms;
dist_comm = T.distributed_communication_ms;
rock_total = T.rocknet_total_ms;

% Prepare grouped bar data: 5 groups, 3 bars each
% Bar 1: Centralized (computation only) — single color
% Bar 2: Distributed PiLot (stacked: computation + communication)
% Bar 3: RockNet Distributed (total)

fig = figure('Position', [100 100 1000 550], 'Visible', 'off');

% Create x positions for grouped bars
x = 1:n_datasets;
bar_width = 0.25;
x1 = x - bar_width;      % Centralized
x2 = x;                   % Distributed PiLot
x3 = x + bar_width;       % RockNet

hold on; grid on;

% Centralized bars (computation only)
b1 = bar(x1, cent_comp, bar_width, 'FaceColor', [0.0 0.45 0.74], ...
         'EdgeColor', 'k', 'LineWidth', 0.8);

% Distributed PiLot: stacked (computation + communication)
b2_comp = bar(x2, dist_comp, bar_width, 'FaceColor', [0.85 0.33 0.10], ...
              'EdgeColor', 'k', 'LineWidth', 0.8);
b2_comm = bar(x2, dist_comm, bar_width, 'FaceColor', [1.0 0.6 0.3], ...
              'EdgeColor', 'k', 'LineWidth', 0.8, 'BaseValue', dist_comp');
% For stacked bar: manually position communication on top
for i = 1:n_datasets
    rectangle('Position', [x2(i)-bar_width/2, dist_comp(i), bar_width, dist_comm(i)], ...
              'FaceColor', [1.0 0.6 0.3], 'EdgeColor', 'k', 'LineWidth', 0.8);
end

% RockNet bars
b3 = bar(x3, rock_total, bar_width, 'FaceColor', [0.47 0.67 0.19], ...
         'EdgeColor', 'k', 'LineWidth', 0.8);

% Fix: Use proper stacked bars approach
% Clear and redo properly
clf;
hold on; grid on;

% Method: grouped bars with stacked for distributed
group_x = (1:n_datasets)';
bar_data = zeros(n_datasets, 4);
bar_data(:,1) = cent_comp;         % Centralized computation
bar_data(:,2) = dist_comp;         % Distributed computation
bar_data(:,3) = dist_comm;         % Distributed communication (stacked on bar 2)
bar_data(:,4) = rock_total;        % RockNet total

% Draw grouped bars manually for proper stacking
bar_w = 0.18;
offsets = [-1.5*bar_w, -0.5*bar_w, -0.5*bar_w, 0.5*bar_w, 1.5*bar_w];

colors = {[0.0 0.45 0.74], [0.85 0.33 0.10], [1.0 0.6 0.3], [0.47 0.67 0.19]};

for i = 1:n_datasets
    gx = i;
    
    % Bar 1: Centralized (single)
    bar(gx - 1.5*bar_w, cent_comp(i), bar_w, 'FaceColor', colors{1}, ...
        'EdgeColor', 'k', 'LineWidth', 0.8);
    
    % Bar 2: Distributed PiLot computation (bottom)
    bar(gx - 0.5*bar_w, dist_comp(i), bar_w, 'FaceColor', colors{2}, ...
        'EdgeColor', 'k', 'LineWidth', 0.8);
    % Bar 2 top: communication
    bar(gx - 0.5*bar_w, dist_comm(i), bar_w, 'FaceColor', colors{3}, ...
        'EdgeColor', 'k', 'LineWidth', 0.8, 'BaseValue', dist_comp(i));
    
    % Bar 3: RockNet (single)
    bar(gx + 0.5*bar_w, rock_total(i), bar_w, 'FaceColor', colors{4}, ...
        'EdgeColor', 'k', 'LineWidth', 0.8);
end

% Labels
set(gca, 'XTick', 1:n_datasets);
ds_labels = cellfun(@(s) strrep(s, '_', '\_'), datasets, 'UniformOutput', false);
set(gca, 'XTickLabel', ds_labels);
xlabel('Dataset', 'FontSize', 12);
ylabel('Inference Latency (ms)', 'FontSize', 12);
title('Inference Latency Comparison — All Datasets', 'FontSize', 14);

% Legend (invisible bars for legend entries)
h1 = bar(nan, nan, 'FaceColor', colors{1}, 'EdgeColor', 'k');
h2 = bar(nan, nan, 'FaceColor', colors{2}, 'EdgeColor', 'k');
h3 = bar(nan, nan, 'FaceColor', colors{3}, 'EdgeColor', 'k');
h4 = bar(nan, nan, 'FaceColor', colors{4}, 'EdgeColor', 'k');
legend([h1 h2 h3 h4], {'Centralized PiLot (Computation)', ...
        'Distributed PiLot (Computation)', ...
        'Distributed PiLot (Communication)', ...
        'Distributed RockNet (Total)'}, ...
        'Location', 'northwest', 'FontSize', 9);

set(gca, 'FontSize', 11);

saveas(fig, fullfile(fig_dir, 'part1_inference_latency.png'));
saveas(fig, fullfile(fig_dir, 'part1_inference_latency.fig'));
fprintf('Saved: part1_inference_latency.png\n');
close(fig);

fprintf('Done: Part 1 Inference Latency plot.\n');
