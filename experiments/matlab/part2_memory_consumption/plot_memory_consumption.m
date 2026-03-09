%% Part 2 - Plot (c): Memory Consumption Bar Plot with Std Deviation
%  One plot per dataset, 9 bars total:
%    For each N in {7, 8, 10}: 3 bars (weight avg, optimizer avg, buffer avg)
%    With error bars for standard deviation
%
%  CSV files in: ../csv_results/part2/memory_consumption_<dataset>.csv

clear; close all; clc;

datasets = {'Coffee', 'Cricket_X', 'ECG5000', 'ElectricDevices', 'FaceAll'};
csv_base = fullfile('..', '..', 'csv_results', 'part2');
fig_dir = fullfile('..', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

weight_color = [0.0 0.45 0.74];   % Blue
optim_color  = [0.85 0.33 0.10];  % Orange
buffer_color = [0.47 0.67 0.19];  % Green

for d = 1:length(datasets)
    ds = datasets{d};
    csv_file = fullfile(csv_base, ['memory_consumption_' ds '.csv']);
    
    if ~isfile(csv_file)
        fprintf('  WARNING: %s not found\n', csv_file);
        continue;
    end
    
    T = readtable(csv_file);
    N_vals = T.N;
    n_configs = length(N_vals);
    
    % Convert bytes to KB
    avg_w = T.avg_weight_bytes / 1024;
    std_w = T.std_weight_bytes / 1024;
    avg_o = T.avg_optimizer_bytes / 1024;
    std_o = T.std_optimizer_bytes / 1024;
    avg_b = T.avg_buffer_bytes / 1024;
    std_b = T.std_buffer_bytes / 1024;
    
    fig = figure('Position', [100 100 800 550], 'Visible', 'off');
    hold on; grid on;
    
    % Create grouped bar data: 3 groups (N=7, N=8, N=10), 3 bars each
    bar_data = [avg_w, avg_o, avg_b];
    b = bar(bar_data, 'grouped');
    b(1).FaceColor = weight_color;
    b(2).FaceColor = optim_color;
    b(3).FaceColor = buffer_color;
    
    % Add error bars
    for k = 1:3
        if k == 1
            err_vals = std_w;
        elseif k == 2
            err_vals = std_o;
        else
            err_vals = std_b;
        end
        
        xtips = b(k).XEndPoints;
        ytips = b(k).YEndPoints;
        errorbar(xtips, ytips, err_vals, 'k.', 'LineWidth', 1.2, ...
                 'HandleVisibility', 'off');
    end
    
    set(gca, 'XTick', 1:n_configs);
    x_labels = arrayfun(@(n) sprintf('N=%d', n), N_vals, 'UniformOutput', false);
    set(gca, 'XTickLabel', x_labels);
    
    xlabel('Number of Devices', 'FontSize', 12);
    ylabel('Memory per Device (KB)', 'FontSize', 12);
    title(['Memory Consumption — ' strrep(ds, '_', '\_') ' (Avg per Device \pm Std)'], 'FontSize', 14);
    legend('Weights (Avg)', 'Optimizer (Avg)', 'Buffers (Avg)', ...
           'Location', 'northwest', 'FontSize', 10);
    set(gca, 'FontSize', 11);
    
    % Add value labels
    for k = 1:3
        xtips = b(k).XEndPoints;
        ytips = b(k).YEndPoints;
        labels = arrayfun(@(v) sprintf('%.1f', v), ytips, 'UniformOutput', false);
        text(xtips, ytips + max(ytips)*0.02, labels, ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
             'FontSize', 8);
    end
    
    saveas(fig, fullfile(fig_dir, ['part2_memory_consumption_' ds '.png']));
    saveas(fig, fullfile(fig_dir, ['part2_memory_consumption_' ds '.fig']));
    fprintf('Saved: part2_memory_consumption_%s.png\n', ds);
    close(fig);
end

fprintf('Done: Part 2 Memory Consumption plots.\n');
