%% Part 2 - Plot (b): Inference Latency Bar Plot
%  One plot per dataset, 6 bars per plot:
%    For each N in {7, 8, 10}: 1 bar computation, 1 bar communication
%
%  CSV files in: ../csv_results/part2/inference_latency_<dataset>.csv

clear; close all; clc;

datasets = {'Coffee', 'Cricket_X', 'ECG5000', 'ElectricDevices', 'FaceAll'};
csv_base = fullfile('..', '..', 'csv_results', 'part2');
fig_dir = fullfile('..', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

comp_color = [0.0 0.45 0.74];
comm_color = [0.85 0.33 0.10];

for d = 1:length(datasets)
    ds = datasets{d};
    csv_file = fullfile(csv_base, ['inference_latency_' ds '.csv']);
    
    if ~isfile(csv_file)
        fprintf('  WARNING: %s not found\n', csv_file);
        continue;
    end
    
    T = readtable(csv_file);
    N_vals = T.N;
    comp_ms = T.computation_ms;
    comm_ms = T.communication_ms;
    
    fig = figure('Position', [100 100 700 500], 'Visible', 'off');
    hold on; grid on;
    
    % Create grouped bar data: 3 groups (N=7, N=8, N=10), 2 bars each
    bar_data = [comp_ms, comm_ms];
    b = bar(bar_data, 'grouped');
    b(1).FaceColor = comp_color;
    b(2).FaceColor = comm_color;
    
    set(gca, 'XTick', 1:length(N_vals));
    x_labels = arrayfun(@(n) sprintf('N=%d', n), N_vals, 'UniformOutput', false);
    set(gca, 'XTickLabel', x_labels);
    
    xlabel('Number of Devices', 'FontSize', 12);
    ylabel('Inference Latency (ms)', 'FontSize', 12);
    title(['Inference Latency — ' strrep(ds, '_', '\_')], 'FontSize', 14);
    legend('Computation', 'Communication', 'Location', 'northwest', 'FontSize', 10);
    set(gca, 'FontSize', 11);
    
    % Add value labels
    for k = 1:2
        xtips = b(k).XEndPoints;
        ytips = b(k).YEndPoints;
        labels = arrayfun(@(v) sprintf('%.3f', v), ytips, 'UniformOutput', false);
        text(xtips, ytips, labels, 'HorizontalAlignment', 'center', ...
             'VerticalAlignment', 'bottom', 'FontSize', 9);
    end
    
    saveas(fig, fullfile(fig_dir, ['part2_inference_latency_' ds '.png']));
    saveas(fig, fullfile(fig_dir, ['part2_inference_latency_' ds '.fig']));
    fprintf('Saved: part2_inference_latency_%s.png\n', ds);
    close(fig);
end

fprintf('Done: Part 2 Inference Latency plots.\n');
