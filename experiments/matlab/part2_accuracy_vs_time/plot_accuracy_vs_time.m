%% Part 2 - Plot (a): Accuracy vs Time — Distributed PiLot N=7, N=8, N=10
%  One plot per dataset, 3 curves per plot (N=7, N=8, N=10)
%
%  CSV files in: ../csv_results/part2/accuracy_vs_time/

clear; close all; clc;

datasets = {'Coffee', 'Cricket_X', 'ECG5000', 'ElectricDevices', 'FaceAll'};
csv_base = fullfile('..', '..', 'csv_results', 'part2', 'accuracy_vs_time');
fig_dir = fullfile('..', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

N_values = [7, 8, 10];
colors = {[0.0 0.45 0.74], [0.85 0.33 0.10], [0.47 0.67 0.19]};
markers = {'o', 's', '^'};

for d = 1:length(datasets)
    ds = datasets{d};
    
    fig = figure('Position', [100 100 800 500], 'Visible', 'off');
    hold on; grid on;
    
    for n = 1:length(N_values)
        N = N_values(n);
        csv_file = fullfile(csv_base, sprintf('distributed_N%d_%s.csv', N, ds));
        
        if isfile(csv_file)
            T = readtable(csv_file);
            plot(T.elapsed_s, T.test_acc, ...
                 ['-' markers{n}], 'LineWidth', 1.8, ...
                 'MarkerSize', 4, 'Color', colors{n}, ...
                 'DisplayName', sprintf('Distributed PiLot (N=%d)', N));
        else
            fprintf('  WARNING: %s not found\n', csv_file);
        end
    end
    
    xlabel('Time (seconds)', 'FontSize', 12);
    ylabel('Test Accuracy (%)', 'FontSize', 12);
    title(['Accuracy vs Time — ' strrep(ds, '_', '\_') ' (Varying N)'], 'FontSize', 14);
    legend('Location', 'southeast', 'FontSize', 10);
    set(gca, 'FontSize', 11);
    
    saveas(fig, fullfile(fig_dir, ['part2_accuracy_vs_time_' ds '.png']));
    saveas(fig, fullfile(fig_dir, ['part2_accuracy_vs_time_' ds '.fig']));
    fprintf('Saved: part2_accuracy_vs_time_%s.png\n', ds);
    close(fig);
end

fprintf('Done: Part 2 Accuracy vs Time plots.\n');
