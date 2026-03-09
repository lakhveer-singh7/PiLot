%% Part 1 - Plot (a): Accuracy vs Time
%  One plot per dataset comparing:
%    - Centralized PiLot
%    - Distributed PiLot (N=7)
%    - Distributed RockNet (N=7)
%
%  CSV files expected in: ../csv_results/part1/accuracy_vs_time/
%  Output figures saved to: ./figures/

clear; close all; clc;

datasets = {'Coffee', 'Cricket_X', 'ECG5000', 'ElectricDevices', 'FaceAll'};
csv_base = fullfile('..', '..', 'csv_results', 'part1', 'accuracy_vs_time');
fig_dir = fullfile('..', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

for d = 1:length(datasets)
    ds = datasets{d};
    
    fig = figure('Position', [100 100 800 500], 'Visible', 'off');
    hold on; grid on;
    
    % --- Centralized PiLot ---
    cent_file = fullfile(csv_base, ['centralized_' ds '.csv']);
    if isfile(cent_file)
        T = readtable(cent_file);
        plot(T.elapsed_s, T.test_acc, '-o', 'LineWidth', 1.8, ...
             'MarkerSize', 4, 'Color', [0.0 0.45 0.74], ...
             'DisplayName', 'Centralized PiLot');
    end
    
    % --- Distributed PiLot (N=7) ---
    dist_file = fullfile(csv_base, ['distributed_N7_' ds '.csv']);
    if isfile(dist_file)
        T = readtable(dist_file);
        plot(T.elapsed_s, T.test_acc, '-s', 'LineWidth', 1.8, ...
             'MarkerSize', 4, 'Color', [0.85 0.33 0.10], ...
             'DisplayName', 'Distributed PiLot (N=7)');
    end
    
    % --- Distributed RockNet (N=7) ---
    rock_file = fullfile(csv_base, ['rocknet_N7_' ds '.csv']);
    if isfile(rock_file)
        T = readtable(rock_file);
        plot(T.elapsed_s, T.test_acc, '-^', 'LineWidth', 1.8, ...
             'MarkerSize', 4, 'Color', [0.47 0.67 0.19], ...
             'DisplayName', 'Distributed RockNet (N=7)');
    end
    
    xlabel('Time (seconds)', 'FontSize', 12);
    ylabel('Test Accuracy (%)', 'FontSize', 12);
    title(['Accuracy vs Time — ' strrep(ds, '_', '\_')], 'FontSize', 14);
    legend('Location', 'southeast', 'FontSize', 10);
    set(gca, 'FontSize', 11);
    
    % Save
    saveas(fig, fullfile(fig_dir, ['part1_accuracy_vs_time_' ds '.png']));
    saveas(fig, fullfile(fig_dir, ['part1_accuracy_vs_time_' ds '.fig']));
    fprintf('Saved: part1_accuracy_vs_time_%s.png\n', ds);
    close(fig);
end

fprintf('Done: Part 1 Accuracy vs Time plots.\n');
