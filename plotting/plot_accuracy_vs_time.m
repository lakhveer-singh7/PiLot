%% plot_accuracy_vs_time.m
%  Plot (a): Accuracy vs Time — 1 figure per dataset
%  Compares Centralized vs Distributed test accuracy over training time.
%
%  Reads CSV files from: results/csv/accuracy_vs_time_<dataset>_<model>.csv
%  Format: epoch, time_s, train_acc, test_acc

clear; clc; close all;

datasets = {'Cricket_X', 'ECG5000', 'FaceAll'};
csv_dir = fullfile('..', 'results', 'csv');

for d = 1:length(datasets)
    ds = datasets{d};
    
    % Read centralized data
    cent_file = fullfile(csv_dir, sprintf('accuracy_vs_time_%s_centralized.csv', ds));
    dist_file = fullfile(csv_dir, sprintf('accuracy_vs_time_%s_distributed.csv', ds));
    
    figure('Name', sprintf('Accuracy vs Time - %s', ds), ...
           'Position', [100 + (d-1)*50, 100 + (d-1)*50, 700, 500]);
    
    hold on;
    legends = {};
    
    if isfile(cent_file)
        cent = readtable(cent_file);
        plot(cent.time_s, cent.test_acc, 'b-o', 'LineWidth', 2, ...
             'MarkerSize', 4, 'MarkerFaceColor', 'b');
        legends{end+1} = 'Centralized';
    end
    
    if isfile(dist_file)
        dist = readtable(dist_file);
        plot(dist.time_s, dist.test_acc, 'r-s', 'LineWidth', 2, ...
             'MarkerSize', 4, 'MarkerFaceColor', 'r');
        legends{end+1} = 'Distributed (7 devices, 64 MHz)';
    end
    
    hold off;
    
    xlabel('Training Time (seconds)', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Test Accuracy (%)', 'FontSize', 13, 'FontWeight', 'bold');
    title(sprintf('Accuracy vs Time — %s', strrep(ds, '_', '\_')), ...
          'FontSize', 15, 'FontWeight', 'bold');
    legend(legends, 'Location', 'southeast', 'FontSize', 11);
    grid on;
    set(gca, 'FontSize', 12);
    ylim([0 100]);
    
    % Save figure
    saveas(gcf, fullfile('..', 'results', sprintf('accuracy_vs_time_%s.png', ds)));
    saveas(gcf, fullfile('..', 'results', sprintf('accuracy_vs_time_%s.fig', ds)));
    fprintf('Saved: accuracy_vs_time_%s.png\n', ds);
end

fprintf('\nAll Accuracy vs Time plots generated.\n');
