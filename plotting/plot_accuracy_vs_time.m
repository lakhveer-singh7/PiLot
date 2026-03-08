%% plot_accuracy_vs_time.m
%  Plot (a): Accuracy vs Time — 1 figure per dataset
%  Compares Centralized vs Distributed test accuracy over training time.
%
%  Reads CSV files from: results/csv/accuracy_vs_time_<dataset>_<model>.csv
%  Format: epoch, time_s, train_acc, test_acc
%
%  Compatible with both MATLAB and GNU Octave.

clear; clc; close all;
graphics_toolkit('gnuplot');
setenv('GNUTERM', 'dumb');
warning('off', 'all');

datasets = {'Cricket_X', 'ECG5000', 'FaceAll'};
csv_dir = fullfile('..', 'results', 'csv');

for d = 1:length(datasets)
    ds = datasets{d};

    cent_file = fullfile(csv_dir, sprintf('accuracy_vs_time_%s_centralized.csv', ds));
    dist_file = fullfile(csv_dir, sprintf('accuracy_vs_time_%s_distributed.csv', ds));

    hf = figure();
    set(hf, 'PaperPositionMode', 'auto', 'Position', [100, 100, 700, 500]);

    hold on;
    legends = {};

    if exist(cent_file, 'file')
        cent = csvread(cent_file, 1, 0);  % skip header row
        % columns: epoch(1) time_s(2) train_acc(3) test_acc(4)
        plot(cent(:,2), cent(:,4), 'b-o', 'LineWidth', 2, ...
             'MarkerSize', 4, 'MarkerFaceColor', 'b');
        legends{end+1} = 'Centralized';
    end

    if exist(dist_file, 'file')
        dist = csvread(dist_file, 1, 0);
        plot(dist(:,2), dist(:,4), 'r-s', 'LineWidth', 2, ...
             'MarkerSize', 4, 'MarkerFaceColor', 'r');
        legends{end+1} = 'Distributed (7 devices, 64 MHz)';
    end

    hold off;

    xlabel('Training Time (seconds)', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Test Accuracy (%)', 'FontSize', 13, 'FontWeight', 'bold');
    title(sprintf('Accuracy vs Time - %s', strrep(ds, '_', '\_')), ...
          'FontSize', 15, 'FontWeight', 'bold');
    legend(legends, 'Location', 'southeast', 'FontSize', 11);
    grid on;
    set(gca, 'FontSize', 12);
    ylim([0 100]);

    out_png = fullfile('..', 'results', sprintf('accuracy_vs_time_%s.png', ds));
    print(hf, out_png, '-dpng', '-r200');
    close(hf);
    fprintf('Saved: %s\n', out_png);
end

fprintf('\nAll Accuracy vs Time plots generated.\n');
