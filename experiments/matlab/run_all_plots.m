%% run_all_plots.m — Master MATLAB Script
%  Runs all Part 1 and Part 2 plots.
%
%  Usage (from experiments/matlab/ directory):
%    run_all_plots

clear; close all; clc;
fprintf('============================================\n');
fprintf('  Running ALL MATLAB Plots\n');
fprintf('============================================\n\n');

%% Part 1 Plots
fprintf('--- Part 1: Accuracy vs Time ---\n');
cd('part1_accuracy_vs_time');
plot_accuracy_vs_time;
cd('..');

fprintf('\n--- Part 1: Inference Latency ---\n');
cd('part1_inference_latency');
plot_inference_latency;
cd('..');

fprintf('\n--- Part 1: Memory Consumption ---\n');
cd('part1_memory_consumption');
plot_memory_consumption;
cd('..');

%% Part 2 Plots
fprintf('\n--- Part 2: Accuracy vs Time (N=7,8,10) ---\n');
cd('part2_accuracy_vs_time');
plot_accuracy_vs_time;
cd('..');

fprintf('\n--- Part 2: Inference Latency (N=7,8,10) ---\n');
cd('part2_inference_latency');
plot_inference_latency;
cd('..');

fprintf('\n--- Part 2: Memory Consumption (N=7,8,10) ---\n');
cd('part2_memory_consumption');
plot_memory_consumption;
cd('..');

fprintf('\n============================================\n');
fprintf('  All plots saved to: matlab/figures/\n');
fprintf('============================================\n');
