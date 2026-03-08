%% plot_multi_all.m
%  Master script: generates all 9 multi-device comparison plots.
%  Run from the plotting/ directory.
%
%  Usage (headless):
%    xvfb-run --auto-servernum octave-cli --no-gui --no-init-file plot_multi_all.m

fprintf('=== PiLot Multi-Device Comparison Plots (Octave) ===\n\n');

fprintf('[1/3] Accuracy vs Time …\n');
run('plot_multi_accuracy_vs_time.m');
fprintf('\n');

fprintf('[2/3] Inference Latency …\n');
run('plot_multi_inference_latency.m');
fprintf('\n');

fprintf('[3/3] Memory Consumption …\n');
run('plot_multi_memory_consumption.m');
fprintf('\n');

fprintf('=== All 9 multi-device plots generated! ===\n');
fprintf('Output: results/multi_device/\n');
