%% plot_all.m
%  Master script — generates all 6 plots (compatible with MATLAB & Octave):
%    (a)  3 × Accuracy vs Time (one per dataset)
%    (b1) 1 × Inference Latency bar chart
%    (b2) 1 × Per-Device Inference Latency (stacked computation + communication)
%    (c)  1 × Memory Consumption bar chart
%
%  Prerequisites:
%    1. Run experiments:  bash run_all_local.sh
%    2. Parse results:    python3 parse_results.py
%    3. Then run this:    octave-cli --no-gui plotting/plot_all.m
%
%  Or:  cd PiLot/plotting/ && octave-cli --no-gui plot_all.m

fprintf('=== PiLot Results Plotting ===\n\n');

fprintf('[1/4] Generating Accuracy vs Time plots...\n');
run('plot_accuracy_vs_time.m');

fprintf('\n[2/4] Generating Inference Latency plot...\n');
run('plot_inference_latency.m');

fprintf('\n[3/4] Generating Per-Device Latency plot...\n');
run('plot_per_device_latency.m');

fprintf('\n[4/4] Generating Memory Consumption plot...\n');
run('plot_memory_consumption.m');

fprintf('\n=== All plots generated! ===\n');
fprintf('Output files in: results/\n');
fprintf('  - accuracy_vs_time_Cricket_X.png\n');
fprintf('  - accuracy_vs_time_ECG5000.png\n');
fprintf('  - accuracy_vs_time_FaceAll.png\n');
fprintf('  - inference_latency.png\n');
fprintf('  - per_device_latency.png\n');
fprintf('  - memory_consumption.png\n');
