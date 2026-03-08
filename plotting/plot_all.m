%% plot_all.m
%  Master script — generates all 5 plots (compatible with MATLAB & Octave):
%    (a) 3 × Accuracy vs Time (one per dataset)
%    (b) 1 × Inference Latency bar chart
%    (c) 1 × Memory Consumption bar chart
%
%  Prerequisites:
%    1. Run experiments:  bash run_all_local.sh
%    2. Parse results:    python3 parse_results.py
%    3. Then run this:    octave-cli --no-gui plotting/plot_all.m
%
%  Or:  cd PiLot/plotting/ && octave-cli --no-gui plot_all.m

fprintf('=== PiLot Results Plotting ===\n\n');

fprintf('[1/3] Generating Accuracy vs Time plots...\n');
run('plot_accuracy_vs_time.m');

fprintf('\n[2/3] Generating Inference Latency plot...\n');
run('plot_inference_latency.m');

fprintf('\n[3/3] Generating Memory Consumption plot...\n');
run('plot_memory_consumption.m');

fprintf('\n=== All plots generated! ===\n');
fprintf('Output files in: results/\n');
fprintf('  - accuracy_vs_time_Cricket_X.png\n');
fprintf('  - accuracy_vs_time_ECG5000.png\n');
fprintf('  - accuracy_vs_time_FaceAll.png\n');
fprintf('  - inference_latency.png\n');
fprintf('  - memory_consumption.png\n');
