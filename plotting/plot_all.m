%% plot_all.m
%  Master MATLAB script — generates all 5 plots:
%    (a) 3 × Accuracy vs Time (one per dataset)
%    (b) 1 × Inference Latency bar chart
%    (c) 1 × Memory Consumption bar chart
%
%  Prerequisites:
%    1. Run experiments:  bash run_all.sh
%    2. Parse results:    python3 parse_results.py
%    3. Then run this:    matlab -r "run('plotting/plot_all.m')"
%
%  Or from MATLAB GUI: cd into PiLot/plotting/ and run plot_all

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
