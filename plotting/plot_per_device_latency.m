%% plot_per_device_latency.m
%  Stacked bar chart: per-device inference latency (Computation + Communication)
%  for the Distributed model across all 3 datasets.
%
%  Architecture: 7 devices in a sequential pipeline
%    Head → L0_W0,L0_W1 → L1_W0,L1_W1,L1_W2 → Tail
%
%  Computation delay = FLOPs / PROC_CLOCK_HZ  (64 MHz Cortex-M4F simulation)
%  Communication     = (pipeline_latency - critical_path_computation) / 3 hops
%
%  Requires: results/csv/inference_latency.csv
%  Output:   results/per_device_latency.png

graphics_toolkit('gnuplot');
setenv('GNUTERM', 'dumb');

csv_path = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'csv', 'inference_latency.csv');
out_path = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'per_device_latency.png');

% ---------- Read inference_latency.csv ----------
fid = fopen(csv_path, 'r');
if fid == -1
    error('Cannot open %s', csv_path);
end
hdr = fgetl(fid);   % skip header
datasets = {};
pipeline_cent = [];
pipeline_dist = [];
while ~feof(fid)
    line = strtrim(fgetl(fid));
    if isempty(line); continue; end
    parts = strsplit(line, ',');
    datasets{end+1} = parts{1};
    pipeline_cent(end+1) = str2double(parts{2});
    pipeline_dist(end+1) = str2double(parts{3});
end
fclose(fid);

% ---------- Dataset properties ----------
%  UCR datasets: input_length and num_classes
ds_props = struct();
ds_props.Cricket_X  = struct('input_length', 300, 'num_classes', 12);
ds_props.ECG5000    = struct('input_length', 140, 'num_classes',  5);
ds_props.FaceAll    = struct('input_length', 131, 'num_classes', 14);

% ---------- CNN architecture constants ----------
PROC_CLOCK_HZ = 64e6;           % 64 MHz Cortex-M4F
L0_in_ch  = 1;   L0_out_ch = 16;  L0_k = 5;  L0_s = 1;  L0_p = 2;
L1_in_ch  = 32;  L1_out_ch = 16;  L1_k = 5;  L1_s = 2;  L1_p = 2;
%  Tail: DualPooling → FC(96 → num_classes)
TAIL_IN = 96;    % 48 channels × 2 (avg+max pooling)

device_names = {'Head', 'L0\_W0', 'L0\_W1', 'L1\_W0', 'L1\_W1', 'L1\_W2', 'Tail'};
n_devices = 7;

% Colours
COL_COMP = [1.0 0.6 0.2];       % orange — computation
COL_COMM = [0.75 0.15 0.05];    % dark red — communication

% ---------- Compute per-device latency per dataset ----------
n_ds = numel(datasets);

% comp(ds, dev) and comm(ds, dev)
comp_all = zeros(n_ds, n_devices);
comm_all = zeros(n_ds, n_devices);

for d = 1:n_ds
    ds = datasets{d};
    props = ds_props.(ds);
    il = props.input_length;
    nc = props.num_classes;

    % Output lengths
    L0_out_len = il;                                     % stride=1, same padding
    L1_out_len = floor((il + 2*L1_p - L1_k) / L1_s) + 1;

    % Forward FLOPs per device
    L0_flops = 2 * L0_out_ch * L0_in_ch * L0_k * L0_out_len;
    L1_flops = 2 * L1_out_ch * L1_in_ch * L1_k * L1_out_len;
    tail_flops = 2 * TAIL_IN * nc;

    % Computation delay (ms)
    head_comp = 0;
    L0_comp   = L0_flops / PROC_CLOCK_HZ * 1000;
    L1_comp   = L1_flops / PROC_CLOCK_HZ * 1000;
    tail_comp = tail_flops / PROC_CLOCK_HZ * 1000;

    % Critical-path computation (one worker per layer, since workers run in parallel)
    crit_comp = head_comp + L0_comp + L1_comp + tail_comp;

    % Total communication overhead
    total_comm = pipeline_dist(d) - crit_comp;
    if total_comm < 0; total_comm = 0; end
    comm_per_hop = total_comm / 3;       % 3 IPC boundaries

    % Assign to devices (communication attributed to receiving device)
    %  Head   : no incoming IPC
    %  L0 W0/1: receives from Head  → 1 hop
    %  L1 W0/1/2: receives from L0  → 1 hop
    %  Tail   : receives from L1    → 1 hop
    comp_all(d, :) = [head_comp, L0_comp, L0_comp, L1_comp, L1_comp, L1_comp, tail_comp];
    comm_all(d, :) = [0, comm_per_hop, comm_per_hop, comm_per_hop, comm_per_hop, comm_per_hop, comm_per_hop];
end

% ---------- Plot: 1 row × 3 subplots ----------
fig = figure('Position', [100 100 1500 500]);

for d = 1:n_ds
    subplot(1, 3, d);
    hold on;

    comp_v = comp_all(d, :);
    comm_v = comm_all(d, :);

    bx = 1:n_devices;
    bar_w = 0.6;

    % Draw bars as filled rectangles (stacked)
    for i = 1:n_devices
        % Computation (bottom)
        fill([bx(i)-bar_w/2, bx(i)+bar_w/2, bx(i)+bar_w/2, bx(i)-bar_w/2], ...
             [0, 0, comp_v(i), comp_v(i)], COL_COMP, ...
             'EdgeColor', 'k', 'LineWidth', 0.5);
        % Communication (top)
        fill([bx(i)-bar_w/2, bx(i)+bar_w/2, bx(i)+bar_w/2, bx(i)-bar_w/2], ...
             [comp_v(i), comp_v(i), comp_v(i)+comm_v(i), comp_v(i)+comm_v(i)], ...
             COL_COMM, 'EdgeColor', 'k', 'LineWidth', 0.5);
        % White dashed line at boundary (if both sections > 0)
        if comp_v(i) > 0 && comm_v(i) > 0
            plot([bx(i)-bar_w/2, bx(i)+bar_w/2], [comp_v(i), comp_v(i)], ...
                 'w--', 'LineWidth', 1.5);
        end
    end

    % Value labels
    for i = 1:n_devices
        total_v = comp_v(i) + comm_v(i);
        % Total on top
        text(bx(i), total_v + 0.15, sprintf('%.2f', total_v), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
             'FontSize', 8, 'FontWeight', 'bold');
        % Comp value inside bottom (if visible)
        if comp_v(i) > 0.3
            text(bx(i), comp_v(i)/2, sprintf('%.2f', comp_v(i)), ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                 'FontSize', 7, 'FontWeight', 'bold', 'Color', 'w');
        end
        % Comm value inside top (if visible)
        if comm_v(i) > 0.3
            text(bx(i), comp_v(i) + comm_v(i)/2, sprintf('%.2f', comm_v(i)), ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                 'FontSize', 7, 'FontWeight', 'bold', 'Color', 'w');
        end
    end

    % Legend via dummy plots
    h1 = fill([NaN NaN NaN NaN], [NaN NaN NaN NaN], COL_COMP, 'EdgeColor', 'k');
    h2 = fill([NaN NaN NaN NaN], [NaN NaN NaN NaN], COL_COMM, 'EdgeColor', 'k');
    legend([h1 h2], {'Computation', 'Communication (IPC)'}, ...
           'Location', 'northwest', 'FontSize', 8);

    set(gca, 'XTick', bx, 'XTickLabel', device_names, 'FontSize', 9);
    xlabel('Device', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Inference Latency (ms)', 'FontSize', 11, 'FontWeight', 'bold');
    ds_label = strrep(datasets{d}, '_', ' ');
    title(ds_label, 'FontSize', 13, 'FontWeight', 'bold');
    grid on;
    set(gca, 'GridAlpha', 0.3);
    hold off;
end

% Super-title
ha = axes('Position', [0 0 1 1], 'Visible', 'off');
text(0.5, 0.98, 'Per-Device Inference Latency (Distributed Model)', ...
     'HorizontalAlignment', 'center', 'FontSize', 15, 'FontWeight', 'bold', ...
     'Parent', ha);

print(fig, out_path, '-dpng', '-r200');
close(fig);
fprintf('Saved: %s\n', out_path);
