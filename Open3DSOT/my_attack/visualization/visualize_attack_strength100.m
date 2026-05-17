% Visualize clean vs adversarial point cloud for attack strength=100
% File expected:
% /workspace/Open3DSOT/Open3DSOT/my_attack/outputs/cfg_attack_strength100_vis.mat

clc; clear; close all;

mat_path = '/workspace/Open3DSOT/Open3DSOT/my_attack/outputs/cfg_attack_strength100_vis.mat';
D = load(mat_path);

S_clean = squeeze(D.S_clean(1,:,:));  % [N,3]
S_adv   = squeeze(D.S_adv(1,:,:));    % [N,3]
delta   = squeeze(D.delta(1,:,:));    % [N,3]
perturb_norm = sqrt(sum(delta.^2,2));

figure('Color','w','Position',[100,100,1600,500]);

subplot(1,3,1);
scatter3(S_clean(:,1), S_clean(:,2), S_clean(:,3), 8, 'b', 'filled');
axis equal; grid on; xlabel('X'); ylabel('Y'); zlabel('Z');
title('Clean Search Point Cloud');
view(3);

subplot(1,3,2);
scatter3(S_adv(:,1), S_adv(:,2), S_adv(:,3), 8, 'r', 'filled');
axis equal; grid on; xlabel('X'); ylabel('Y'); zlabel('Z');
title('Adversarial Search Point Cloud');
view(3);

subplot(1,3,3);
scatter3(S_adv(:,1), S_adv(:,2), S_adv(:,3), 12, perturb_norm, 'filled');
axis equal; grid on; xlabel('X'); ylabel('Y'); zlabel('Z');
title('Perturbation Magnitude on S_{adv}');
colormap(jet); colorbar;
view(3);

sgtitle('Critical Feature Guided Attack (Strength=100)');

fprintf('L_inf(delta)=%.6f\n', max(abs(delta(:))));
fprintf('Mean L2(delta_i)=%.6f\n', mean(sqrt(sum(delta.^2,2))));
