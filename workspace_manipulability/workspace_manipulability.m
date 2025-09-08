clc; clear; close all;

% --- Define a single finger link ---
% Link1 is given
L1 = Link('d', 16.5, 'a', 51, 'alpha', pi/2);

% Target: displacement (dx, dy) = (-20, +125) in the end frame of Link1
dx = -20;
dy = 125;

% Compute translation length a2 and offset angle theta0
a2 = hypot(dx, dy);          % = sqrt(dx^2 + dy^2)
theta0 = atan2(dy, dx);      % ensures a2*cosθ=dx, a2*sinθ=dy

% Link2 definition (standard DH)
L2 = Link('d', 0, 'a', a2, 'alpha', 0, 'offset', theta0);

% --- Create three finger robots ---
% Finger 1 - base at original position
finger1 = SerialLink([L1 L2], 'name', 'Finger 1');
finger1.base = transl(28, 0, 0);

% Finger 2 - rotated 120° about Z axis
finger2 = SerialLink([L1 L2], 'name', 'Finger 2');
rotation_120 = trotz(2*pi/3); % 120° = 2π/3 rad
finger2.base = rotation_120 * transl(28, 0, 0);

% Finger 3 - rotated 240° about Z axis
finger3 = SerialLink([L1 L2], 'name', 'Finger 3');
rotation_240 = trotz(4*pi/3); % 240° = 4π/3 rad
finger3.base = rotation_240 * transl(28, 0, 0);

% --- Define joint limits ---
% Joint 1: -60° to +60°
q1_min = -60 * pi/180;
q1_max =  60 * pi/180;

% Joint 2: -60° to +30°
q2_min = -60 * pi/180;
q2_max =  30 * pi/180;

fprintf('Joint limits:\n');
fprintf('Joint 1: %.1f° to %.1f°\n', q1_min*180/pi, q1_max*180/pi);
fprintf('Joint 2: %.1f° to %.1f°\n', q2_min*180/pi, q2_max*180/pi);

% --- Monte Carlo sampling parameters ---
num_samples = 5000; % number of sampled points per finger
fprintf('Number of Monte Carlo samples per finger: %d\n', num_samples);

% --- Perform Monte Carlo sampling for each finger ---
fprintf('Performing Monte Carlo sampling for each finger...\n');

% Storage for fingertip positions
workspace_finger1 = [];
workspace_finger2 = [];
workspace_finger3 = [];

% Finger 1 sampling
fprintf('Sampling Finger 1...\n');
for i = 1:num_samples
    % Random joint angles
    q1 = q1_min + (q1_max - q1_min) * rand();
    q2 = q2_min + (q2_max - q2_min) * rand();
    q  = [q1, q2];

    % Forward kinematics
    T1 = finger1.fkine(q);

    % Extract position vector
    if isa(T1, 'SE3')
        pos1 = T1.t';
    elseif size(T1,1) == 4 && size(T1,2) == 4
        pos1 = T1(1:3,4)';
    else
        if isnumeric(T1) && length(T1) >= 3
            pos1 = T1(1:3)';
        else
            error('Unrecognized transform matrix format');
        end
    end

    workspace_finger1 = [workspace_finger1; pos1];

    % Progress
    if mod(i, 1000) == 0
        fprintf('  Finger 1 progress: %d/%d (%.1f%%)\n', i, num_samples, i/num_samples*100);
    end
end

% Finger 2 sampling
fprintf('Sampling Finger 2...\n');
for i = 1:num_samples
    q1 = q1_min + (q1_max - q1_min) * rand();
    q2 = q2_min + (q2_max - q2_min) * rand();
    q  = [q1, q2];

    T2 = finger2.fkine(q);

    if isa(T2, 'SE3')
        pos2 = T2.t';
    elseif size(T2,1) == 4 && size(T2,2) == 4
        pos2 = T2(1:3,4)';
    else
        if isnumeric(T2) && length(T2) >= 3
            pos2 = T2(1:3)';
        else
            error('Unrecognized transform matrix format');
        end
    end

    workspace_finger2 = [workspace_finger2; pos2];

    if mod(i, 1000) == 0
        fprintf('  Finger 2 progress: %d/%d (%.1f%%)\n', i, num_samples, i/num_samples*100);
    end
end

% Finger 3 sampling
fprintf('Sampling Finger 3...\n');
for i = 1:num_samples
    q1 = q1_min + (q1_max - q1_min) * rand();
    q2 = q2_min + (q2_max - q2_min) * rand();
    q  = [q1, q2];

    T3 = finger3.fkine(q);

    if isa(T3, 'SE3')
        pos3 = T3.t';
    elseif size(T3,1) == 4 && size(T3,2) == 4
        pos3 = T3(1:3,4)';
    else
        if isnumeric(T3) && length(T3) >= 3
            pos3 = T3(1:3)';
        else
            error('Unrecognized transform matrix format');
        end
    end

    workspace_finger3 = [workspace_finger3; pos3];

    if mod(i, 1000) == 0
        fprintf('  Finger 3 progress: %d/%d (%.1f%%)\n', i, num_samples, i/num_samples*100);
    end
end

fprintf('Sampling complete!\n');

% --- Plot workspaces and initial finger configurations ---
figure('Color', 'w', 'Position', [100, 100, 1200, 900]);
hold on; grid on;

% Initial configuration for drawing (q = [0 0])
q_init = [0 0];

% Set default plot options via plotopt
finger1.plotopt = {'linkcolor', [0.5 0.5 0.5], 'jointcolor', [0.5 0.5 0.5]};
finger2.plotopt = {'linkcolor', [0.5 0.5 0.5], 'jointcolor', [0.5 0.5 0.5]};
finger3.plotopt = {'linkcolor', [0.5 0.5 0.5], 'jointcolor', [0.5 0.5 0.5]};

% Draw three fingers at initial pose
finger1.plot(q_init, 'workspace', [-200 200 -200 200 -50 300], ...
             'notiles', 'nobase', 'nowrist', 'noshadow', 'noname');
finger2.plot(q_init, 'workspace', [-200 200 -200 200 -50 300], ...
             'notiles', 'nobase', 'nowrist', 'noshadow', 'noname');
finger3.plot(q_init, 'workspace', [-200 200 -200 200 -50 300], ...
             'notiles', 'nobase', 'nowrist', 'noshadow', 'noname');

% Merge all workspace points
all_workspace = [workspace_finger1; workspace_finger2; workspace_finger3];

% Use Z coordinate as color (normalized to [0,1])
z_values = all_workspace(:,3);
z_min = min(z_values);
z_max = max(z_values);
z_normalized = (z_values - z_min) / (z_max - z_min + eps);

% Scatter plot colored by normalized Z
scatter3(all_workspace(:,1), all_workspace(:,2), all_workspace(:,3), ...
         1, z_normalized, 'filled');
colormap winter;
caxis([0 1]);

% Mark the origin and three bases (gray)
plot3(0, 0, 0, 'ko', 'MarkerSize', 10, 'MarkerFaceColor', [0.5 0.5 0.5], 'DisplayName', 'Origin');
plot3(28, 0, 0, 'ks', 'MarkerSize', 8, 'MarkerFaceColor', [0.5 0.5 0.5]);
plot3(28*cos(2*pi/3), 28*sin(2*pi/3), 0, 'ks', 'MarkerSize', 8, 'MarkerFaceColor', [0.5 0.5 0.5]);
plot3(28*cos(4*pi/3), 28*sin(4*pi/3), 0, 'ks', 'MarkerSize', 8, 'MarkerFaceColor', [0.5 0.5 0.5]);

% Add a hollow cylinder (diameter 100 mm, extend −Z by 100 mm)
cylinder_radius = 50;   % radius 50 mm (diameter 100 mm)
cylinder_height = 100;  % extend downwards by 100 mm
n_points = 50;          % points around the circle

theta = linspace(0, 2*pi, n_points);
x_circle = cylinder_radius * cos(theta);
y_circle = cylinder_radius * sin(theta);

[X_cyl, Y_cyl, Z_cyl] = cylinder(cylinder_radius, n_points-1);
Z_cyl = Z_cyl * cylinder_height;
Z_cyl = -Z_cyl; % extend along negative Z

% Cylinder wall (semi-transparent)
surf(X_cyl, Y_cyl, Z_cyl, 'FaceColor', [0.7, 0.3, 0.7], 'FaceAlpha', 0.3, ...
     'EdgeColor', [0.5, 0.2, 0.5], 'EdgeAlpha', 0.6, 'LineWidth', 1);

% Top ring at z=0
plot3(x_circle, y_circle, zeros(size(x_circle)), ...
      'Color', [0.5, 0.2, 0.5], 'LineWidth', 3);

% Bottom ring at z = -height
plot3(x_circle, y_circle, -cylinder_height * ones(size(x_circle)), ...
      'Color', [0.5, 0.2, 0.5], 'LineWidth', 3);

% Optional vertical guide lines
n_vertical_lines = 8;
vertical_indices = round(linspace(1, length(x_circle), n_vertical_lines));
for i = vertical_indices
    plot3([x_circle(i), x_circle(i)], [y_circle(i), y_circle(i)], ...
          [0, -cylinder_height], 'Color', [0.5, 0.2, 0.5], 'LineWidth', 1);
end

% xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)'); % optional labels
axis equal; view(3);
zlim([-100 200]);

% Adjust axes placement in the figure window
set(gca, 'Position', [0.05, 0.05, 0.9, 0.9]);
hold off;

% --- Workspace statistics ---
fprintf('\n=== Workspace statistics ===\n');
x_range = [min(all_workspace(:,1)), max(all_workspace(:,1))];
y_range = [min(all_workspace(:,2)), max(all_workspace(:,2))];
z_range = [min(all_workspace(:,3)), max(all_workspace(:,3))];

fprintf('X range: [%.2f, %.2f] mm, span: %.2f mm\n', x_range(1), x_range(2), diff(x_range));
fprintf('Y range: [%.2f, %.2f] mm, span: %.2f mm\n', y_range(1), y_range(2), diff(y_range));
fprintf('Z range: [%.2f, %.2f] mm, span: %.2f mm\n', z_range(1), z_range(2), diff(z_range));

distances = sqrt(sum(all_workspace.^2, 2));
fprintf('Distance to origin: [%.2f, %.2f] mm\n', min(distances), max(distances));
fprintf('Mean distance: %.2f mm\n', mean(distances));
fprintf('Total sampled points: %d (each finger %d)\n', size(all_workspace,1), num_samples);

% --- Reachability analysis (distance to nearest sampled point) ---
fprintf('\n=== Reachability analysis (nearest distance) ===\n');
test_points = [
    0,   0, 100;   % above center
    0,   0,  50;   % mid height
    50,  0,  75;   % +X direction
   -25, 43.3, 75;  % +120° direction
   -25,-43.3, 75;  % +240° direction
];

fprintf('Test point reachability:\n');
for i = 1:size(test_points, 1)
    target = test_points(i, :);
    dist1 = min(sqrt(sum((workspace_finger1 - target).^2, 2)));
    dist2 = min(sqrt(sum((workspace_finger2 - target).^2, 2)));
    dist3 = min(sqrt(sum((workspace_finger3 - target).^2, 2)));
    fprintf('Point [%.1f, %.1f, %.1f]: F1 %.2f mm, F2 %.2f mm, F3 %.2f mm\n', ...
            target(1), target(2), target(3), dist1, dist2, dist3);
end

% --- Convex-hull based workspace size estimates ---
fprintf('\n=== Workspace size estimates (convex hull) ===\n');

% Surface area helper (sum of triangle areas from hull facets)
hull_area = @(P,K) sum( 0.5 * vecnorm(cross(P(K(:,2),:) - P(K(:,1),:), ...
                                             P(K(:,3),:) - P(K(:,1),:), 2), 2) );

% Individual fingers (3D hull surface area & volume)
[K1, vol1] = convhulln(workspace_finger1);
area1 = hull_area(workspace_finger1, K1);
fprintf('Finger 1: surface area ≈ %.2f mm^2, volume ≈ %.2f mm^3\n', area1, vol1);

[K2, vol2] = convhulln(workspace_finger2);
area2 = hull_area(workspace_finger2, K2);
fprintf('Finger 2: surface area ≈ %.2f mm^2, volume ≈ %.2f mm^3\n', area2, vol2);

[K3, vol3] = convhulln(workspace_finger3);
area3 = hull_area(workspace_finger3, K3);
fprintf('Finger 3: surface area ≈ %.2f mm^2, volume ≈ %.2f mm^3\n', area3, vol3);

% Combined (union by convex hull of all points)
[Kall, vol_all] = convhulln(all_workspace);
area_all = hull_area(all_workspace, Kall);
fprintf('All fingers (combined points): surface area ≈ %.2f mm^2, volume ≈ %.2f mm^3\n', area_all, vol_all);

% XY-projection area of the combined workspace (2D convex hull)
[~, area_xy] = convhull(all_workspace(:,1), all_workspace(:,2));
fprintf('XY-projection convex-hull area (combined): ≈ %.2f mm^2\n', area_xy);

% --- Save sampled workspace data ---
fprintf('\nSaving workspace data to file...\n');
save('workspace_data.mat', 'workspace_finger1', 'workspace_finger2', 'workspace_finger3', ...
     'num_samples', 'q1_min', 'q1_max', 'q2_min', 'q2_max', 'all_workspace');
fprintf('Data saved to workspace_data.mat\n');

%% --- Yoshikawa manipulability analysis (single finger: finger1) ---
fprintf('\n=== Yoshikawa manipulability analysis ===\n');

finger = finger1;

% Joint grids (degrees for axes; radians for computation)
th1_deg = linspace(-60, 60, 121);   % θ1: -60° ~ 60°
th2_deg = linspace(-60, 30,  91);   % θ2: -60° ~ 30°
[TH1d, TH2d] = meshgrid(th1_deg, th2_deg);

TH1 = deg2rad(TH1d);
TH2 = deg2rad(TH2d);

% Yoshikawa manipulability: w(theta) = sqrt(det(J^T J)), translational part
W = zeros(size(TH1));
for i = 1:numel(TH1)
    q  = [TH1(i), TH2(i)];
    J6 = finger.jacob0(q);     % 6x2 Jacobian in base frame
    J  = J6(1:3,:);            % take translational part (3x2)
    G  = J.' * J;              % 2x2
    detG = det(G);
    if detG < 0 && detG > -1e-12, detG = 0; end % numeric safety
    W(i) = sqrt(detG);
end

[minW, kmin] = min(W(:)); [imin, jmin] = ind2sub(size(W), kmin);
[maxW, kmax] = max(W(:)); [imax, jmax] = ind2sub(size(W), kmax);

% Plot manipulability surface
figure('Color','w', 'Position', [200, 100, 1000, 750]); 
hold on; box on; grid on;
surf(TH1d, TH2d, W, 'EdgeColor','none');
view(45,25); 
colormap winter; 
cb = colorbar; ylabel(cb, 'w', 'FontSize', 12);

% If your W is small, you can relax z-limits; here we keep the original idea commented:
% zlabel('w = sqrt(det(J^T J))', 'FontSize', 12);
% xlabel('\theta_1 (deg)', 'FontSize', 12);
% ylabel('\theta_2 (deg)', 'FontSize', 12);
% title('Yoshikawa Manipulability Surface for Single Finger', 'FontSize', 14);

% Original script used a fixed lower z bound (80); switch to a safe bound:
zlim([max(0, minW), maxW]);

fprintf('w min = %.4f at (θ1, θ2) = (%.1f°, %.1f°)\n', minW, TH1d(imin,1), TH2d(1,jmin));
fprintf('w max = %.4f at (θ1, θ2) = (%.1f°, %.1f°)\n', maxW, TH1d(imax,1), TH2d(1,jmax));

%% ---- Finger model (same as above) for sigma_1 coloring ----
L1 = Link('d', 16.5, 'a', 51, 'alpha', pi/2);

dx = -20; dy = 125;
a2 = hypot(dx, dy);
theta0 = atan2(dy, dx);
L2 = Link('d', 0, 'a', a2, 'alpha', 0, 'offset', theta0);

finger = SerialLink([L1 L2], 'name', 'Finger');
finger.base = transl(28, 0, 0);

%% ---- Joint limits (radians) ----
q1_lim = deg2rad([-60 60]);   % q1: -60° ~ 60°
q2_lim = deg2rad([-60 30]);   % q2: -60° ~ 30°

% Sampling resolution (tunable)
n1 = 101;   % samples on q1
n2 =  91;   % samples on q2
[q1g, q2g] = meshgrid(linspace(q1_lim(1), q1_lim(2), n1), ...
                      linspace(q2_lim(1), q2_lim(2), n2));

%% ---- Compute end positions and σ1 ----
P  = zeros(numel(q1g), 3);   % end-effector positions
S1 = zeros(numel(q1g), 1);   % σ1 = s_min / s_max (isotropy index)

for k = 1:numel(q1g)
    q = [q1g(k), q2g(k)];

    % End position
    T = finger.fkine(q);
    if isa(T,'SE3'), P(k,:) = T.t.'; else, P(k,:) = T(1:3,4).'; end

    % Translational Jacobian J ∈ R^{3x2}
    J6 = finger.jacob0(q);     % 6x2
    J  = J6(1:3,:);            % 3x2

    % Singular values s(1) >= s(2) >= 0
    s = svd(J);
    if s(1) > 0
        S1(k) = s(end) / s(1); % σ1 = s_min / s_max
    else
        S1(k) = 0;
    end
end

%% ---- Fig: workspace colored by σ1 ----
figure('Name','Workspace colored by sigma_1','Color','w');
scatter3(P(:,1), P(:,2), P(:,3), 10, S1, 'filled'); hold on; grid on; axis equal;
colormap winter; colorbar; caxis([0 1]);      % σ1 ∈ [0,1]
finger.plot([0 0], 'nobase','noname','nowrist','noshadow');
% xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
zlim([-100 200]);
xlim([-50 200]);
ylim([-200 200]);
