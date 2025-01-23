% .mat dosyasını yükle
load('data_v3.mat');

% 'Data' yapısının alanlarını görüntüle
disp("Data yapısındaki alanlar:");
disp(fieldnames(Data));

% 'task' yapısını al
tasks = Data.task;

% Görev sayısını kontrol et
disp("Görev sayısı:");
disp(length(tasks));

% İlk görev verisini incele
participant_task_1 = tasks(1); % İlk struct elemanını al
disp("İlk görevdeki mevcut alanlar:");
disp(fieldnames(participant_task_1));

% Örneğin, 'gaze' verisini kontrol et
if isfield(participant_task_1, 'gaze')
    gaze_data = participant_task_1.gaze;
    disp("Gaze verisi sütun adları:");
    disp(fieldnames(gaze_data));

else
    disp("'Gaze' verisi bulunamadı.");

end

% Örneğin, 'gaze' verisini kontrol et
if isfield(participant_task_1, 'pupil')
    pupil_data = participant_task_1.pupil;
    disp("Pupil verisi sütun adları:");
    disp(fieldnames(pupil_data));

else
    disp("'Pupil' verisi bulunamadı.");

end
    % Örneğin, 'gaze' verisini kontrol et
if isfield(participant_task_1, 'blinks')
    blinks_data = participant_task_1.blinks;
    disp("Blinks verisi sütun adları:");
    disp(fieldnames(blinks_data));

else
    disp("'Blinks' verisi bulunamadı.");
end
% Örneğin, 'gaze' verisini kontrol et
if isfield(participant_task_1, 'annotation')
    annotation_data = participant_task_1.annotation;
    disp("Annotation verisi sütun adları:");
    disp(fieldnames(annotation_data));

else
    disp("'Annotation' verisi bulunamadı.");

end

%% 
% .mat dosyasını yükle
load('data_v3.mat');

% Özellik ve etiketlerin saklanacağı hücre dizisi
features_labels = {};

% Katılımcılar için döngü
for i = 1:numel(Data)
    participant_tasks = Data(i).task;
    
    % Görevler için döngü
    for j = 1:numel(participant_tasks)
        task = participant_tasks(j);

        % Özellik çıkarımı (gaze, pupil, blink, annotation)
        % Varsayılan olarak NaN atanır ve kontrol edilir
        if isempty(task.gaze)
            gaze_features = nan(1, 5);
            low_freq_energy = NaN;
            high_freq_energy = NaN;
        else
            gaze_features = mean(task.gaze{:, {'norm_pos_x', 'norm_pos_y', ...
                'gaze_point_3d_x', 'gaze_point_3d_y', 'gaze_point_3d_z'}}, 'omitnan');
            
            % Perform FFT on one of the gaze signals (e.g., norm_pos_x)
            gaze_signal = task.gaze.norm_pos_x; % Replace with appropriate column
            fft_result = fft(gaze_signal);
            power_spectrum = abs(fft_result).^2; % Power spectrum
            
            % Define frequency ranges for low and high frequencies
            low_freq_range = 1:10; % Example: indices corresponding to low frequencies
            high_freq_range = 11:20; % Example: indices corresponding to high frequencies
            
            % Compute energy in each range
            low_freq_energy = sum(power_spectrum(low_freq_range));
            high_freq_energy = sum(power_spectrum(high_freq_range));
        end

        if isempty(task.pupil)
            pupil_features = nan(1, 5);
        else
            pupil_features = mean(task.pupil{:, {'diameter', 'model_confidence', ...
                'sphere_center_x', 'sphere_center_y', 'sphere_center_z'}}, 'omitnan');
        end

        if isempty(task.blinks)
            blink_count = 0;
            blink_duration = NaN;
        else
            blink_count = size(task.blinks, 1);
            blink_duration = mean(task.blinks.duration, 'omitnan');
        end

        if isempty(task.annotation)
            annotation_features = nan(1, 7);
            label = 'Unknown';
        else
            annotation_features = task.annotation{1, :};
            mean_score = annotation_features(end);
            if mean_score < 40
                label = 'Low';
            elseif mean_score <= 60
                label = 'Medium';
            else
                label = 'High';
            end
        end

        % Özellikleri birleştir
        all_features = {i, j, gaze_features(1), gaze_features(2), gaze_features(3), gaze_features(4), gaze_features(5), ...
                        low_freq_energy, high_freq_energy, ...
                        pupil_features(1), pupil_features(2), pupil_features(3), pupil_features(4), pupil_features(5), ...
                        blink_count, blink_duration, ...
                        annotation_features(1), annotation_features(2), annotation_features(3), ...
                        annotation_features(4), annotation_features(5), annotation_features(6), annotation_features(7), ...
                        label};

        % Satırı ekle
        features_labels = [features_labels; all_features];
    end
end

feature_names = {'Participant', 'Task', 'gaze_norm_pos_x', 'gaze_norm_pos_y', ...
                 'gaze_3d_x', 'gaze_3d_y', 'gaze_3d_z', 'low_freq_energy', ...
                 'high_freq_energy', 'pupil_diameter', 'pupil_confidence', ...
                 'sphere_3d_x', 'sphere_3d_y', 'sphere_3d_z', 'blink_count', ...
                 'blink_duration', 'mental', 'physical', 'temporal', ...
                 'performance', 'effort', 'frustration', 'mean', 'Label'};

% Tablo oluştur ve kaydet
features_table = cell2table(features_labels, 'VariableNames', feature_names);
output_file = 'features_and_labels.csv';
writetable(features_table, output_file);
disp(['Tablo başarıyla kaydedildi: ', output_file]);
