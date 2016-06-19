clc
clear all
close all
recObj = audiorecorder;
disp('Start speaking.')
try
    recordblocking(recObj, 5);
catch
    disp('Error')
    pause
    recordblocking(recObj, 5);
end

disp('End of Recording.');

% Play back the recording.
play(recObj);

% Store data in double-precision array.
myRecording = getaudiodata(recObj);

disp('Training Neural Network');

% Plot the waveform.
plot(myRecording);

audiowrite('test.wav',myRecording,8000);

[file,fs]=audioread('test.wav');
fs
%sound(file,fs);
%plot (file);
frame_duration=0.3;
frame_len=frame_duration*fs;
n=length(file);
num_frames=floor(n/frame_len);

inp=max(file)
thr=py.four.trainNetwork(inp)


new_sig=zeros(n,1);
count=0;


for k=1:num_frames
   frame=file((k-1)*frame_len+1:frame_len*k);
   max_val=max(frame);
   if (max_val>thr)
       count=count+1;
       new_sig((count-1)*frame_len+1:frame_len*count)=frame;
   end
end

disp ('After Removal')
pause

figure
plot (new_sig)
sound(new_sig,fs)

audiowrite('CleanFIle.wav',new_sig,fs)
strr='CleanFIle.wav'
recog=py.three.sperecog(strr)