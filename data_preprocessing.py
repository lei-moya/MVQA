import tensorflow as tf
from transformers import RobertaTokenizer
import librosa
import numpy as np
from PIL import Image

class MultimodalDataset:
    def __init__(self, video_paths, audio_paths, text_data, labels, segment_length=10):
        self.video_paths = video_paths
        self.audio_paths = audio_paths
        self.text_data = text_data  # List of lists: [[text1, text2, ...], ...]
        self.labels = labels
        self.segment_length = segment_length

        self.tokenizer = RobertaTokenizer.from_pretrained('hfl/roberta-wwm-ext')

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        # Load video frames (simplified: assume pre-extracted frames)
        frames = []
        for i in range(self.segment_length):
            # In practice, load from video_path
            frame = Image.open(f"{self.video_paths[idx]}/frame_{i}.jpg")  # Placeholder
            frame = np.array(frame.resize((224, 224))) / 255.0  # Normalize
            frames.append(frame)
        frames = np.stack(frames)  # (segment_length, 224, 224, 3)

        # Load audio
        audio, sr = librosa.load(self.audio_paths[idx], sr=22050)
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        # Segment into parts
        spec_segments = []
        hop_length = len(spectrogram[0]) // self.segment_length
        for i in range(self.segment_length):
            start = i * hop_length
            end = (i + 1) * hop_length
            spec_segments.append(spectrogram[:, start:end])
        spec_segments = np.array(spec_segments)  # (segment_length, 128, time)

        # Process texts (danmu)
        texts = self.text_data[idx]  # List of strings for segments
        tokenized_texts = []
        attention_masks = []
        for txt in texts:
            tokens = self.tokenizer(txt, padding='max_length', max_length=50, truncation=True, return_tensors='tf')
            tokenized_texts.append(tokens['input_ids'].numpy())
            attention_masks.append(tokens['attention_mask'].numpy())
        tokenized_texts = np.stack(tokenized_texts)  # (segment_length, 50)
        attention_masks = np.stack(attention_masks)  # (segment_length, 50)

        label = np.array(self.labels[idx], dtype=np.float32)

        return frames, spec_segments, tokenized_texts, attention_masks, label

def get_tf_dataset(video_paths, audio_paths, text_data, labels, batch_size=4):
    dataset = tf.data.Dataset.from_tensor_slices((video_paths, audio_paths, text_data, labels))
    dataset = dataset.map(lambda vp, ap, td, l: tf.py_function(
        func=lambda vp, ap, td, l: MultimodalDataset([vp.numpy().decode()], [ap.numpy().decode()], [td.numpy()], [l.numpy()])[0],
        inp=[vp, ap, td, l],
        Tout=[tf.float32, tf.float32, tf.int32, tf.int32, tf.float32]
    ))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Placeholder for actual data
if __name__ == "__main__":
    # Dummy data
    video_paths = ["path/to/video1", "path/to/video2"]
    audio_paths = ["path/to/audio1.wav", "path/to/audio2.wav"]
    text_data = [["弹幕1", "弹幕2"], ["弹幕3", "弹幕4"]]
    labels = [0.8, 0.6]
    dataset = get_tf_dataset(video_paths, audio_paths, text_data, labels)
    for batch in dataset:
        print("Batch loaded")