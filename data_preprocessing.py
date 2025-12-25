import torch
import torchvision.transforms as transforms
from transformers import RobertaTokenizer
import librosa
import numpy as np
from PIL import Image

class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, video_paths, audio_paths, text_data, labels, segment_length=10, frame_rate=1):
        self.video_paths = video_paths
        self.audio_paths = audio_paths
        self.text_data = text_data  # List of lists: [[text1, text2, ...], ...]
        self.labels = labels
        self.segment_length = segment_length
        self.frame_rate = frame_rate

        # Transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.tokenizer = RobertaTokenizer.from_pretrained('hfl/roberta-wwm-ext')

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        # Load video frames (simplified: assume pre-extracted frames)
        frames = []
        for i in range(self.segment_length):
            # In practice, load from video_path
            frame = Image.open(f"{self.video_paths[idx]}/frame_{i}.jpg")  # Placeholder
            frames.append(self.image_transform(frame))
        frames = torch.stack(frames)  # (segment_length, 3, 224, 224)

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
        spec_segments = torch.tensor(np.array(spec_segments), dtype=torch.float32)  # (segment_length, 128, time)

        # Process texts (danmu)
        texts = self.text_data[idx]  # List of strings for segments
        tokenized_texts = []
        attention_masks = []
        for txt in texts:
            tokens = self.tokenizer(txt, padding='max_length', max_length=50, truncation=True, return_tensors='pt')
            tokenized_texts.append(tokens['input_ids'].squeeze())
            attention_masks.append(tokens['attention_mask'].squeeze())
        tokenized_texts = torch.stack(tokenized_texts)  # (segment_length, 50)
        attention_masks = torch.stack(attention_masks)  # (segment_length, 50)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return frames, spec_segments, tokenized_texts, attention_masks, label

# Example data loader
def get_data_loader(video_paths, audio_paths, text_data, labels, batch_size=4):
    dataset = MultimodalDataset(video_paths, audio_paths, text_data, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Placeholder for actual data
if __name__ == "__main__":
    # Dummy data
    video_paths = ["path/to/video1", "path/to/video2"]
    audio_paths = ["path/to/audio1.wav", "path/to/audio2.wav"]
    text_data = [["弹幕1", "弹幕2"], ["弹幕3", "弹幕4"]]
    labels = [0.8, 0.6]
    loader = get_data_loader(video_paths, audio_paths, text_data, labels)
    for batch in loader:
        print("Batch loaded")