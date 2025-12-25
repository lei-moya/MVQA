import torch
from model import MultimodalVQAModel
from data_preprocessing import MultimodalDataset
import matplotlib.pyplot as plt

def inference(model_path, video_path, audio_path, text_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultimodalVQAModel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Create dataset for single sample
    dataset = MultimodalDataset([video_path], [audio_path], [text_data], [0])  # Dummy label
    frames, audio, texts, masks, _ = dataset[0]
    frames = frames.unsqueeze(0).to(device)
    audio = audio.unsqueeze(0).to(device)
    texts = texts.unsqueeze(0).to(device)
    masks = masks.unsqueeze(0).to(device)

    with torch.no_grad():
        segment_scores, global_score = model(frames, audio, texts, masks)

    print(f"Global Quality Score: {global_score.item():.4f}")
    print(f"Segment Scores: {segment_scores.squeeze().cpu().numpy()}")

    # Plot segment scores
    plt.plot(segment_scores.squeeze().cpu().numpy())
    plt.title("Segment-Level Quality Scores")
    plt.xlabel("Segment")
    plt.ylabel("Score")
    plt.show()

if __name__ == "__main__":
    # Example
    inference('multimodal_vqa_model.pth', 'path/to/video', 'path/to/audio.wav', ['弹幕1', '弹幕2'])