import tensorflow as tf
from model import MultimodalVQAModel
from data_preprocessing import MultimodalDataset
import matplotlib.pyplot as plt

def inference(model_path, video_path, audio_path, text_data):
    model = MultimodalVQAModel()
    model.load_weights(model_path)

    # Create dataset for single sample
    dataset = MultimodalDataset([video_path], [audio_path], [text_data], [0])  # Dummy label
    frames, audio, texts, masks, _ = dataset[0]
    frames = tf.expand_dims(frames, axis=0)
    audio = tf.expand_dims(audio, axis=0)
    texts = tf.expand_dims(texts, axis=0)
    masks = tf.expand_dims(masks, axis=0)

    segment_scores, global_score = model([frames, audio, texts, masks])

    print(f"Global Quality Score: {global_score.numpy()[0][0]:.4f}")
    print(f"Segment Scores: {segment_scores.numpy()[0].flatten()}")

    # Plot segment scores
    plt.plot(segment_scores.numpy()[0].flatten())
    plt.title("Segment-Level Quality Scores")
    plt.xlabel("Segment")
    plt.ylabel("Score")
    plt.show()

if __name__ == "__main__":
    # Example
    inference('multimodal_vqa_model.h5', 'path/to/video', 'path/to/audio.wav', ['弹幕1', '弹幕2'])