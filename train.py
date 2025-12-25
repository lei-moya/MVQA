import tensorflow as tf
from model import MultimodalVQAModel
from data_preprocessing import get_tf_dataset
from sklearn.metrics import mean_squared_error
import numpy as np

def train_model(model, train_dataset, val_dataset, epochs=10, lr=1e-4):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.MeanSquaredError()

    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            segment_scores, global_scores = model(inputs)
            loss = loss_fn(global_scores, labels) + 0.5 * loss_fn(tf.reduce_mean(segment_scores, axis=1), labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    for epoch in range(epochs):
        train_loss = 0
        for batch in train_dataset:
            frames, audio, texts, masks, labels = batch
            inputs = [frames, audio, texts, masks]
            loss = train_step(inputs, labels)
            train_loss += loss.numpy()

        # Validation
        val_preds = []
        val_labels = []
        for batch in val_dataset:
            frames, audio, texts, masks, labels = batch
            inputs = [frames, audio, texts, masks]
            _, global_scores = model(inputs)
            val_preds.extend(global_scores.numpy().flatten())
            val_labels.extend(labels.numpy().flatten())

        val_mse = mean_squared_error(val_labels, val_preds)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_dataset):.4f}, Val MSE: {val_mse:.4f}")

    model.save_weights('multimodal_vqa_model.h5')

if __name__ == "__main__":
    # Dummy data
    video_paths = ["dummy"] * 10
    audio_paths = ["dummy.wav"] * 10
    text_data = [["dummy text"] * 10] * 10
    labels = np.random.rand(10)

    train_dataset = get_tf_dataset(video_paths[:8], audio_paths[:8], text_data[:8], labels[:8])
    val_dataset = get_tf_dataset(video_paths[8:], audio_paths[8:], text_data[8:], labels[8:])

    model = MultimodalVQAModel()
    train_model(model, train_dataset, val_dataset)