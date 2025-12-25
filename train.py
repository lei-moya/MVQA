import torch
import torch.nn as nn
import torch.optim as optim
from model import MultimodalVQAModel
from data_preprocessing import get_data_loader
from sklearn.metrics import mean_squared_error
import numpy as np

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # For regression

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for frames, audio, texts, masks, labels in train_loader:
            frames, audio, texts, masks, labels = frames.to(device), audio.to(device), texts.to(device), masks.to(device), labels.to(device)

            optimizer.zero_grad()
            segment_scores, global_scores = model(frames, audio, texts, masks)
            loss = criterion(global_scores.squeeze(), labels) + 0.5 * criterion(segment_scores.mean(dim=1).squeeze(), labels)  # Weighted loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for frames, audio, texts, masks, labels in val_loader:
                frames, audio, texts, masks, labels = frames.to(device), audio.to(device), texts.to(device), masks.to(device), labels.to(device)
                _, global_scores = model(frames, audio, texts, masks)
                val_preds.extend(global_scores.squeeze().cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_mse = mean_squared_error(val_labels, val_preds)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val MSE: {val_mse:.4f}")

    torch.save(model.state_dict(), 'multimodal_vqa_model.pth')

if __name__ == "__main__":
    # Dummy data
    video_paths = ["dummy"] * 10
    audio_paths = ["dummy.wav"] * 10
    text_data = [["dummy text"] * 10] * 10
    labels = np.random.rand(10)

    train_loader = get_data_loader(video_paths[:8], audio_paths[:8], text_data[:8], labels[:8])
    val_loader = get_data_loader(video_paths[8:], audio_paths[8:], text_data[8:], labels[8:])

    model = MultimodalVQAModel()
    train_model(model, train_loader, val_loader)