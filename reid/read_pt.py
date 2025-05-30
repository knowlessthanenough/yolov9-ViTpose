import torch
import torch.nn.functional as F
import os

def load_features(file_path):
    """Load features from a .pt file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    return torch.load(file_path)

if __name__ == "__main__":
    # Example usage
    team_feat_path = "./reid_test_image/team/team.pt"

    try:
        team_data = load_features(team_feat_path)
        print("team features loaded successfully.")
        print(team_data)

        # You can now use player_data and team_data as needed
    except Exception as e:
        print(f"Error loading features: {e}")