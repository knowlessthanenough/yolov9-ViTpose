import torch
import torch.nn.functional as F
import os

def match_features_to_teams(player_feat_path, team_feat_path):
    # Load .pt files
    player_data = torch.load(player_feat_path)
    team_data = torch.load(team_feat_path)

    player_features = player_data['features']      # [N, 512]
    player_filenames = player_data['filenames']    # list of N strings

    team_features = team_data['features']          # [M, 512]
    team_names = team_data['filenames']           # list of M strings (can have duplicates)

    # Normalize for cosine similarity
    player_features = F.normalize(player_features, dim=1)
    team_features = F.normalize(team_features, dim=1)

    results = {}

    for i, crop_feat in enumerate(player_features):
        crop_name = os.path.splitext(os.path.basename(player_filenames[i]))[0]

        # Compute similarity to all team reference features
        sims = torch.matmul(team_features, crop_feat)  # [M]

        # Group scores by team name
        team_scores = {}
        for team, sim in zip(team_names, sims):
            team_key = os.path.splitext(os.path.basename(team))[0]
            team_scores.setdefault(team_key, []).append(sim.item())

        # Average scores per team
        team_avg_scores = {team: sum(scores)/len(scores) for team, scores in team_scores.items()}

        # Pick best team
        best_team = max(team_avg_scores.items(), key=lambda x: x[1])

        results[crop_name] = {
            "team": best_team[0],
            "score": round(best_team[1], 4)
        }

    return results


if __name__ == "__main__":
    results = match_features_to_teams("./reid_test_image/crop/crop.pt", "./reid_test_image/team/team.pt")
    print(results)