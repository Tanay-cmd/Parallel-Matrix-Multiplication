import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

# Initialize Faker and set seeds for reproducibility
fake = Faker()

# Constants
NUM_USERS = 1000
NUM_CHANNELS = 50
VIDEOS_PER_CHANNEL = 100
MAX_INTERACTIONS_PER_USER = 200  # Maximum loops; each loop yields 2 interactions => max interactions ~ 400

# Updated categories and tags
CATEGORIES = ["Anime", "Tech", "Gaming", "Music", "Education", "Sports", "Comedy", "Cooking", "Fitness", "News"]
probabilities = [0.4, 0.2, 0.3, 0.06, 0.02, 0.01, 0.01, 0, 0, 0]
p2 = [0.1, 0.1, 0.15, 0.15, 0.2, 0.1, 0.1, 0.05, 0.05, 0]
TAGS = {
    "Anime": [
        "Shonen", "Shojo", "Seinen", "Isekai", "Mecha", "Fantasy", "Slice of Life",
        "One Piece", "Attack on Titan", "Naruto", "Demon Slayer", "Jujutsu Kaisen",
        "Studio Ghibli", "My Hero Academia", "Cosplay", "AMV", "Anime Convention",
        "Voice Actors", "Manga Adaptation", "OVA", "Anime Review"
    ],
    "Tech": [
        "Python", "JavaScript", "Machine Learning", "Cloud Computing", "IoT",
        "Cybersecurity", "Blockchain", "Quantum Computing", "AR/VR", "Robotics",
        "Data Science", "Linux", "Open Source", "Startups", "Tech Reviews",
        "DevOps", "API Development", "Computer Vision", "Edge Computing", "5G"
    ],
    "Gaming": [
        "FPS", "RPG", "MMORPG", "Battle Royale", "Speedrunning", "Esports",
        "Game Development", "Retro Gaming", "Indie Games", "VR Gaming",
        "PC Gaming", "Console Wars", "Game Mods", "Walkthrough", "Let's Play",
        "Speedrun", "Game Analysis", "Mobile Gaming", "Simulation", "Survival"
    ],
    "Music": [
        "Pop", "Rock", "Hip-Hop", "EDM", "Classical", "Jazz", "K-Pop",
        "Music Production", "Guitar Covers", "Piano Tutorials", "Live Concerts",
        "Album Reviews", "Music Theory", "Songwriting", "Vinyl Collection",
        "Music Videos", "Lyrics Analysis", "DJ Sets", "Acoustic Sessions",
        "Music History"
    ],
    "Education": [
        "STEM", "Online Courses", "Coding Bootcamps", "Language Learning",
        "University Lectures", "Skill Development", "Career Advice",
        "Documentaries", "Science Experiments", "Book Summaries",
        "Critical Thinking", "Research Papers", "MOOCs", "Khan Academy",
        "Coursera", "TED Talks", "Educational Animation", "History Lessons",
        "Philosophy", "Financial Literacy"
    ],
    "Sports": [
        "Football", "Cricket", "Basketball", "Tennis", "Olympics",
        "Athletics", "Swimming", "Extreme Sports", "Sports Science",
        "Player Interviews", "Match Analysis", "Fantasy Leagues",
        "Sports Medicine", "Training Routines", "Esports",
        "Winter Sports", "Martial Arts", "Cycling", "Gymnastics",
        "Sports Nutrition"
    ],
    "Comedy": [
        "Standup Specials", "Sketch Comedy", "Improv", "Satire",
        "Dark Comedy", "Prank Videos", "Roast Battles", "Parody",
        "Comedy Podcasts", "Sitcom Clips", "Dad Jokes", "Memes",
        "Viral Challenges", "Comedy Roasts", "Wholesome Humor",
        "Political Satire", "Comedy Music", "Pun Compilations",
        "Fail Compilations", "Cringe Comedy"
    ],
    "Cooking": [
        "Gourmet", "Meal Prep", "Street Food", "Baking", "Vegan Recipes",
        "Keto Diet", "Food Challenges", "Culinary Travel", "Kitchen Hacks",
        "Food Science", "MasterChef", "Fermentation", "Mixology",
        "Food Photography", "Restaurant Reviews", "Historical Recipes",
        "Molecular Gastronomy", "Food Trucks", "Zero Waste Cooking",
        "Food Festivals"
    ],
    "Fitness": [
        "Calisthenics", "CrossFit", "Pilates", "Marathon Training",
        "Home Workouts", "Weight Loss Journey", "Physical Therapy",
        "Sports Nutrition", "Yoga Flow", "Powerlifting",
        "Mobility Training", "Athlete Training", "Gym Vlogs",
        "Fitness Challenges", "Bodybuilding", "HIIT Workouts",
        "Mind-Body Connection", "Recovery Techniques", "Fitness Gear",
        "Macro Counting"
    ],
    "News": [
        "Breaking News", "Political Analysis", "Tech News",
        "Environmental Issues", "Global Economy", "Health Updates",
        "Science News", "Entertainment News", "Investigative Journalism",
        "War Reporting", "Financial Markets", "Social Media Trends",
        "Cybersecurity Alerts", "Space Exploration", "Cultural Shifts",
        "Education Reform", "Sports Updates", "Celebrity News",
        "Local Events", "Human Interest"
    ]
}

# Parameterized probabilities and thresholds
PREFERRED_CHANNEL_PROB = 0.8   # For interactions based on followed channels
TRENDING_PROB = 0.05           # 5% of videos are trending
LIKE_THRESHOLD = 0.95          # Fully watched videos (>=95%) are liked
MIN_WATCH_FOR_DISLIKE = 0.1    # <10% watch indicates likely dislike

# Helper function: Generate a timestamp with recent bias
def get_weighted_ts(days=365):
    """Generate a datetime object between now and 'days' ago."""
    return fake.date_time_between(start_date=f'-{days}d', end_date='now')

###############################
# 1. Generate Users with 3 Followed Categories
###############################
users = []
for _ in range(NUM_USERS):
    # Each user follows exactly 3 unique categories
    followed_categories = list(np.random.choice(CATEGORIES, size=3, replace=False))
    followed_categories_str = ",".join(followed_categories)
    
    user = {
        "user_id": fake.uuid4(),
        "followed_categories": followed_categories_str,
        "preferred_length": np.random.choice(["short", "medium", "long"], p=[0.4, 0.5, 0.1]),
        "attention_span": np.clip(np.random.normal(loc=0.7, scale=0.2), 0, 1)
    }
    users.append(user)
users_df = pd.DataFrame(users)
print("Users generated.")

###############################
# 2. Generate Channels
###############################
channels = []
for _ in range(NUM_CHANNELS):
    channel = {
        "channel_id": fake.uuid4(),
        "primary_category": np.random.choice(CATEGORIES),
        "consistency": np.random.beta(2, 2),  # Likelihood to stick to primary category
        "upload_frequency": np.random.poisson(3)
    }
    channels.append(channel)
channels_df = pd.DataFrame(channels)
print("Channels generated.")

###############################
# 3. Generate Videos
###############################
videos = []
current_date = datetime.now()

for channel in channels:
    for _ in range(VIDEOS_PER_CHANNEL):
        # Determine video category: 80% chance to use channel's primary category; otherwise random.
        if random.random() < channel["consistency"]:
            category = channel["primary_category"]
        else:
            category = np.random.choice(CATEGORIES)
        
        # Generate tags as a comma-separated string
        tags = ",".join(np.random.choice(TAGS[category], size=2, replace=False))
        
        video = {
            "video_id": fake.uuid4(),
            "channel_id": channel["channel_id"],
            "category": category,
            "tags": tags,
            "length": np.random.choice(["short", "medium", "long"], p=[0.4, 0.5, 0.1]),
            "upload_date": get_weighted_ts(180),
            "views": 0,
            "is_trending": False
        }
        # Set trending videos based on TRENDING_PROB
        if random.random() < TRENDING_PROB:
            video["views"] = np.random.randint(50000, 100000)
            video["is_trending"] = True
        else:
            video["views"] = np.random.randint(100, 10000)
        videos.append(video)
videos_df = pd.DataFrame(videos)
print("Videos generated.")

###############################
# 4. Generate Interactions (Two Interactions per Loop)
###############################
# Fixed probabilities for followed categories (for the second type of interaction)
followed_probs = [0.5, 0.3, 0.2]

interactions = []
# Define recency threshold for newly uploaded videos (e.g., within last 7 days)
RECENCY_THRESHOLD = timedelta(days=7)

# For each user, set a minimum of 50 iterations (yielding at least 100 interactions)
for user in users:
    print(user)
    # Get the followed categories for this user (list of 3)
    followed_categories = user["followed_categories"].split(",")
    
    # Sample followed channels for the user based on each followed category separately:
    # For simplicity, we collect channels for each followed category in a dictionary.
    channels_by_cat = {}
    for cat in followed_categories:
        cat_channels = channels_df[channels_df["primary_category"] == cat]
        # If available, sample up to 3 channels per category; else, empty list
        sample_size = min(3, len(cat_channels))
        channels_by_cat[cat] = (
            cat_channels.sample(sample_size)["channel_id"].tolist() if sample_size > 0 else []
        )
    
    # Determine the number of loop iterations (each loop yields 2 interactions)
    num_loops = np.random.randint(50, MAX_INTERACTIONS_PER_USER)
    
    for _ in range(num_loops):
        # --- Interaction 1: Based on Followed Channels with newly uploaded feature ---
        # Randomly choose one followed category for channel-based interaction
        chosen_cat_for_channel = np.random.choice(followed_categories)
        followed_channels = channels_by_cat.get(chosen_cat_for_channel, [])
        
        if random.random() < PREFERRED_CHANNEL_PROB and followed_channels:
            candidate_videos = videos_df[
                (videos_df["channel_id"].isin(followed_channels)) &
                (videos_df["category"] == chosen_cat_for_channel)
            ]
            # Get current time for recency check
            now = datetime.now()
            # Further filter for newly uploaded videos within the recency threshold
            new_candidate_videos = candidate_videos[
                candidate_videos["upload_date"].apply(lambda x: (now - x) <= RECENCY_THRESHOLD)
            ]
            
            # Prefer newly uploaded video if available with a probability of 70%
            if not new_candidate_videos.empty and random.random() < 0.7:
                video1 = new_candidate_videos.sample(1).iloc[0]
            elif not candidate_videos.empty:
                video1 = candidate_videos.sample(1).iloc[0]
            else:
                video1 = videos_df.sample(1).iloc[0]
        else:
            # Fallback: select a video from the chosen category
            candidate_videos = videos_df[videos_df["category"] == chosen_cat_for_channel]
            if candidate_videos.empty:
                video1 = videos_df.sample(1).iloc[0]
            else:
                video1 = candidate_videos.sample(1).iloc[0]
        
        base_watch = user["attention_span"]
        # For simplicity, assume watch percentage calculation similar to before
        if video1["length"] == np.random.choice(["short", "medium", "long"], p=[0.4, 0.5, 0.1]):
            raw_watch_pct1 = base_watch + np.random.normal(0.2, 0.1)
        else:
            raw_watch_pct1 = base_watch - np.random.normal(0.3, 0.15)
        watch_pct1 = np.clip(raw_watch_pct1, 0, 1)
        
        interaction1 = {
            "user_id": user["user_id"],
            "video_id": video1["video_id"],
            "watch_percentage": watch_pct1,
            "liked": 1 if watch_pct1 >= LIKE_THRESHOLD else (1 if watch_pct1 > 0.8 and random.random() < 0.3 else 0),
            "disliked": 1 if watch_pct1 < MIN_WATCH_FOR_DISLIKE and random.random() < 0.7 else 0,
            "timestamp": get_weighted_ts(90)
        }
        if watch_pct1 >= LIKE_THRESHOLD:
            interaction1["liked"] = 1
            interaction1["disliked"] = 0
        
        interactions.append(interaction1)
        
        # --- Interaction 2: Based on Chosen Followed Category (using fixed probabilities) ---
        chosen_cat = np.random.choice(followed_categories, p=followed_probs)
        candidate_videos = videos_df[videos_df["category"] == chosen_cat]
        if candidate_videos.empty:
            video2 = videos_df.sample(1).iloc[0]
        else:
            video2 = candidate_videos.sample(1).iloc[0]
        
        if video2["length"] == np.random.choice(["short", "medium", "long"], p=[0.4, 0.5, 0.1]):
            raw_watch_pct2 = base_watch + np.random.normal(0.2, 0.1)
        else:
            raw_watch_pct2 = base_watch - np.random.normal(0.3, 0.15)
        watch_pct2 = np.clip(raw_watch_pct2, 0, 1)
        
        interaction2 = {
            "user_id": user["user_id"],
            "video_id": video2["video_id"],
            "watch_percentage": watch_pct2,
            "liked": 1 if watch_pct2 >= LIKE_THRESHOLD else (1 if watch_pct2 > 0.8 and random.random() < 0.3 else 0),
            "disliked": 1 if watch_pct2 < MIN_WATCH_FOR_DISLIKE and random.random() < 0.7 else 0,
            "timestamp": get_weighted_ts(90)
        }
        if watch_pct2 >= LIKE_THRESHOLD:
            interaction2["liked"] = 1
            interaction2["disliked"] = 0
        
        interactions.append(interaction2)
interactions_df = pd.DataFrame(interactions)
print("Interactions generated.")

###############################
# 5. Save Datasets
###############################
users_df.to_csv("users.csv", index=False)
channels_df.to_csv("channels.csv", index=False)
videos_df.to_csv("videos.csv", index=False)
interactions_df.to_csv("interactions.csv", index=False)

print("Synthetic dataset created successfully!")