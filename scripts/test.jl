using Waves

data = load_episode_data.(readdir("data/episode1", join = true))
s, a = first(data)