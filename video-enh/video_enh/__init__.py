from gym.envs.registration import register

register(
    id='video-enh-v0',
    entry_point='video_enh.envs:VideoEnh',
)
