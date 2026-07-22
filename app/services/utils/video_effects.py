from moviepy import Clip, ColorClip, CompositeVideoClip, vfx


# FadeIn
def fadein_transition(clip: Clip, t: float) -> Clip:
    return clip.with_effects([vfx.FadeIn(t)])


# FadeOut
def fadeout_transition(clip: Clip, t: float) -> Clip:
    return clip.with_effects([vfx.FadeOut(t)])


# SlideIn
def slidein_transition(clip: Clip, t: float, side: str) -> Clip:
    width, height = clip.size

    # MoviePy 内置 SlideIn 在当前这条处理链里对全屏素材不稳定，
    # 会出现“逻辑上应用了转场，但画面几乎看不出变化”的情况。
    # 这里改成显式黑底 + 位移动画，保证转场效果可见且行为可控。
    def position(current_time: float):
        progress = min(max(current_time / max(t, 0.001), 0), 1)

        if side == "left":
            return (-width + width * progress, 0)
        if side == "right":
            return (width - width * progress, 0)
        if side == "top":
            return (0, -height + height * progress)
        if side == "bottom":
            return (0, height - height * progress)
        return (0, 0)

    background = ColorClip(size=(width, height), color=(0, 0, 0)).with_duration(
        clip.duration
    )
    moving_clip = clip.with_position(position)
    return CompositeVideoClip([background, moving_clip], size=(width, height)).with_duration(
        clip.duration
    )


# ZoomPunch
def zoom_punch_transition(clip: Clip, t: float, punch_scale: float = 1.15) -> Clip:
    """新镜头出现时的"怼镜头"效果：从放大状态快速回落到原始大小，
    短视频剪辑里常用来强调切镜节奏，比 1 秒淡入/滑入更有冲击力。

    MoviePy 的 resized()（随时间变化的缩放）叠加 cropped() 在当前版本
    会产生逐帧尺寸错位、画面花屏；这里改用 transform() 直接对每一帧
    做像素级缩放+居中裁剪，一步到位，不经过会出问题的组合链路。
    """
    import numpy as np
    from PIL import Image

    width, height = clip.size

    def apply_zoom(get_frame, current_time):
        frame = get_frame(current_time)
        progress = min(max(current_time / max(t, 0.001), 0), 1)
        eased = 1 - (1 - progress) ** 3  # ease-out：快速回落而不是线性
        scale = punch_scale - (punch_scale - 1.0) * eased
        if abs(scale - 1.0) < 1e-3:
            return frame

        new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
        resized = Image.fromarray(frame).resize((new_w, new_h), Image.LANCZOS)
        left = max(0, (new_w - width) // 2)
        top = max(0, (new_h - height) // 2)
        cropped = resized.crop((left, top, left + width, top + height))
        return np.asarray(cropped)

    return clip.transform(apply_zoom)


# SlideOut
def slideout_transition(clip: Clip, t: float, side: str) -> Clip:
    width, height = clip.size
    transition_start = max(clip.duration - t, 0)

    # SlideOut 同样改成显式位移，保证片段末尾能稳定滑出画面。
    def position(current_time: float):
        if current_time <= transition_start:
            return (0, 0)

        progress = min(
            max((current_time - transition_start) / max(t, 0.001), 0), 1
        )

        if side == "left":
            return (-width * progress, 0)
        if side == "right":
            return (width * progress, 0)
        if side == "top":
            return (0, -height * progress)
        if side == "bottom":
            return (0, height * progress)
        return (0, 0)

    background = ColorClip(size=(width, height), color=(0, 0, 0)).with_duration(
        clip.duration
    )
    moving_clip = clip.with_position(position)
    return CompositeVideoClip([background, moving_clip], size=(width, height)).with_duration(
        clip.duration
    )
