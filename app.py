import math
import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import imageio.v2 as imageio


st.set_page_config(
    page_title="Project 03: Advanced Humanoid Movement Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)


DEFAULTS = {
    "duration_s": 10.0,
    "dt": 0.02,
    "seed": 42,

    "vx_cmd": 0.55,
    "vy_cmd": 0.0,
    "yaw_rate_cmd": 0.15,

    "step_height_m": 0.06,
    "slope_deg": 5.0,
    "enable_step": True,
    "step_x": 3.0,
    "step_length_m": 0.25,
    "foot_lateral_m": 0.12,

    "push_force_n": 90.0,
    "push_time_s": 5.0,
    "push_duration_s": 0.2,
    "push_angle_deg": 0.0,

    "nominal_height_m": 0.92,
    "mass_kg": 35.0,
    "gravity": 9.81,

    "tilt_warn_rad": 0.15,
    "tilt_fail_rad": 0.30,

    "obs_noise": 0.05,
    "friction_scale": 1.0,

    "max_video_frames": 120,
    "gif_max_frames": 90,
    "history_frames": 120,
    "min_fps": 5.0,

    "success_tracking_rmse_m_s": 0.38,
    "success_heading_deg": 75.0,
}


def _val(value, default):
    """Return default when a Streamlit widget value is None."""
    return default if value is None else value


def _unlink(path: str) -> None:
    """Safely remove a temporary file."""
    try:
        if os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp a scalar value to a closed interval."""
    return max(low, min(high, value))


def _ground_z(x: float, params: dict) -> float:
    """Return ground height at world x for slope plus optional localized step."""
    x = float(x)
    slope_rad = math.radians(float(params.get("slope_deg", DEFAULTS["slope_deg"])))
    z = x * math.tan(slope_rad)

    if bool(params.get("enable_step", DEFAULTS["enable_step"])):
        step_x = float(params.get("step_x", DEFAULTS["step_x"]))
        step_h = float(params.get("step_height_m", DEFAULTS["step_height_m"]))
        step_len = float(params.get("step_length_m", DEFAULTS["step_length_m"]))

        if step_x <= x <= step_x + step_len:
            z += step_h * math.sin(math.pi * (x - step_x) / max(step_len, 1e-6))

    return z


def _foot_world(
    com_x: float,
    com_y: float,
    yaw: float,
    rel_x: float,
    rel_y: float,
    lift: float,
    params: dict,
) -> tuple[float, float, float, bool]:
    """Transform a foot from body-relative coordinates to world coordinates."""
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)

    fx = com_x + rel_x * cos_y - rel_y * sin_y
    fy = com_y + rel_x * sin_y + rel_y * cos_y
    fz = _ground_z(fx, params) + lift
    contact = bool(lift <= 0.004)

    return fx, fy, fz, contact


def _world_velocity_vector(vx: float, vy: float, yaw: float) -> tuple[float, float]:
    """Convert body-frame velocity to world-frame xy vector."""
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    return vx * cos_y - vy * sin_y, vx * sin_y + vy * cos_y


def _compute_metrics(df: pd.DataFrame, params: dict) -> dict:
    """Compute summary metrics from one simulated run."""
    if df.empty:
        return {
            "status": "NO DATA",
            "fell": False,
            "fall_time_s": None,
            "tracking_rmse_m_s": float("nan"),
            "heading_error_deg": float("nan"),
            "cost_of_transport_proxy": float("nan"),
            "distance_m": 0.0,
            "energy_proxy_j": 0.0,
            "max_recovered_push_n": None,
            "safety_events": 0,
            "success": False,
        }

    mass = float(params.get("mass_kg", DEFAULTS["mass_kg"]))
    gravity = float(params.get("gravity", DEFAULTS["gravity"]))
    dt = float(params.get("dt", DEFAULTS["dt"]))

    vx_err = df["vx_meas"].to_numpy() - df["vx_cmd"].to_numpy()
    vy_err = df["vy_meas"].to_numpy() - df["vy_cmd"].to_numpy()
    tracking_rmse = float(np.sqrt(np.mean(vx_err * vx_err + vy_err * vy_err)))

    if len(df) > 1:
        dx = np.diff(df["com_x"].to_numpy())
        dy = np.diff(df["com_y"].to_numpy())
        distance = float(np.sum(np.hypot(dx, dy)))

        dz = np.diff(df["com_z"].to_numpy())
        vertical_work = float(mass * gravity * np.sum(np.clip(dz, 0.0, None)))
    else:
        distance = 0.0
        vertical_work = 0.0

    speed_sq = df["vx_meas"].to_numpy() ** 2 + df["vy_meas"].to_numpy() ** 2
    kinetic_proxy = float(0.018 * mass * np.sum(speed_sq) * dt)
    energy = vertical_work + kinetic_proxy
    cot = float(energy / (mass * gravity * max(distance, 1e-6)))

    final_yaw = float(df["yaw"].iloc[-1])
    expected_yaw = float(params.get("yaw_rate_cmd", DEFAULTS["yaw_rate_cmd"])) * float(
        params.get("duration_s", DEFAULTS["duration_s"])
    )
    heading_error_deg = float(abs(math.degrees(final_yaw - expected_yaw)))

    fell = bool(df["fall"].iloc[-1])
    fall_times = df.loc[df["fall_time"].notna(), "fall_time"]
    fall_time = float(fall_times.iloc[0]) if len(fall_times) > 0 else None

    push_force = float(params.get("push_force_n", DEFAULTS["push_force_n"]))
    if fell:
        status = "FELL"
    elif push_force > 1.0:
        status = "RECOVERED FROM PUSH"
    else:
        status = "STABLE"

    max_recovered_push = push_force if not fell else None
    safety_events = int(np.count_nonzero(df["safety_active"].to_numpy()))

    tracking_ok = tracking_rmse < float(
        params.get("success_tracking_rmse_m_s", DEFAULTS["success_tracking_rmse_m_s"])
    )
    heading_ok = heading_error_deg < float(
        params.get("success_heading_deg", DEFAULTS["success_heading_deg"])
    )
    success = bool((not fell) and tracking_ok and heading_ok)

    return {
        "status": status,
        "fell": fell,
        "fall_time_s": fall_time,
        "tracking_rmse_m_s": tracking_rmse,
        "heading_error_deg": heading_error_deg,
        "cost_of_transport_proxy": cot,
        "distance_m": distance,
        "energy_proxy_j": energy,
        "max_recovered_push_n": max_recovered_push,
        "safety_events": safety_events,
        "success": success,
    }


def simulate(params: dict) -> tuple[pd.DataFrame, dict]:
    """
    Simulate a reduced-order humanoid locomotion demo.

    This is intentionally lightweight and deterministic. It is not a full
    MuJoCo/Isaac humanoid solver, but it produces physically plausible
    whole-body style motion for evaluation and visualization.
    """
    rng = np.random.default_rng(int(params.get("seed", DEFAULTS["seed"])))

    dt = float(params.get("dt", DEFAULTS["dt"]))
    n_steps = max(50, int(float(params.get("duration_s", DEFAULTS["duration_s"])) / dt))

    mass = float(params.get("mass_kg", DEFAULTS["mass_kg"]))
    friction = _clamp(float(params.get("friction_scale", DEFAULTS["friction_scale"])), 0.35, 1.3)
    noise_scale = float(params.get("obs_noise", DEFAULTS["obs_noise"]))

    cmd_vx = float(params.get("vx_cmd", DEFAULTS["vx_cmd"]))
    cmd_vy = float(params.get("vy_cmd", DEFAULTS["vy_cmd"]))
    cmd_yaw = float(params.get("yaw_rate_cmd", DEFAULTS["yaw_rate_cmd"]))

    push_force = float(params.get("push_force_n", DEFAULTS["push_force_n"]))
    push_t = float(params.get("push_time_s", DEFAULTS["push_time_s"]))
    push_dur = float(params.get("push_duration_s", DEFAULTS["push_duration_s"]))
    push_angle = math.radians(float(params.get("push_angle_deg", DEFAULTS["push_angle_deg"])))

    step_height = float(params.get("step_height_m", DEFAULTS["step_height_m"]))
    tilt_warn = float(params.get("tilt_warn_rad", DEFAULTS["tilt_warn_rad"]))
    tilt_fail = float(params.get("tilt_fail_rad", DEFAULTS["tilt_fail_rad"]))
    foot_lateral = float(params.get("foot_lateral_m", DEFAULTS["foot_lateral_m"]))

    com_x = 0.0
    com_y = 0.0
    yaw = 0.0

    vx = 0.0
    vy = 0.0
    yaw_rate_meas = 0.0

    tilt = 0.0
    tilt_rate = 0.0
    crouch = 0.0

    phase = 0.0
    fall = False
    fall_time = float("nan")

    rows = []

    for i in range(n_steps):
        t = i * dt

        in_push = (not fall) and (push_t <= t <= push_t + push_dur)
        if in_push:
            acc = push_force / max(mass, 1e-3)
            vx += acc * math.cos(push_angle) * dt
            vy += acc * math.sin(push_angle) * dt
            tilt_rate += acc * 0.95 * dt

        warn = abs(tilt) > tilt_warn
        fail = abs(tilt) > tilt_fail

        if fail and not fall:
            fall = True
            fall_time = t

        safety = bool(warn and not fall)

        speed_scale = 0.22 if safety else 1.0
        target_vx = cmd_vx * speed_scale
        target_vy = cmd_vy * speed_scale
        target_yaw = cmd_yaw * speed_scale
        target_crouch = 0.18 if safety else 0.0

        if fall:
            vx *= 0.82
            vy *= 0.82
            yaw_rate_meas *= 0.82
            tilt_rate += 0.65 * dt
            crouch += (0.55 - crouch) * min(1.0, 4.0 * dt)
        else:
            gain = 2.0 * max(0.35, friction)
            slip_std = max(0.0, 1.0 - friction) * 0.35

            vx += (target_vx - vx) * gain * dt
            vx += rng.normal(0.0, noise_scale * 0.05 + slip_std * math.sqrt(dt))

            vy += (target_vy - vy) * gain * dt
            vy += rng.normal(0.0, noise_scale * 0.05 + slip_std * math.sqrt(dt))

            yaw_rate_meas += (target_yaw - yaw_rate_meas) * gain * dt
            yaw_rate_meas += rng.normal(0.0, noise_scale * 0.03)

            stiff = 24.0 if safety else 11.0
            damp = 8.5 if safety else 3.2
            tilt_process_noise = 0.010 + 0.08 * noise_scale

            tilt_accel = -stiff * tilt - damp * tilt_rate
            tilt_accel += rng.normal(0.0, tilt_process_noise)

            tilt_rate += tilt_accel * dt
            crouch += (target_crouch - crouch) * min(1.0, 3.5 * dt)

        tilt += tilt_rate * dt
        tilt = _clamp(tilt, -0.75, 0.75)
        tilt_rate = _clamp(tilt_rate, -3.5, 3.5)

        yaw += yaw_rate_meas * dt

        world_vx, world_vy = _world_velocity_vector(vx, vy, yaw)
        com_x += world_vx * dt
        com_y += world_vy * dt

        speed = math.hypot(vx, vy)

        if fall:
            step_freq = 0.0
            stride = 0.18
            left_lift = 0.0
            right_lift = 0.0
            left_rel_x = 0.20
            right_rel_x = -0.24
        else:
            step_freq = 0.55 + 0.95 * min(2.0, speed / 0.75)
            phase += 2.0 * math.pi * step_freq * dt

            stride = 0.09 + 0.075 * min(1.6, speed / 0.55)
            lift_gain = 0.35 + 0.65 * min(1.0, speed / 0.45)

            left_lift = step_height * max(0.0, math.sin(phase)) * lift_gain
            right_lift = step_height * max(0.0, math.sin(phase + math.pi)) * lift_gain

            left_rel_x = stride * math.cos(phase)
            right_rel_x = stride * math.cos(phase + math.pi)

        lx, ly, lz, lcontact = _foot_world(
            com_x, com_y, yaw, left_rel_x, foot_lateral, left_lift, params
        )
        rx, ry, rz, rcontact = _foot_world(
            com_x, com_y, yaw, right_rel_x, -foot_lateral, right_lift, params
        )

        ground_com = _ground_z(com_x, params)
        nominal = float(params.get("nominal_height_m", DEFAULTS["nominal_height_m"])) - crouch
        bob = 0.010 * abs(math.sin(2.0 * phase)) if not fall else 0.0
        com_z = ground_com + max(0.24, nominal) + bob

        rows.append(
            {
                "time": t,
                "com_x": com_x,
                "com_y": com_y,
                "com_z": com_z,
                "yaw": yaw,
                "vx_cmd": cmd_vx,
                "vy_cmd": cmd_vy,
                "yaw_rate_cmd": cmd_yaw,
                "vx_target": target_vx,
                "vy_target": target_vy,
                "vx_meas": vx,
                "vy_meas": vy,
                "yaw_rate_meas": yaw_rate_meas,
                "tilt": tilt,
                "tilt_rate": tilt_rate,
                "crouch": crouch,
                "safety_active": safety,
                "fall": fall,
                "fall_time": fall_time,
                "push_active": bool(in_push),
                "left_foot_x": lx,
                "left_foot_y": ly,
                "left_foot_z": lz,
                "left_contact": lcontact,
                "right_foot_x": rx,
                "right_foot_y": ry,
                "right_foot_z": rz,
                "right_contact": rcontact,
            }
        )

    df = pd.DataFrame(rows)
    metrics = _compute_metrics(df, params)
    return df, metrics


def _ensure_even_frame(frame: np.ndarray) -> np.ndarray:
    """
    Ensure frame height and width are even.

    This is required for yuv420p MP4/H.264 encoding. The previous error was
    caused by an odd frame height such as 391.
    """
    frame = np.asarray(frame, dtype=np.uint8)

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Frame must have shape (H, W, 3).")

    h, w, _ = frame.shape

    if h % 2 == 0 and w % 2 == 0:
        return frame

    new_h = h + (h % 2)
    new_w = w + (w % 2)

    out = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    out[:h, :w, :] = frame

    return out


def _figure_to_array(fig) -> np.ndarray:
    """Convert a Matplotlib figure to an even-sized RGB NumPy array."""
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    plt.close(fig)

    frame = img[:, :, :3].copy()
    return _ensure_even_frame(frame)


def _draw_frame(df: pd.DataFrame, idx: int, params: dict) -> np.ndarray:
    """Render one video frame from the simulated trajectory."""
    row = df.iloc[idx]

    history = int(params.get("history_frames", DEFAULTS["history_frames"]))
    start = max(0, idx - history)
    past = df.iloc[start : idx + 1]

    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(8.8, 4.6), dpi=85)

    # -------------------------
    # Top-down view
    # -------------------------
    ax_top.plot(
        past["com_x"].to_numpy(),
        past["com_y"].to_numpy(),
        color="#1f77b4",
        lw=1.6,
        alpha=0.75,
        label="CoM trace",
    )

    ax_top.scatter(
        float(row["com_x"]),
        float(row["com_y"]),
        s=75,
        color="red",
        zorder=6,
        label="CoM",
    )

    feet_top = [
        (
            float(row["left_foot_x"]),
            float(row["left_foot_y"]),
            bool(row["left_contact"]),
            "L",
        ),
        (
            float(row["right_foot_x"]),
            float(row["right_foot_y"]),
            bool(row["right_contact"]),
            "R",
        ),
    ]

    for fx, fy, contact, label in feet_top:
        if bool(row["fall"]):
            color = "#7f7f7f"
        elif contact:
            color = "#2ca02c"
        else:
            color = "#ff7f0e"

        ax_top.scatter(fx, fy, s=52, color=color, edgecolor="black", zorder=7)
        ax_top.text(
            fx,
            fy,
            label,
            fontsize=7,
            ha="center",
            va="center",
            color="white",
            zorder=8,
        )

    cmd_dx, cmd_dy = _world_velocity_vector(
        float(row["vx_cmd"]), float(row["vy_cmd"]), float(row["yaw"])
    )
    cmd_mag = math.hypot(cmd_dx, cmd_dy)
    if cmd_mag > 0.02:
        scale = 0.55 / cmd_mag
        ax_top.arrow(
            float(row["com_x"]),
            float(row["com_y"]),
            cmd_dx * scale,
            cmd_dy * scale,
            head_width=0.055,
            head_length=0.075,
            fc="#9467bd",
            ec="#9467bd",
            alpha=0.95,
            zorder=5,
            label="Velocity command",
        )

    meas_dx, meas_dy = _world_velocity_vector(
        float(row["vx_meas"]), float(row["vy_meas"]), float(row["yaw"])
    )
    meas_mag = math.hypot(meas_dx, meas_dy)
    if meas_mag > 0.02:
        scale = 0.45 / meas_mag
        ax_top.arrow(
            float(row["com_x"]),
            float(row["com_y"]),
            meas_dx * scale,
            meas_dy * scale,
            head_width=0.045,
            head_length=0.065,
            fc="#2ca02c",
            ec="#2ca02c",
            alpha=0.95,
            zorder=5,
            label="Measured velocity",
        )

    margin = 1.7 + 0.35 * abs(float(row["vx_meas"]))
    ax_top.set_xlim(float(row["com_x"]) - margin, float(row["com_x"]) + margin)
    ax_top.set_ylim(float(row["com_y"]) - margin, float(row["com_y"]) + margin)
    ax_top.set_aspect("equal")
    ax_top.grid(True, alpha=0.28)
    ax_top.set_xlabel("x (m)")
    ax_top.set_ylabel("y (m)")
    ax_top.set_title("Top-down: CoM, feet, velocity")
    ax_top.legend(loc="upper left", fontsize=7, framealpha=0.75)

    # -------------------------
    # Sagittal view
    # -------------------------
    x_min = float(row["com_x"]) - 2.0
    x_max = float(row["com_x"]) + 2.0
    xs = np.linspace(x_min, x_max, 220)
    gs = np.array([_ground_z(float(x), params) for x in xs], dtype=float)

    ax_side.plot(xs, gs, color="saddlebrown", lw=3.0, label="Ground")
    ax_side.fill_between(xs, gs, -0.25, color="saddlebrown", alpha=0.18)

    ax_side.plot(
        past["com_x"].to_numpy(),
        past["com_z"].to_numpy(),
        color="red",
        lw=1.5,
        alpha=0.75,
        label="CoM height",
    )

    ax_side.scatter(
        float(row["com_x"]),
        float(row["com_z"]),
        s=70,
        color="red",
        zorder=6,
    )

    feet_side = [
        (
            float(row["left_foot_x"]),
            float(row["left_foot_z"]),
            bool(row["left_contact"]),
        ),
        (
            float(row["right_foot_x"]),
            float(row["right_foot_z"]),
            bool(row["right_contact"]),
        ),
    ]

    for fx, fz, contact in feet_side:
        if bool(row["fall"]):
            color = "#7f7f7f"
        elif contact:
            color = "#2ca02c"
        else:
            color = "#ff7f0e"

        ax_side.plot(
            [float(row["com_x"]), fx],
            [float(row["com_z"]), fz],
            color="black",
            lw=2.1,
            zorder=4,
        )
        ax_side.scatter(fx, fz, s=42, color=color, edgecolor="black", zorder=5)

    head_z = float(row["com_z"]) + 0.28 - 0.5 * float(row["crouch"])
    ax_side.plot(
        [float(row["com_x"]), float(row["com_x"])],
        [float(row["com_z"]), head_z],
        color="black",
        lw=3.8,
        zorder=3,
    )
    ax_side.scatter(
        float(row["com_x"]),
        head_z,
        s=65,
        color="black",
        zorder=4,
        label="Head",
    )

    ax_side.set_xlim(x_min, x_max)
    ax_side.set_ylim(-0.25, max(1.55, float(row["com_z"]) + 0.75))
    ax_side.grid(True, alpha=0.28)
    ax_side.set_xlabel("x (m)")
    ax_side.set_ylabel("z (m)")
    ax_side.set_title("Sagittal: slope, step, CoM, feet")
    ax_side.legend(loc="upper left", fontsize=7, framealpha=0.75)

    if bool(row["fall"]):
        status = "FALL"
        status_color = "red"
    elif bool(row["safety_active"]):
        status = "SAFETY ACTIVE"
        status_color = "#d62728"
    elif bool(row["push_active"]):
        status = "PUSH"
        status_color = "#ff7f0e"
    else:
        status = "STABLE"
        status_color = "#2ca02c"

    hud = (
        f"t={float(row['time']):.2f}s | "
        f"vx={float(row['vx_meas']):.2f}/{float(row['vx_cmd']):.2f} m/s | "
        f"vy={float(row['vy_meas']):.2f}/{float(row['vy_cmd']):.2f} m/s | "
        f"yaw={math.degrees(float(row['yaw'])):.1f} deg | "
        f"tilt={math.degrees(float(row['tilt'])):.1f} deg"
    )

    fig.text(0.01, 0.035, hud, fontsize=8.2)
    fig.text(
        0.99,
        0.035,
        status,
        color=status_color,
        ha="right",
        fontsize=10,
        fontweight="bold",
    )
    fig.suptitle("Project 03: Advanced Humanoid Movement Demo", fontsize=11)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    return _figure_to_array(fig)


def _frame_indices(n: int, max_frames: int = 120) -> np.ndarray:
    """Return evenly spaced frame indices for video rendering."""
    if n <= 0:
        return np.array([], dtype=int)

    count = min(int(max_frames), int(n))
    return np.unique(np.linspace(0, n - 1, count).astype(int))


def _fps_for_indices(count: int, duration: float, min_fps: float = 5.0) -> float:
    """Compute a safe FPS value for the selected number of frames."""
    duration = max(float(duration), 0.1)
    fps = float(count) / duration
    return max(float(min_fps), fps)


def _video_writer_candidates(fps: float):
    """
    Yield video writer attempts.

    Order:
    1. MP4 / H.264 with faststart
    2. MP4 / H.264 with lighter settings
    3. MP4 / H.264 minimal settings
    4. WebM / VP8 fallback
    """
    base_mp4 = {
        "fps": float(fps),
        "codec": "libx264",
        "pixelformat": "yuv420p",
        "macro_block_size": 1,
    }

    yield (
        "mp4",
        dict(
            base_mp4,
            ffmpeg_params=[
                "-crf",
                "30",
                "-preset",
                "ultrafast",
                "-movflags",
                "+faststart",
            ],
        ),
    )

    yield (
        "mp4",
        dict(
            base_mp4,
            ffmpeg_params=[
                "-crf",
                "32",
                "-preset",
                "ultrafast",
            ],
        ),
    )

    yield ("mp4", dict(base_mp4))

    yield (
        "mp4",
        {
            "fps": float(fps),
            "codec": "libx264",
            "pixelformat": "yuv420p",
        },
    )

    yield (
        "webm",
        {
            "fps": float(fps),
            "codec": "libvpx",
            "pixelformat": "yuv420p",
            "macro_block_size": 1,
        },
    )

    yield (
        "webm",
        {
            "fps": float(fps),
            "codec": "libvpx",
            "pixelformat": "yuv420p",
        },
    )


def _write_video(
    df: pd.DataFrame,
    params: dict,
    suffix: str,
    writer_kwargs: dict,
) -> bytes:
    """Try to write an MP4 or WebM file and return its bytes."""
    max_frames = int(params.get("max_video_frames", DEFAULTS["max_video_frames"]))
    indices = _frame_indices(len(df), max_frames=max_frames)

    if len(indices) == 0:
        return b""

    fd, tmp_path = tempfile.mkstemp(suffix=f".{suffix}")
    os.close(fd)

    try:
        writer = imageio.get_writer(tmp_path, **writer_kwargs)
    except Exception:
        _unlink(tmp_path)
        return b""

    write_failed = False

    try:
        for idx in indices:
            frame = _draw_frame(df, int(idx), params)
            writer.append_data(frame)
    except Exception:
        write_failed = True

    try:
        writer.close()
    except Exception:
        pass

    if write_failed:
        _unlink(tmp_path)
        return b""

    try:
        with open(tmp_path, "rb") as f:
            data = f.read()
    except Exception:
        data = b""

    _unlink(tmp_path)

    if len(data) < 1000:
        return b""

    return data


def _write_gif(df: pd.DataFrame, params: dict) -> bytes:
    """Fallback animated GIF writer using Pillow when available."""
    max_frames = int(params.get("gif_max_frames", DEFAULTS["gif_max_frames"]))
    min_fps = float(params.get("min_fps", DEFAULTS["min_fps"]))

    indices = _frame_indices(len(df), max_frames=max_frames)
    if len(indices) == 0:
        return b""

    frames = []

    try:
        for idx in indices:
            frames.append(_draw_frame(df, int(idx), params))
    except Exception:
        return b""

    fps = _fps_for_indices(
        len(frames),
        float(params.get("duration_s", DEFAULTS["duration_s"])),
        min_fps,
    )

    fd, tmp_path = tempfile.mkstemp(suffix=".gif")
    os.close(fd)

    try:
        from PIL import Image

        pil_frames = [Image.fromarray(frame) for frame in frames]
        pil_frames[0].save(
            tmp_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000.0 / max(fps, 1.0)),
            loop=0,
            optimize=False,
        )

        with open(tmp_path, "rb") as f:
            data = f.read()
    except Exception:
        data = b""

    _unlink(tmp_path)

    if len(data) < 100:
        return b""

    return data


def render_video(df: pd.DataFrame, params: dict) -> tuple[bytes, str]:
    """
    Render the simulated trajectory to a video byte payload.

    Returns:
        (video_bytes, kind)
        kind is one of: mp4, webm, gif, none
    """
    if df.empty:
        return b"", "none"

    max_frames = int(params.get("max_video_frames", DEFAULTS["max_video_frames"]))
    min_fps = float(params.get("min_fps", DEFAULTS["min_fps"]))
    duration = float(params.get("duration_s", DEFAULTS["duration_s"]))

    indices = _frame_indices(len(df), max_frames=max_frames)
    fps = _fps_for_indices(len(indices), duration, min_fps)

    for suffix, writer_kwargs in _video_writer_candidates(fps):
        data = _write_video(df, params, suffix, writer_kwargs)
        if data:
            return data, suffix

    gif_data = _write_gif(df, params)
    if gif_data:
        return gif_data, "gif"

    return b"", "none"


def _static_trajectory_figure(df: pd.DataFrame, params: dict):
    """Static fallback plot if all video encoders fail."""
    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(10.0, 4.2), dpi=95)

    if df.empty:
        ax_top.text(0.5, 0.5, "No trajectory data", ha="center", va="center")
        ax_side.text(0.5, 0.5, "No trajectory data", ha="center", va="center")
        fig.suptitle("Static trajectory fallback")
        fig.tight_layout()
        return fig

    ax_top.plot(
        df["com_x"].to_numpy(),
        df["com_y"].to_numpy(),
        color="#1f77b4",
        lw=1.8,
        label="CoM trace",
    )

    last = df.iloc[-1]

    ax_top.scatter(
        float(last["com_x"]),
        float(last["com_y"]),
        s=80,
        color="red",
        zorder=6,
        label="Final CoM",
    )

    ax_top.scatter(
        float(last["left_foot_x"]),
        float(last["left_foot_y"]),
        s=55,
        color="#2ca02c" if bool(last["left_contact"]) else "#ff7f0e",
        edgecolor="black",
        label="Left foot",
    )

    ax_top.scatter(
        float(last["right_foot_x"]),
        float(last["right_foot_y"]),
        s=55,
        color="#2ca02c" if bool(last["right_contact"]) else "#ff7f0e",
        edgecolor="black",
        label="Right foot",
    )

    ax_top.set_aspect("equal")
    ax_top.grid(alpha=0.3)
    ax_top.set_xlabel("x (m)")
    ax_top.set_ylabel("y (m)")
    ax_top.set_title("Top-down trajectory")
    ax_top.legend(fontsize=7)

    x_min = float(df["com_x"].min()) - 1.0
    x_max = float(df["com_x"].max()) + 1.0
    xs = np.linspace(x_min, x_max, 300)
    gs = np.array([_ground_z(float(x), params) for x in xs], dtype=float)

    ax_side.plot(xs, gs, color="saddlebrown", lw=3.0, label="Ground")
    ax_side.fill_between(xs, gs, -0.25, color="saddlebrown", alpha=0.18)

    ax_side.plot(
        df["com_x"].to_numpy(),
        df["com_z"].to_numpy(),
        color="red",
        lw=1.8,
        label="CoM height",
    )

    ax_side.scatter(
        float(last["com_x"]),
        float(last["com_z"]),
        s=80,
        color="red",
        zorder=6,
    )

    ax_side.grid(alpha=0.3)
    ax_side.set_xlabel("x (m)")
    ax_side.set_ylabel("z (m)")
    ax_side.set_title("Sagittal trajectory")
    ax_side.legend(fontsize=7)

    fig.suptitle("Static trajectory fallback")
    fig.tight_layout()

    return fig


def robustness_sweep(params: dict, seeds: tuple[int, ...] = (0, 1, 2)) -> pd.DataFrame:
    """Run a small multi-seed robustness check without rendering video."""
    rows = []

    for seed in seeds:
        p = dict(params)
        p["seed"] = int(seed)
        _, m = simulate(p)

        rows.append(
            {
                "seed": seed,
                "status": m["status"],
                "fell": m["fell"],
                "tracking_rmse_m_s": m["tracking_rmse_m_s"],
                "heading_error_deg": m["heading_error_deg"],
                "cost_of_transport_proxy": m["cost_of_transport_proxy"],
                "max_recovered_push_n": m["max_recovered_push_n"],
            }
        )

    return pd.DataFrame(rows)


def _fmt(value, unit: str = "", digits: int = 2) -> str:
    """Format metric values safely for Streamlit metrics."""
    if value is None:
        return "n/a"

    try:
        return f"{float(value):.{digits}f}{unit}"
    except Exception:
        return str(value)


def _show_video(data: bytes, mime: str) -> bool:
    """
    Display a video using only arguments supported by the installed Streamlit.

    This avoids errors such as:
        TypeError: MediaMixin.video() got an unexpected keyword argument 'controls'
    """
    if not data:
        return False

    attempts = (
        {"format": mime, "loop": True, "autoplay": True, "muted": True},
        {"format": mime, "loop": True, "autoplay": True},
        {"format": mime, "loop": True},
        {"format": mime},
        {},
    )

    for kwargs in attempts:
        try:
            st.video(data, **kwargs)
            return True
        except TypeError:
            continue
        except Exception:
            return False

    return False


def _show_image(data: bytes) -> bool:
    """Display an image/GIF using only supported arguments."""
    if not data:
        return False

    attempts = (
        {"use_container_width": True},
        {},
    )

    for kwargs in attempts:
        try:
            st.image(data, **kwargs)
            return True
        except TypeError:
            continue
        except Exception:
            break

    fd, tmp_path = tempfile.mkstemp(suffix=".gif")
    os.close(fd)

    try:
        with open(tmp_path, "wb") as f:
            f.write(data)

        for kwargs in attempts:
            try:
                st.image(tmp_path, **kwargs)
                return True
            except TypeError:
                continue
            except Exception:
                return False
    except Exception:
        return False
    finally:
        _unlink(tmp_path)

    return False


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------

st.title("Project 03: Advanced Humanoid Movement")
st.caption(
    "Self-contained Streamlit demo. The app automatically loads synthetic "
    "humanoid movement data, runs a reduced-order simulation, generates a "
    "video inside the app, and exposes interactive controls. Simulation only."
)

with st.sidebar:
    st.header("Simulation controls")

    with st.form("control_form"):
        duration_s = st.number_input(
            "Duration (s)",
            min_value=4.0,
            max_value=20.0,
            value=float(DEFAULTS["duration_s"]),
            step=1.0,
        )

        seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=100000,
            value=int(DEFAULTS["seed"]),
            step=1,
        )

        st.subheader("Velocity command")
        vx_cmd = st.slider(
            "Forward vx (m/s)",
            min_value=-0.3,
            max_value=0.8,
            value=float(DEFAULTS["vx_cmd"]),
            step=0.01,
        )
        vy_cmd = st.slider(
            "Lateral vy (m/s)",
            min_value=-0.3,
            max_value=0.3,
            value=float(DEFAULTS["vy_cmd"]),
            step=0.01,
        )
        yaw_rate_cmd = st.slider(
            "Yaw rate (rad/s)",
            min_value=-0.8,
            max_value=0.8,
            value=float(DEFAULTS["yaw_rate_cmd"]),
            step=0.01,
        )

        st.subheader("Terrain and gait")
        enable_step = st.checkbox(
            "Enable localized step",
            value=bool(DEFAULTS["enable_step"]),
        )
        step_height_m = st.slider(
            "Step height (m)",
            min_value=0.0,
            max_value=0.08,
            value=float(DEFAULTS["step_height_m"]),
            step=0.005,
        )
        slope_deg = st.slider(
            "Slope (deg)",
            min_value=-10.0,
            max_value=10.0,
            value=float(DEFAULTS["slope_deg"]),
            step=0.5,
        )
        step_x = st.number_input(
            "Step x location (m)",
            value=float(DEFAULTS["step_x"]),
            step=0.1,
        )

        st.subheader("Push perturbation")
        push_force_n = st.slider(
            "Push force (N)",
            min_value=0.0,
            max_value=220.0,
            value=float(DEFAULTS["push_force_n"]),
            step=5.0,
        )
        push_time_s = st.number_input(
            "Push time (s)",
            min_value=0.0,
            max_value=18.0,
            value=float(DEFAULTS["push_time_s"]),
            step=0.1,
        )
        push_duration_s = st.slider(
            "Push duration (s)",
            min_value=0.05,
            max_value=0.5,
            value=float(DEFAULTS["push_duration_s"]),
            step=0.05,
        )
        push_angle_deg = st.slider(
            "Push angle (deg)",
            min_value=-180.0,
            max_value=180.0,
            value=float(DEFAULTS["push_angle_deg"]),
            step=5.0,
        )

        st.subheader("Robustness")
        friction_scale = st.slider(
            "Friction scale",
            min_value=0.35,
            max_value=1.3,
            value=float(DEFAULTS["friction_scale"]),
            step=0.05,
        )
        obs_noise = st.slider(
            "Observation noise",
            min_value=0.0,
            max_value=0.2,
            value=float(DEFAULTS["obs_noise"]),
            step=0.01,
        )

        submitted = st.form_submit_button("Generate live demo")

    st.divider()
    st.markdown(
        "**Safety statement:** this is a simulation-only demo. "
        "Real humanoid testing requires a gantry/harness, torque and speed "
        "limits, safe-lowering behavior, hardware emergency stop, and staged tests."
    )


params = {
    "duration_s": float(_val(duration_s, DEFAULTS["duration_s"])),
    "dt": float(DEFAULTS["dt"]),
    "seed": int(_val(seed, DEFAULTS["seed"])),

    "vx_cmd": float(_val(vx_cmd, DEFAULTS["vx_cmd"])),
    "vy_cmd": float(_val(vy_cmd, DEFAULTS["vy_cmd"])),
    "yaw_rate_cmd": float(_val(yaw_rate_cmd, DEFAULTS["yaw_rate_cmd"])),

    "step_height_m": float(_val(step_height_m, DEFAULTS["step_height_m"])),
    "slope_deg": float(_val(slope_deg, DEFAULTS["slope_deg"])),
    "enable_step": bool(_val(enable_step, DEFAULTS["enable_step"])),
    "step_x": float(_val(step_x, DEFAULTS["step_x"])),
    "step_length_m": float(DEFAULTS["step_length_m"]),
    "foot_lateral_m": float(DEFAULTS["foot_lateral_m"]),

    "push_force_n": float(_val(push_force_n, DEFAULTS["push_force_n"])),
    "push_time_s": float(_val(push_time_s, DEFAULTS["push_time_s"])),
    "push_duration_s": float(_val(push_duration_s, DEFAULTS["push_duration_s"])),
    "push_angle_deg": float(_val(push_angle_deg, DEFAULTS["push_angle_deg"])),

    "nominal_height_m": float(DEFAULTS["nominal_height_m"]),
    "mass_kg": float(DEFAULTS["mass_kg"]),
    "gravity": float(DEFAULTS["gravity"]),

    "tilt_warn_rad": float(DEFAULTS["tilt_warn_rad"]),
    "tilt_fail_rad": float(DEFAULTS["tilt_fail_rad"]),

    "obs_noise": float(_val(obs_noise, DEFAULTS["obs_noise"])),
    "friction_scale": float(_val(friction_scale, DEFAULTS["friction_scale"])),

    "max_video_frames": int(DEFAULTS["max_video_frames"]),
    "gif_max_frames": int(DEFAULTS["gif_max_frames"]),
    "history_frames": int(DEFAULTS["history_frames"]),
    "min_fps": float(DEFAULTS["min_fps"]),

    "success_tracking_rmse_m_s": float(DEFAULTS["success_tracking_rmse_m_s"]),
    "success_heading_deg": float(DEFAULTS["success_heading_deg"]),
}


first_run = "last_result" not in st.session_state
should_generate = bool(submitted or first_run)


if should_generate:
    sim_df = pd.DataFrame()
    metrics = {}
    video_bytes = b""
    video_kind = "none"
    error_message = None

    try:
        with st.spinner("Simulating humanoid movement and encoding video..."):
            sim_df, metrics = simulate(params)
            video_bytes, video_kind = render_video(sim_df, params)

        if not video_bytes:
            error_message = (
                "Video encoder did not produce a file. "
                "Showing static trajectory fallback."
            )

        st.session_state.last_result = {
            "df": sim_df,
            "metrics": metrics,
            "video": video_bytes,
            "kind": video_kind,
            "params": params.copy(),
            "error": error_message,
        }

        if error_message:
            st.warning(error_message)

    except Exception as exc:
        previous = st.session_state.get("last_result")

        if previous is not None and previous.get("video"):
            st.warning(
                f"Could not regenerate demo: {exc}. "
                "Showing previous successful demo."
            )
        else:
            st.session_state.last_result = {
                "df": sim_df,
                "metrics": metrics,
                "video": b"",
                "kind": "none",
                "params": params.copy(),
                "error": str(exc),
            }


result = st.session_state.get("last_result")

if result is None:
    st.error("No demo result is available.")
    st.stop()


df = result.get("df", pd.DataFrame())
metrics = result.get("metrics", {})
video_bytes = result.get("video", b"")
video_kind = result.get("kind", "none")
active_params = result.get("params", params)
result_error = result.get("error")


if df.empty and not video_bytes:
    st.error(f"Demo generation failed: {result_error}")
    st.info(
        "Install the requirements, then rerun the app. "
        "If video encoding still fails, the app will show a static fallback."
    )
    st.stop()


if result_error and not video_bytes:
    st.warning(result_error)


tab_demo, tab_metrics, tab_failure, tab_repro = st.tabs(
    ["Live demo", "Metrics", "Failure and safety", "Reproduce"]
)


with tab_demo:
    left, right = st.columns([3, 1])

    with left:
        shown = False

        if video_kind in {"mp4", "webm"} and video_bytes:
            fmt = "video/mp4" if video_kind == "mp4" else "video/webm"
            shown = _show_video(video_bytes, fmt)

        elif video_kind == "gif" and video_bytes:
            shown = _show_image(video_bytes)
            if shown:
                st.info(
                    "H.264/VP8 video encoding was unavailable, so an animated GIF "
                    "fallback is shown."
                )

        if not shown:
            st.warning("Video encoding failed or is unsupported. Showing static trajectory fallback.")
            if not df.empty:
                st.pyplot(
                    _static_trajectory_figure(df, active_params),
                    clear_figure=True,
                )

    with right:
        st.metric("Status", metrics.get("status", "n/a"))
        st.metric(
            "Velocity RMSE",
            _fmt(metrics.get("tracking_rmse_m_s"), " m/s", 3),
        )
        st.metric(
            "Heading error",
            _fmt(metrics.get("heading_error_deg"), " deg", 1),
        )
        st.metric(
            "CoT proxy",
            _fmt(metrics.get("cost_of_transport_proxy"), "", 3),
        )
        st.metric(
            "Fall time",
            _fmt(metrics.get("fall_time_s"), " s", 2),
        )

        if video_bytes:
            if video_kind == "mp4":
                file_name = f"project03_humanoid_demo_seed{int(active_params['seed'])}.mp4"
                mime = "video/mp4"
                label = "Download generated MP4"
            elif video_kind == "webm":
                file_name = f"project03_humanoid_demo_seed{int(active_params['seed'])}.webm"
                mime = "video/webm"
                label = "Download generated WebM"
            elif video_kind == "gif":
                file_name = f"project03_humanoid_demo_seed{int(active_params['seed'])}.gif"
                mime = "image/gif"
                label = "Download generated GIF"
            else:
                file_name = ""
                mime = ""
                label = ""

            if file_name:
                st.download_button(
                    label,
                    data=video_bytes,
                    file_name=file_name,
                    mime=mime,
                    use_container_width=True,
                )

        st.caption(
            "The video is generated locally inside the Streamlit app. "
            "No external video service or API key is used."
        )

    with st.expander("Legend and modeling assumptions"):
        st.markdown(
            """
            - **Red dot:** center of mass, CoM.
            - **Blue trace:** recent CoM trajectory.
            - **Green feet:** stance/contact.
            - **Orange feet:** swing.
            - **Gray feet:** fallen posture.
            - **Purple arrow:** commanded velocity direction.
            - **Green arrow:** measured velocity direction.
            - **Brown line:** ground profile with optional slope and localized step.
            - **Safety monitor:** reduces speed and crouches when torso tilt exceeds the warning threshold.
            - **Fall detector:** terminates nominal locomotion when tilt exceeds the failure threshold.

            This demo uses a reduced-order, physically plausible humanoid movement model.
            It is intended for visualization, interaction, and early metric sanity checks,
            not as a substitute for full rigid-body simulation or hardware validation.
            """
        )


with tab_metrics:
    if df.empty:
        st.info("No trajectory data available.")
    else:
        st.subheader("Velocity tracking")

        plot_df = df.copy()
        plot_df["tilt_deg"] = np.degrees(plot_df["tilt"])

        st.line_chart(
            plot_df.set_index("time")[
                ["vx_cmd", "vx_meas", "vy_cmd", "vy_meas"]
            ]
        )

        st.subheader("Height and tilt")
        st.line_chart(plot_df.set_index("time")[["com_z", "tilt_deg"]])

        st.subheader("Summary table")
        st.dataframe(pd.DataFrame([metrics]), use_container_width=True)

        st.subheader("Small robustness sweep")
        st.caption(
            "Runs three seeds without video rendering. Useful for quick fall-rate "
            "and tracking sanity checks."
        )

        if st.button("Run 3-seed robustness check"):
            with st.spinner("Running 3 seeds..."):
                rob_df = robustness_sweep(active_params, seeds=(0, 1, 2))

            st.dataframe(rob_df, use_container_width=True)

            fall_rate = float(rob_df["fell"].mean()) if not rob_df.empty else float("nan")
            mean_rmse = (
                float(rob_df["tracking_rmse_m_s"].mean())
                if not rob_df.empty
                else float("nan")
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Fall rate", _fmt(100.0 * fall_rate, " %", 1))
            c2.metric("Mean tracking RMSE", _fmt(mean_rmse, " m/s", 3))
            c3.metric("Seeds", "3")


with tab_failure:
    if df.empty:
        st.info("No trajectory data available.")
    else:
        st.subheader("Safety monitor and failure analysis")

        st.markdown(
            f"""
            - Warning tilt threshold: **{math.degrees(float(active_params['tilt_warn_rad'])):.1f} deg**
            - Failure tilt threshold: **{math.degrees(float(active_params['tilt_fail_rad'])):.1f} deg**
            - Safety events in this run: **{metrics.get('safety_events', 0)}**
            - Final status: **{metrics.get('status', 'n/a')}**
            """
        )

        fig, ax = plt.subplots(figsize=(8.5, 3.8), dpi=95)
        ax.plot(df["time"].to_numpy(), np.degrees(df["tilt"].to_numpy()), label="Torso tilt")

        warn_deg = math.degrees(float(active_params["tilt_warn_rad"]))
        fail_deg = math.degrees(float(active_params["tilt_fail_rad"]))

        ax.axhline(warn_deg, color="orange", linestyle="--", label="Warning")
        ax.axhline(-warn_deg, color="orange", linestyle="--")
        ax.axhline(fail_deg, color="red", linestyle="--", label="Fall threshold")
        ax.axhline(-fail_deg, color="red", linestyle="--")

        ax.set_xlabel("time (s)")
        ax.set_ylabel("tilt (deg)")
        ax.set_title("Tilt-based fall monitoring")
        ax.grid(alpha=0.3)
        ax.legend()

        st.pyplot(fig, clear_figure=True)

        st.subheader("Event log")
        event_mask = df["safety_active"] | df["push_active"] | df["fall"]
        event_cols = [
            "time",
            "vx_meas",
            "vy_meas",
            "tilt",
            "safety_active",
            "push_active",
            "fall",
        ]
        event_df = df.loc[event_mask, event_cols].tail(80)
        st.dataframe(event_df, use_container_width=True)

        st.subheader("Known failure modes represented in this demo")
        st.markdown(
            """
            1. **Toe stubbing on steps**  
               Occurs when step height is too large relative to forward speed or foot clearance.

            2. **Slippery-floor degradation**  
               Lower friction increases velocity noise and reduces tracking gain, making push recovery harder.

            3. **Overcorrection after push**  
               The safety monitor reduces speed and crouches, but excessive tilt can still cross the fall threshold.

            4. **Latency and observation noise stress**  
               Higher noise degrades velocity estimate quality and can induce oscillatory correction.

            This demo does not claim full sim-to-real transfer. It is a simulation-only
            visualization and evaluation scaffold.
            """
        )


with tab_repro:
    st.subheader("Run locally")

    st.markdown("Linux/macOS:")
    st.code(
        """
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
        """.strip(),
        language="bash",
    )

    st.markdown("Windows PowerShell:")
    st.code(
        """
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
        """.strip(),
        language="powershell",
    )

    st.markdown("Windows CMD:")
    st.code(
        """
python -m venv .venv
.venv\\Scripts\\activate.bat
pip install -r requirements.txt
streamlit run app.py
        """.strip(),
        language="bat",
    )

    st.info(
        "The app generates the demo video automatically on first launch. "
        "Use the sidebar controls and press **Generate live demo** to update it."
    )

    st.subheader("If video encoding still fails")
    st.markdown(
        """
        - Reinstall dependencies:
          ```bash
          pip install -r requirements.txt
          ```
        - Reduce **Duration** to 6 or 8 seconds.
        - The app will automatically try:
          1. MP4 / H.264
          2. WebM / VP8
          3. Animated GIF
          4. Static trajectory plot
        """
    )