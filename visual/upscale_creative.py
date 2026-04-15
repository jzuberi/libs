import subprocess
import argparse
import os


# -----------------------------
# Core FFmpeg upscaling helpers
# -----------------------------
import math

def upscale_video(input_path, output_path, width=None, height=None, scale_factor=None):

    # Determine target resolution
    if scale_factor:
        scale_expr = f"scale=iw*{scale_factor}:ih*{scale_factor}:flags=lanczos"
        # We need to know the actual numbers to check macroblocks
        # Use ffprobe to get input size
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0", input_path],
            capture_output=True, text=True
        )
        in_w, in_h = map(int, probe.stdout.strip().split(','))
        out_w = int(in_w * scale_factor)
        out_h = int(in_h * scale_factor)

    elif width and height:
        out_w, out_h = width, height
        scale_expr = f"scale={width}:{height}:flags=lanczos"
    else:
        raise ValueError("Provide either scale_factor OR width and height.")

    # Compute macroblocks
    mb_w = math.ceil(out_w / 16)
    mb_h = math.ceil(out_h / 16)
    macroblocks = mb_w * mb_h

    # H.264 Level 5.2 limit
    H264_MB_LIMIT = 139264

    # Decide codec
    if macroblocks > H264_MB_LIMIT:
        codec = "libx265"
        extra_flags = ["-tag:v", "hvc1"]
        print(f"Switching to HEVC: {macroblocks} macroblocks exceeds H.264 limit {H264_MB_LIMIT}")
    else:
        codec = "libx264"
        extra_flags = []
        print(f"Using H.264: {macroblocks} macroblocks within limit {H264_MB_LIMIT}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vf", scale_expr,
        "-c:v", codec,
        "-preset", "slow",
        "-crf", "14",
        "-pix_fmt", "yuv420p",
        "-an",
        *extra_flags,
        output_path
    ]

    print("Running FFmpeg command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Upscaled video saved to: {output_path}")


def upscale_image(input_path, output_path, width=None, height=None, scale_factor=None):
    """
    Upscale an image using FFmpeg with Lanczos scaling + optional enhancement.
    """
    if scale_factor:
        scale_expr = (
            f"scale=iw*{scale_factor}:ih*{scale_factor}:flags=lanczos,"
            f"eq=contrast=1.05:gamma=1.03,"
            f"unsharp=5:5:0.5"
        )
    elif width and height:
        scale_expr = (
            f"scale={width}:{height}:flags=lanczos,"
            f"eq=contrast=1.05:gamma=1.03,"
            f"unsharp=5:5:0.5"
        )
    else:
        raise ValueError("Provide either scale_factor OR width and height.")

    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-vf", scale_expr,
        "-y",
        output_path
    ]

    print("Running FFmpeg command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Upscaled image saved to: {output_path}")


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Upscale images or videos using FFmpeg (Lanczos).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- video subcommand ----
    video_parser = subparsers.add_parser("video", help="Upscale an MP4 video")
    video_parser.add_argument("input", help="Path to input MP4")
    video_parser.add_argument("output", help="Path to output MP4")

    vgroup = video_parser.add_mutually_exclusive_group(required=True)
    vgroup.add_argument("--scale", type=float, help="Scale factor (e.g., 2 for 2×)")
    vgroup.add_argument("--size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
                        help="Output resolution, e.g., --size 3840 2160")

    # ---- image subcommand ----
    image_parser = subparsers.add_parser("image", help="Upscale an image (PNG/JPG/WebP/etc.)")
    image_parser.add_argument("input", help="Path to input image")
    image_parser.add_argument("output", help="Path to output image")

    igroup = image_parser.add_mutually_exclusive_group(required=True)
    igroup.add_argument("--scale", type=float, help="Scale factor (e.g., 2 for 2×)")
    igroup.add_argument("--size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
                        help="Output resolution, e.g., --size 2048 2048")

    args = parser.parse_args()

    if args.command == "video":
        if args.scale:
            upscale_video(args.input, args.output, scale_factor=args.scale)
        else:
            w, h = args.size
            upscale_video(args.input, args.output, width=w, height=h)

    elif args.command == "image":
        if args.scale:
            upscale_image(args.input, args.output, scale_factor=args.scale)
        else:
            w, h = args.size
            upscale_image(args.input, args.output, width=w, height=h)


if __name__ == "__main__":
    main()
